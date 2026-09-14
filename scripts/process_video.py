import cv2
from dotenv import load_dotenv

from app.detection.tracker import WorkerTracker
from app.detection.safety_zone import SafetyZoneGenerator
from app.incidents.incident_manager import IncidentManager
from app.incidents.frame_buffer import FrameBuffer
from app.vlm.analyzer import VLMAnalyzer


VIDEO_PATH = "data/videos/test_video.mp4"
OUTPUT_DIR = "data/incident_frames"

DWELL_THRESHOLD = 5.0
BUFFER_SECONDS = 5.0

STARTUP_TIMES = [0.5, 1.5, 2.5]


def open_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {path}"
        )

    return cap


def extract_startup_frames(cap):
    frames = []

    for target_time in STARTUP_TIMES:
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            target_time * 1000,
        )

        ret, frame = cap.read()

        if ret:
            frames.append(
                {
                    "timestamp": target_time,
                    "frame": frame,
                }
            )

    return frames


def analyze_scene(cap):
    print("\nAnalyzing startup scene...")

    startup_frames = extract_startup_frames(cap)

    if not startup_frames:
        print(
            "Could not extract startup frames."
        )
        return None

    analyzer = VLMAnalyzer()

    scene_context = analyzer.analyze_startup_frames(
        startup_frames
    )

    print("\nDetected fixed equipment:")

    if not scene_context.machines:
        print("No fixed equipment detected.")
    else:
        for index, machine in enumerate(
            scene_context.machines,
            start=1,
        ):
            print(
                f"machine_{index}: "
                f"{machine.machine_type} "
                f"({machine.location}) "
                f"bbox={machine.bbox}"
            )

    return scene_context


def generate_safety_zones(
    scene_context,
    frame_width,
    frame_height,
):
    generator = SafetyZoneGenerator(
        side_padding=35,
        front_padding=90,
        top_padding=15,
    )

    zones = []

    if scene_context is None:
        return zones

    for index, machine in enumerate(
        scene_context.machines,
        start=1,
    ):
        zone = generator.generate(
            bbox=machine.bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        zones.append(
            {
                "machine_id": f"machine_{index}",
                "machine_type": machine.machine_type,
                "bbox": machine.bbox,
                "zone": zone,
            }
        )

    return zones


def draw_machine_zones(
    frame,
    machine_zones,
):
    for machine in machine_zones:
        x1, y1, x2, y2 = machine["bbox"]

        zone = machine["zone"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2,
        )

        for index in range(
            len(zone)
        ):
            start = zone[index]

            end = zone[
                (index + 1) % len(zone)
            ]

            cv2.line(
                frame,
                start,
                end,
                (0, 0, 255),
                2,
            )

        label = (
            f'{machine["machine_id"]} | '
            f'{machine["machine_type"]}'
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(y1 - 10, 20),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )

    return frame


def main():
    load_dotenv()

    cap = open_video(VIDEO_PATH)

    scene_context = analyze_scene(cap)

    if scene_context is None:
        print(
            "\nContinuing without scene context."
        )

    ret, reference_frame = cap.read()

    if not ret:
        cap.release()

        raise RuntimeError(
            "Could not read reference frame."
        )

    frame_height, frame_width = (
        reference_frame.shape[:2]
    )

    machine_zones = generate_safety_zones(
        scene_context=scene_context,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    print("\nGenerated safety zones:")

    for machine in machine_zones:
        print(
            f'{machine["machine_id"]}: '
            f'{machine["machine_type"]}'
        )

        print(
            f'  machine bbox: '
            f'{machine["bbox"]}'
        )

        print(
            f'  safety zone: '
            f'{machine["zone"]}'
        )

    cap.release()

    cap = open_video(VIDEO_PATH)

    tracker = WorkerTracker()

    incident_manager = IncidentManager(
        max_missing_frames=15
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    frame_buffers = {}

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number = cap.get(
            cv2.CAP_PROP_POS_FRAMES
        )

        timestamp = frame_number / fps

        detected_worker_ids = set()

        results = tracker.track(frame)
        result = results[0]

        if result.boxes.id is not None:
            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            for box, worker_id in zip(
                boxes,
                track_ids,
            ):
                detected_worker_ids.add(
                    worker_id
                )

                x1, y1, x2, y2 = (
                    box.astype(int)
                )

                point = (
                    int((x1 + x2) / 2),
                    int(y2),
                )

                if worker_id not in frame_buffers:
                    frame_buffers[worker_id] = (
                        FrameBuffer(
                            buffer_seconds=BUFFER_SECONDS,
                            fps=fps,
                        )
                    )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.circle(
                    frame,
                    point,
                    5,
                    (255, 0, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"WORKER {worker_id}",
                    (
                        x1,
                        max(y1 - 10, 20),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        for worker_id in list(
            incident_manager.active_incidents
        ):
            if (
                worker_id
                not in detected_worker_ids
            ):
                incident = (
                    incident_manager.mark_missing(
                        worker_id
                    )
                )

                if incident is not None:
                    print(
                        "\nIncident closed after "
                        "tracking loss:"
                    )

                    print(incident)

        frame = draw_machine_zones(
            frame,
            machine_zones,
        )

        cv2.imshow(
            "Industrial Worker Safety Inspector",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()