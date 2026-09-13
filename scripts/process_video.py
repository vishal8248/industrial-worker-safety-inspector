import cv2
from dotenv import load_dotenv

from app.detection.tracker import WorkerTracker
from app.detection.zone_monitor import ZoneMonitor
from app.detection.zone_selector import ZoneSelector
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


def select_zone(cap):
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError(
            "Could not read video frame."
        )

    selector = ZoneSelector(frame)
    zone = selector.select()

    if len(zone) < 3:
        raise RuntimeError(
            "A valid zone requires at least 3 points."
        )

    return zone


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
                f"({machine.location})"
            )

    return scene_context


def main():
    load_dotenv()

    cap = open_video(VIDEO_PATH)

    zone = select_zone(cap)

    print("\nSafety zone selected:")
    print(zone)

    cap.release()

    cap = open_video(VIDEO_PATH)

    scene_context = analyze_scene(cap)

    if scene_context is None:
        print(
            "\nContinuing without scene context."
        )

    tracker = WorkerTracker()

    zone_monitor = ZoneMonitor(
        polygon=zone,
        dwell_threshold=DWELL_THRESHOLD,
    )

    incident_manager = IncidentManager(
        max_missing_frames=15
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

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

                status = zone_monitor.update(
                    worker_id=worker_id,
                    point=point,
                    timestamp=timestamp,
                )

                frame_buffers[worker_id].add(
                    frame=frame,
                    timestamp=timestamp,
                    worker_id=worker_id,
                    bbox=(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    foot_point=point,
                    inside_zone=status[
                        "inside_zone"
                    ],
                    dwell_time=status[
                        "dwell_time"
                    ],
                    violation=status[
                        "violation"
                    ],
                )

                incident = (
                    incident_manager.update(
                        status
                    )
                )

                if incident is not None:
                    incident_id = (
                        f"{incident.worker_id}_"
                        f"{int(incident.violation_time)}"
                    )

                    saved_paths = (
                        frame_buffers[
                            worker_id
                        ].save(
                            output_dir=OUTPUT_DIR,
                            incident_id=incident_id,
                            entry_time=(
                                incident.start_time
                            ),
                            violation_time=(
                                incident.violation_time
                            ),
                            exit_time=(
                                incident.end_time
                            ),
                        )
                    )

                    print(
                        "\nIncident completed:"
                    )
                    print(incident)

                    print("Saved frames:")

                    for path in saved_paths:
                        print(path)

                    frame_buffers[
                        worker_id
                    ].clear()

                if status["violation"]:
                    label = (
                        f"WORKER {worker_id} | "
                        f"UNSAFE | "
                        f"{status['dwell_time']:.1f}s"
                    )

                elif status["inside_zone"]:
                    label = (
                        f"WORKER {worker_id} | "
                        f"INSIDE | "
                        f"{status['dwell_time']:.1f}s"
                    )

                else:
                    label = (
                        f"WORKER {worker_id}"
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
                    label,
                    (
                        x1,
                        max(y1 - 10, 20),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
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

        frame = zone_monitor.draw_zone(
            frame
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

    print("\nCompleted incidents:")

    for incident in (
        incident_manager
        .get_completed_incidents()
    ):
        print(incident)


if __name__ == "__main__":
    main()