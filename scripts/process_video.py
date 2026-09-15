import threading

import cv2
from dotenv import load_dotenv

from app.config.zone_config import ZoneConfig
from app.detection.tracker import WorkerTracker
from app.detection.zone_monitor import ZoneMonitor
from app.rules.safety_rules import SafetyRules
from app.vlm.analyzer import VLMAnalyzer


VIDEO_PATH = "data/videos/test_video.mp4"
CAMERA_ID = "camera_01"

VLM_INTERVAL_SECONDS = 5.0


def open_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {path}"
        )

    return cap


def draw_kavach(frame, polygon):
    points = [
        tuple(point)
        for point in polygon
    ]

    for index in range(len(points)):
        start = points[index]
        end = points[
            (index + 1) % len(points)
        ]

        cv2.line(
            frame,
            start,
            end,
            (0, 0, 255),
            3,
        )

    return frame


def draw_worker(
    frame,
    box,
    worker_id,
    zone_status,
):
    x1, y1, x2, y2 = box

    foot_point = (
        int((x1 + x2) / 2),
        int(y2),
    )

    if zone_status["inside_zone"]:
        label = (
            f"WORKER {worker_id} | "
            f"KAVACH "
            f"{zone_status['dwell_time']:.1f}s"
        )

        if zone_status["violation"]:
            label = (
                f"WORKER {worker_id} | "
                "VIOLATION"
            )

        box_color = (0, 0, 255)

    else:
        label = f"WORKER {worker_id}"
        box_color = (0, 255, 0)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        box_color,
        2,
    )

    cv2.circle(
        frame,
        foot_point,
        6,
        (255, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(y1 - 10, 25),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        box_color,
        2,
    )

    return frame


def run_vlm_analysis(
    vlm,
    safety_rules,
    frame,
    box,
    worker_id,
    timestamp,
):
    try:
        analysis = vlm.analyze_worker(
            frame=frame,
            box=box,
            worker_id=worker_id,
            timestamp=timestamp,
        )

        observations = analysis.observations

        rule_result = safety_rules.evaluate(
            observations
        )

        print()
        print(
            f"VLM observations - "
            f"Worker {worker_id}"
        )

        print(
            f"Phone usage: "
            f"{observations.phone_usage}"
        )

        print(
            f"PPE violation: "
            f"{observations.ppe_violation}"
        )

        print(
            f"Machine interaction: "
            f"{observations.machine_interaction}"
        )

        print(
            f"Unsafe position: "
            f"{observations.unsafe_position}"
        )

        if observations.evidence:
            print("Evidence:")

            for evidence in observations.evidence:
                print(
                    f"- {evidence}"
                )

        if rule_result.incident_detected:
            print()
            print(
                "CONFIRMED SAFETY INCIDENT"
            )

            print(
                f"Worker: {worker_id}"
            )

            print(
                f"Time: {timestamp:.2f}s"
            )

            print(
                f"Types: "
                f"{rule_result.incident_types}"
            )

            print("Reasons:")

            for reason in rule_result.reasons:
                print(
                    f"- {reason}"
                )

        print()

    except Exception as error:
        print(
            f"VLM analysis failed for "
            f"worker {worker_id}: {error}"
        )


def main():
    load_dotenv()

    zone_config = ZoneConfig()

    camera_config = (
        zone_config.get_camera_config(
            CAMERA_ID
        )
    )

    polygon = [
        tuple(point)
        for point in camera_config[
            "safety_zone"
        ]
    ]

    dwell_threshold = camera_config[
        "dwell_threshold_seconds"
    ]

    zone_monitor = ZoneMonitor(
        polygon=polygon,
        dwell_threshold=dwell_threshold,
    )

    tracker = WorkerTracker()
    vlm = VLMAnalyzer()
    safety_rules = SafetyRules()

    cap = open_video(VIDEO_PATH)

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    video_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    video_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print()
    print(
        "Industrial Worker Safety Inspector"
    )
    print("----------------------------------")
    print(
        f"Camera: {CAMERA_ID}"
    )
    print(
        f"Machine: "
        f"{camera_config['machine']}"
    )
    print(
        f"Video resolution: "
        f"{video_width}x{video_height}"
    )
    print(
        f"Kavach dwell threshold: "
        f"{dwell_threshold}s"
    )
    print(
        f"VLM interval: "
        f"{VLM_INTERVAL_SECONDS}s"
    )
    print()
    print("Press Q to exit.")
    print()

    window_name = (
        "Industrial Worker Safety Inspector"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    display_width = 1280

    aspect_ratio = (
        video_height / video_width
    )

    display_height = int(
        display_width * aspect_ratio
    )

    cv2.resizeWindow(
        window_name,
        display_width,
        display_height,
    )

    last_vlm_time = (
        -VLM_INTERVAL_SECONDS
    )

    vlm_running = False

    vlm_lock = threading.Lock()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number = cap.get(
            cv2.CAP_PROP_POS_FRAMES
        )

        timestamp = (
            frame_number / fps
        )

        results = tracker.track(
            frame
        )

        result = results[0]

        workers = []

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
                x1, y1, x2, y2 = (
                    box.astype(int)
                )

                point = (
                    int((x1 + x2) / 2),
                    int(y2),
                )

                zone_status = (
                    zone_monitor.update(
                        worker_id=worker_id,
                        point=point,
                        timestamp=timestamp,
                    )
                )

                if zone_status["event"] is not None:
                    print(
                        f"[{timestamp:.2f}s] "
                        f"WORKER {worker_id}: "
                        f"{zone_status['event']}"
                    )

                workers.append(
                    {
                        "worker_id": worker_id,
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                        "zone_status": zone_status,
                    }
                )

                frame = draw_worker(
                    frame=frame,
                    box=(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    worker_id=worker_id,
                    zone_status=zone_status,
                )

        if (
            timestamp - last_vlm_time
            >= VLM_INTERVAL_SECONDS
            and workers
        ):
            with vlm_lock:
                if not vlm_running:
                    selected_worker = workers[0]

                    analysis_frame = frame.copy()

                    worker_id = (
                        selected_worker[
                            "worker_id"
                        ]
                    )

                    worker_box = (
                        selected_worker[
                            "box"
                        ]
                    )

                    last_vlm_time = timestamp
                    vlm_running = True

                    def vlm_task():
                        nonlocal vlm_running

                        try:
                            run_vlm_analysis(
                                vlm=vlm,
                                safety_rules=safety_rules,
                                frame=analysis_frame,
                                box=worker_box,
                                worker_id=worker_id,
                                timestamp=timestamp,
                            )
                        finally:
                            with vlm_lock:
                                vlm_running = False

                    thread = threading.Thread(
                        target=vlm_task,
                        daemon=True,
                    )

                    thread.start()

        frame = draw_kavach(
            frame,
            polygon,
        )

        cv2.imshow(
            window_name,
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()