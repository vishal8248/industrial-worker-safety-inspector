import threading

import cv2
from dotenv import load_dotenv

from app.config.zone_config import ZoneConfig
from app.detection.tracker import WorkerTracker
from app.detection.zone_monitor import ZoneMonitor
from app.rules.safety_rules import SafetyRules
from app.vlm.analyzer import VLMAnalyzer
from app.workflow.incident_graph import build_incident_graph


VIDEO_PATH = "data/videos/test_video.mp4"
CAMERA_ID = "camera_01"

VLM_INTERVAL_SECONDS = 5.0
NOTIFICATION_COOLDOWN_SECONDS = 120.0


def clean_evidence(evidence):
    cleaned = []

    for item in evidence:
        text = str(item).strip()

        while text.startswith(">"):
            text = text[1:].strip()

        if text:
            cleaned.append(text)

    return cleaned


def main():
    load_dotenv()

    zone_config = ZoneConfig()

    camera_config = zone_config.get_camera_config(
        CAMERA_ID
    )

    machine = camera_config["machine"]
    polygon = camera_config["safety_zone"]
    dwell_threshold = camera_config[
        "dwell_threshold_seconds"
    ]

    tracker = WorkerTracker()
    vlm = VLMAnalyzer()
    safety_rules = SafetyRules()
    incident_graph = build_incident_graph()

    zone_monitor = ZoneMonitor(
        polygon=polygon,
        dwell_threshold=dwell_threshold,
    )

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print()
    print(
        "Industrial Worker Safety Inspector"
    )
    print(
        "----------------------------------"
    )
    print(
        f"Camera: {CAMERA_ID}"
    )
    print(
        f"Machine: {machine}"
    )
    print(
        f"Video resolution: {width}x{height}"
    )
    print(
        f"Kavach dwell threshold: "
        f"{dwell_threshold}s"
    )
    print(
        f"VLM interval: "
        f"{VLM_INTERVAL_SECONDS}s"
    )
    print(
        f"Notification cooldown: "
        f"{NOTIFICATION_COOLDOWN_SECONDS}s"
    )
    print()
    print("Press Q to exit.")
    print()

    vlm_lock = threading.Lock()

    vlm_running = False
    last_vlm_timestamp = -VLM_INTERVAL_SECONDS
    vlm_thread = None

    stop_event = threading.Event()

    last_notification_times = {}

    def run_vlm(
        frame,
        box,
        worker_id,
        timestamp,
    ):
        nonlocal vlm_running

        try:
            analysis = vlm.analyze_worker(
                frame=frame,
                box=box,
                worker_id=worker_id,
                timestamp=timestamp,
            )

            observations = analysis.observations

            print()
            print(
                f"VLM observations - "
                f"Worker {worker_id}"
            )

            print(
                "Phone usage:",
                observations.phone_usage,
            )

            print(
                "PPE violation:",
                observations.ppe_violation,
            )

            print(
                "Machine interaction:",
                observations.machine_interaction,
            )

            print(
                "Unsafe position:",
                observations.unsafe_position,
            )

            evidence = clean_evidence(
                observations.evidence
            )

            if evidence:
                print("Evidence:")

                for item in evidence:
                    print(
                        f"- {item}"
                    )

            rule_result = safety_rules.evaluate(
                observations
            )

            if not rule_result.incident_detected:
                return

            print()
            print(
                "CONFIRMED SAFETY INCIDENT"
            )

            print(
                f"Worker: {worker_id}"
            )

            print(
                f"Time: "
                f"{timestamp:.2f}s"
            )

            print(
                "Types:",
                rule_result.incident_types,
            )

            print("Reasons:")

            for reason in rule_result.reasons:
                print(
                    f"- {reason}"
                )

            if not evidence:
                evidence = [
                    reason
                    for reason in rule_result.reasons
                ]

            for incident_type in (
                rule_result.incident_types
            ):
                incident_key = (
                    worker_id,
                    incident_type,
                )

                last_notification_time = (
                    last_notification_times.get(
                        incident_key
                    )
                )

                if (
                    last_notification_time
                    is not None
                ):
                    elapsed = (
                        timestamp
                        - last_notification_time
                    )

                    if (
                        elapsed
                        < NOTIFICATION_COOLDOWN_SECONDS
                    ):
                        remaining = max(
                            0,
                            NOTIFICATION_COOLDOWN_SECONDS
                            - elapsed,
                        )

                        print()
                        print(
                            "Notification cooldown "
                            "active."
                        )

                        print(
                            f"Worker: {worker_id}"
                        )

                        print(
                            f"Incident: "
                            f"{incident_type}"
                        )

                        print(
                            f"Next notification "
                            f"allowed in: "
                            f"{remaining:.0f}s"
                        )

                        continue

                print()
                print(
                    "Starting LangGraph "
                    "incident workflow..."
                )

                graph_result = (
                    incident_graph.invoke(
                        {
                            "incident_type": (
                                incident_type
                            ),
                            "worker_id": (
                                worker_id
                            ),
                            "timestamp": (
                                timestamp
                            ),
                            "camera_id": (
                                CAMERA_ID
                            ),
                            "machine": (
                                machine
                            ),
                            "evidence": (
                                evidence
                            ),
                        }
                    )
                )

                notification_sent = (
                    graph_result.get(
                        "notification_sent",
                        False,
                    )
                )

                if notification_sent:
                    last_notification_times[
                        incident_key
                    ] = timestamp

                print()
                print(
                    "LANGGRAPH INCIDENT WORKFLOW"
                )

                print(
                    "==========================="
                )

                print()
                print(
                    graph_result[
                        "incident_report"
                    ]
                )

                print()

                if notification_sent:
                    print(
                        "Supervisor "
                        "notification: SENT"
                    )
                else:
                    print(
                        "Supervisor "
                        "notification: NOT SENT"
                    )

        except Exception as exc:
            print()
            print(
                "VLM/incident workflow error:"
            )
            print(exc)

        finally:
            with vlm_lock:
                vlm_running = False

    while not stop_event.is_set():
        success, frame = cap.read()

        if not success:
            break

        timestamp = (
            cap.get(
                cv2.CAP_PROP_POS_MSEC
            )
            / 1000.0
        )

        original_frame = frame.copy()

        results = tracker.track(
            original_frame
        )

        display_frame = frame.copy()

        zone_monitor.draw_zone(
            display_frame
        )

        if results:
            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                ids = (
                    boxes.id
                    if boxes.id is not None
                    else []
                )

                for index, box in enumerate(
                    boxes.xyxy
                ):
                    if index >= len(ids):
                        continue

                    worker_id = int(
                        ids[index]
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box,
                    )

                    x1 = max(
                        0,
                        min(
                            x1,
                            width - 1,
                        ),
                    )

                    y1 = max(
                        0,
                        min(
                            y1,
                            height - 1,
                        ),
                    )

                    x2 = max(
                        0,
                        min(
                            x2,
                            width - 1,
                        ),
                    )

                    y2 = max(
                        0,
                        min(
                            y2,
                            height - 1,
                        ),
                    )

                    foot_x = int(
                        (x1 + x2) / 2
                    )

                    foot_y = y2

                    zone_result = (
                        zone_monitor.update(
                            worker_id=worker_id,
                            point=(
                                foot_x,
                                foot_y,
                            ),
                            timestamp=timestamp,
                        )
                    )

                    if zone_result["event"] is not None:
                        print()
                        print(
                            "KAVACH EVENT"
                        )

                        print(
                            f"Worker: "
                            f"{worker_id}"
                        )

                        print(
                            f"Event: "
                            f"{zone_result['event']}"
                        )

                        print(
                            f"Dwell: "
                            f"{zone_result['dwell_time']:.2f}s"
                        )

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    label = (
                        f"Worker "
                        f"{worker_id}"
                    )

                    if zone_result[
                        "inside_zone"
                    ]:
                        label += (
                            " | IN KAVACH"
                        )

                    cv2.putText(
                        display_frame,
                        label,
                        (
                            x1,
                            max(
                                20,
                                y1 - 10,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    should_run_vlm = (
                        timestamp
                        - last_vlm_timestamp
                        >= VLM_INTERVAL_SECONDS
                    )

                    with vlm_lock:
                        can_start_vlm = (
                            not vlm_running
                        )

                        if (
                            should_run_vlm
                            and can_start_vlm
                        ):
                            vlm_running = True

                    if (
                        should_run_vlm
                        and can_start_vlm
                    ):
                        last_vlm_timestamp = (
                            timestamp
                        )

                        vlm_frame = (
                            original_frame.copy()
                        )

                        vlm_box = (
                            x1,
                            y1,
                            x2,
                            y2,
                        )

                        vlm_thread = (
                            threading.Thread(
                                target=run_vlm,
                                args=(
                                    vlm_frame,
                                    vlm_box,
                                    worker_id,
                                    timestamp,
                                ),
                                daemon=False,
                            )
                        )

                        vlm_thread.start()

        display_width = 1280

        scale = (
            display_width
            / width
        )

        display_height = int(
            height * scale
        )

        resized_frame = cv2.resize(
            display_frame,
            (
                display_width,
                display_height,
            ),
        )

        cv2.imshow(
            "Industrial Worker Safety Inspector",
            resized_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    stop_event.set()

    if vlm_thread is not None:
        vlm_thread.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()