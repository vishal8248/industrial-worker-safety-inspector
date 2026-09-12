import cv2

from app.detection.tracker import WorkerTracker
from app.detection.zone_monitor import ZoneMonitor
from app.detection.zone_selector import ZoneSelector
from app.incidents.incident_manager import IncidentManager
from app.incidents.frame_buffer import FrameBuffer


VIDEO_PATH = "data/videos/test_video.mp4"
OUTPUT_DIR = "data/incident_frames"
DWELL_THRESHOLD = 5.0
BUFFER_SECONDS = 5.0


def open_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    return cap


def select_zone(cap):
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Could not read video frame.")

    selector = ZoneSelector(frame)
    zone = selector.select()

    if len(zone) < 3:
        raise RuntimeError(
            "A valid zone requires at least 3 points."
        )

    return zone


def main():
    cap = open_video(VIDEO_PATH)

    zone = select_zone(cap)

    print("\nSafety zone selected:")
    print(zone)

    cap.release()
    cap = open_video(VIDEO_PATH)

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

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        timestamp = frame_number / fps

        detected_worker_ids = set()

        results = tracker.track(frame)
        result = results[0]

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            for box, worker_id in zip(boxes, track_ids):
                detected_worker_ids.add(worker_id)

                x1, y1, x2, y2 = box.astype(int)

                point = (
                    int((x1 + x2) / 2),
                    int(y2),
                )

                if worker_id not in frame_buffers:
                    frame_buffers[worker_id] = FrameBuffer(
                        buffer_seconds=BUFFER_SECONDS,
                        fps=fps,
                    )

                frame_buffers[worker_id].add(
                    frame,
                    timestamp,
                )

                status = zone_monitor.update(
                    worker_id=worker_id,
                    point=point,
                    timestamp=timestamp,
                )

                incident = incident_manager.update(status)

                if incident is not None:
                    incident_id = (
                        f"{incident.worker_id}_"
                        f"{int(incident.violation_time)}"
                    )

                    saved_paths = frame_buffers[
                        worker_id
                    ].save(
                        OUTPUT_DIR,
                        incident_id,
                    )

                    print("\nIncident completed:")
                    print(incident)

                    print("Saved frames:")

                    for path in saved_paths:
                        print(path)

                    frame_buffers[worker_id].clear()

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
                    label = f"WORKER {worker_id}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        for worker_id in list(
            incident_manager.active_incidents
        ):
            if worker_id not in detected_worker_ids:
                incident = incident_manager.mark_missing(
                    worker_id
                )

                if incident is not None:
                    print(
                        "\nIncident closed after "
                        "tracking loss:"
                    )
                    print(incident)

        frame = zone_monitor.draw_zone(frame)

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

    for incident in incident_manager.get_completed_incidents():
        print(incident)


if __name__ == "__main__":
    main()