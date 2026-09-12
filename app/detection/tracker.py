from ultralytics import YOLO


class WorkerTracker:
    def __init__(self, model_path: str = "yolo11n.pt"):
        self.model = YOLO(model_path)

    def track(self, frame):
        return self.model.track(
            frame,
            persist=True,
            classes=[0],  # person
            tracker="bytetrack.yaml",
            verbose=False,
        )