from ultralytics import YOLO


class WorkerDetector:
    def __init__(self, model_path: str = "yolo11n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, classes=[0], verbose=False)
        return results