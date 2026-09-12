from collections import deque
from pathlib import Path

import cv2


class FrameBuffer:
    def __init__(
        self,
        buffer_seconds: float = 5.0,
        fps: float = 30.0,
    ):
        max_frames = max(1, int(buffer_seconds * fps))

        self.frames = deque(
            maxlen=max_frames
        )

    def add(self, frame, timestamp):
        self.frames.append(
            {
                "timestamp": timestamp,
                "frame": frame.copy(),
            }
        )

    def get_frames(self):
        return list(self.frames)

    def get_representative_frames(
        self,
        count: int = 8,
    ):
        frames = list(self.frames)

        if not frames:
            return []

        if len(frames) <= count:
            return frames

        step = (len(frames) - 1) / (count - 1)

        selected = []

        for i in range(count):
            index = round(i * step)
            selected.append(frames[index])

        return selected

    def clear(self):
        self.frames.clear()

    def save(
        self,
        output_dir,
        incident_id,
        count: int = 8,
    ):
        output_path = Path(output_dir)
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        incident_dir = (
            output_path / f"incident_{incident_id}"
        )

        incident_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        frames = self.get_representative_frames(
            count=count
        )

        saved_paths = []

        for index, item in enumerate(frames):
            filename = (
                f"frame_{index:02d}_"
                f"{item['timestamp']:.2f}s.jpg"
            )

            path = incident_dir / filename

            cv2.imwrite(
                str(path),
                item["frame"],
            )

            saved_paths.append(str(path))

        return saved_paths