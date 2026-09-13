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

        self.frames = deque(maxlen=max_frames)

    def add(
        self,
        frame,
        timestamp,
        worker_id=None,
        bbox=None,
        foot_point=None,
        inside_zone=False,
        dwell_time=0.0,
        violation=False,
    ):
        self.frames.append(
            {
                "frame": frame.copy(),
                "timestamp": timestamp,
                "worker_id": worker_id,
                "bbox": bbox,
                "foot_point": foot_point,
                "inside_zone": inside_zone,
                "dwell_time": dwell_time,
                "violation": violation,
            }
        )

    def get_frames(self):
        return list(self.frames)

    def get_event_frames(
        self,
        entry_time,
        violation_time,
        exit_time,
        count=8,
    ):
        frames = list(self.frames)

        if not frames:
            return []

        event_frames = []

        for item in frames:
            timestamp = item["timestamp"]

            if timestamp <= entry_time:
                event_frames.append(item)

            elif (
                violation_time > 0
                and timestamp >= violation_time
            ):
                event_frames.append(item)

            elif (
                exit_time is not None
                and timestamp >= exit_time
            ):
                event_frames.append(item)

        if not event_frames:
            event_frames = frames

        if len(event_frames) <= count:
            return event_frames

        step = (len(event_frames) - 1) / (count - 1)

        selected = []

        for i in range(count):
            index = round(i * step)
            selected.append(event_frames[index])

        return selected

    def clear(self):
        self.frames.clear()

    def save(
        self,
        output_dir,
        incident_id,
        entry_time,
        violation_time,
        exit_time,
        count=8,
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

        frames = self.get_event_frames(
            entry_time=entry_time,
            violation_time=violation_time,
            exit_time=exit_time,
            count=count,
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