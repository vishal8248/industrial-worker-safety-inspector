from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


@dataclass
class WorkerZoneState:
    inside: bool = False
    entered_at: float | None = None
    dwell_time: float = 0.0


class ZoneMonitor:
    def __init__(
        self,
        polygon: List[Point],
        dwell_threshold: float = 5.0,
    ):
        self.polygon = polygon
        self.dwell_threshold = dwell_threshold
        self.worker_states: Dict[int, WorkerZoneState] = {}

    def is_inside(self, point: Point) -> bool:
        polygon = np.array(
            self.polygon,
            dtype=np.int32,
        )

        result = cv2.pointPolygonTest(
            polygon,
            point,
            False,
        )

        return result >= 0

    def update(
        self,
        worker_id: int,
        point: Point,
        timestamp: float,
    ) -> dict:
        inside = self.is_inside(point)

        state = self.worker_states.setdefault(
            worker_id,
            WorkerZoneState(),
        )

        if inside and not state.inside:
            state.inside = True
            state.entered_at = timestamp
            state.dwell_time = 0.0

        elif inside and state.inside:
            if state.entered_at is not None:
                state.dwell_time = (
                    timestamp - state.entered_at
                )

        elif not inside and state.inside:
            if state.entered_at is not None:
                state.dwell_time = (
                    timestamp - state.entered_at
                )

            state.inside = False
            state.entered_at = None

        violation = (
            state.inside
            and state.dwell_time >= self.dwell_threshold
        )

        return {
            "worker_id": worker_id,
            "inside_zone": state.inside,
            "dwell_time": round(
                state.dwell_time,
                2,
            ),
            "violation": violation,
            "timestamp": timestamp,
        }

    def draw_zone(self, frame):
        polygon = np.array(
            self.polygon,
            dtype=np.int32,
        )

        cv2.polylines(
            frame,
            [polygon],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2,
        )

        return frame