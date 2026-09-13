from typing import List, Tuple


Point = Tuple[int, int]


class SafetyZoneGenerator:
    def __init__(
        self,
        padding: int = 30,
    ):
        self.padding = padding

    def generate(
        self,
        bbox: List[int],
        frame_width: int,
        frame_height: int,
    ) -> List[Point]:
        x1, y1, x2, y2 = bbox

        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)

        x2 = min(
            frame_width - 1,
            x2 + self.padding,
        )

        y2 = min(
            frame_height - 1,
            y2 + self.padding,
        )

        return [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]