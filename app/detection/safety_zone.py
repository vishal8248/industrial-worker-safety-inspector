from typing import List, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


class SafetyZoneGenerator:
    def __init__(
        self,
        padding: int = 35,
    ):
        self.padding = padding

    def from_mask(
        self,
        mask: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> List[Point]:
        mask = (mask > 0).astype(np.uint8) * 255

        kernel_size = (
            self.padding * 2 + 1
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size,
            ),
        )

        expanded_mask = cv2.dilate(
            mask,
            kernel,
            iterations=1,
        )

        expanded_mask = cv2.morphologyEx(
            expanded_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            expanded_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return []

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        contour = contour.reshape(
            -1,
            2,
        )

        points = []

        for x, y in contour:
            x = max(
                0,
                min(
                    int(x),
                    frame_width - 1,
                ),
            )

            y = max(
                0,
                min(
                    int(y),
                    frame_height - 1,
                ),
            )

            points.append(
                (x, y)
            )

        return points