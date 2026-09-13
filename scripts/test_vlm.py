import os

import cv2
from dotenv import load_dotenv

from app.vlm.analyzer import VLMAnalyzer


VIDEO_PATH = "data/videos/test_video.mp4"


def main():
    load_dotenv()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    frames = []

    target_times = [0.5, 1.5, 2.5]

    for target_time in target_times:
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            target_time * 1000,
        )

        ret, frame = cap.read()

        if ret:
            frames.append(
                {
                    "timestamp": target_time,
                    "frame": frame,
                }
            )

    cap.release()

    if not frames:
        raise RuntimeError(
            "Could not extract startup frames."
        )

    print(
        f"Sending {len(frames)} frames to VLM..."
    )

    analyzer = VLMAnalyzer()

    result = analyzer.analyze_startup_frames(
        frames
    )

    print("\nVLM result:")
    print(result)


if __name__ == "__main__":
    main()