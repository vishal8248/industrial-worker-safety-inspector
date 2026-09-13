import os

import cv2
from dotenv import load_dotenv

from app.vlm.analyzer import VLMAnalyzer


VIDEO_PATH = "data/videos/test_video.mp4"
OUTPUT_DIR = "outputs/vlm_startup_frames"


def main():
    load_dotenv()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    frames = []

    target_times = [0.5, 1.5, 2.5]

    for index, target_time in enumerate(target_times):
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            target_time * 1000,
        )

        ret, frame = cap.read()

        if not ret:
            continue

        path = (
            f"{OUTPUT_DIR}/frame_{index}.jpg"
        )

        cv2.imwrite(path, frame)

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

    print("\nSaved startup frames:")

    for index in range(len(frames)):
        print(
            f"{OUTPUT_DIR}/frame_{index}.jpg"
        )


if __name__ == "__main__":
    main()