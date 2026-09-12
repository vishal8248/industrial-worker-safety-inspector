import cv2

from app.detection.zone_selector import ZoneSelector


VIDEO_PATH = "data/videos/test_video.mp4"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    ret, frame = cap.read()

    cap.release()

    if not ret:
        raise RuntimeError(
            "Could not read first video frame."
        )

    selector = ZoneSelector(frame)

    zone = selector.select()

    if not zone:
        print("Zone selection cancelled.")
        return

    print("\nSelected Safety Zone:")
    print(zone)


if __name__ == "__main__":
    main()