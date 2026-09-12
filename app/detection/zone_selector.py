import cv2


class ZoneSelector:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.points = []

        self.window_name = "Draw Safety Zone"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))

    def select(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(
            self.window_name,
            self.mouse_callback,
        )

        while True:
            display = self.frame.copy()

            # Draw selected points
            for point in self.points:
                cv2.circle(
                    display,
                    point,
                    5,
                    (0, 255, 0),
                    -1,
                )

            # Draw lines between points
            if len(self.points) > 1:
                for i in range(len(self.points) - 1):
                    cv2.line(
                        display,
                        self.points[i],
                        self.points[i + 1],
                        (0, 255, 0),
                        2,
                    )

            # Close polygon visually
            if len(self.points) >= 3:
                cv2.line(
                    display,
                    self.points[-1],
                    self.points[0],
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                display,
                "Click points | ENTER = confirm | R = reset | ESC = cancel",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                self.window_name,
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            # ENTER
            if key == 13:
                if len(self.points) >= 3:
                    break

            # Reset
            elif key in (ord("r"), ord("R")):
                self.points.clear()

            # Cancel
            elif key == 27:
                self.points.clear()
                break

        cv2.destroyWindow(self.window_name)

        return self.points