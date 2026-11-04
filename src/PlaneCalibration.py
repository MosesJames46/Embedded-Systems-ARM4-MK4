import cv2
import numpy as np
import json

class PlaneCalibration:
    def __init__(self):
        self.image_points = []
        self.robot_points = []
        self.transform = None

    def select_image_points(self, frame):
        self.image_points = []
        display = frame.copy()

        def on_mouse(event, x, y, flags, userdata):
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            if len(self.image_points) >= 4:
                return

            self.image_points.append((x, y))

            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                display,
                str(len(self.image_points)),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        window = "Select Calibration Points"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_mouse)

        print("Click four calibration points:")
        print("1. Top-left")
        print("2. Top-right")
        print("3. Bottom-right")
        print("4. Bottom-left")
        print("Press ESC to cancel.")

        while len(self.image_points) < 4:
            cv2.imshow(window, display)

            if cv2.waitKey(20) & 0xFF == 27:
                break

        cv2.destroyWindow(window)

        if len(self.image_points) != 4:
            raise RuntimeError("Calibration cancelled.")

        return self.image_points

    def calculate_transform(self, robot_points):
        if len(self.image_points) != 4:
            raise ValueError("Four image points are required.")

        if len(robot_points) != 4:
            raise ValueError("Four robot points are required.")

        self.robot_points = robot_points

        image = np.array(
            self.image_points, dtype=np.float32
        )

        robot = np.array(
            robot_points, dtype=np.float32
        )

        self.transform = cv2.getPerspectiveTransform(
            image, robot
        )

        return self.transform

    def pixel_to_robot(self, pixel):
        if self.transform is None:
            raise RuntimeError("Calibration is not complete.")

        point = np.array(
            [[pixel]], dtype=np.float32
        )

        result = cv2.perspectiveTransform(
            point, self.transform
        )

        x, y = result[0][0]

        return float(x), float(y)
    
def save_callibration(self, filename = "calibration.json"):
    data = {
        "image_points": self.image_points,
        "robot_points" : self.robot_points
    }

    with open(filename, "w") as file:
        json.dump(data, file, indent = 4)

def load_calibration(self, filename = "calibration.json"):
    with open(filename, "r") as file:
        data = json.load(file)

    self.image_points = data["image_points"]
    self.calculate_transform(data["robot_points"])