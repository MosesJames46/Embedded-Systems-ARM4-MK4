
import cv2
import numpy as np


class RobotCalibration:
    def __init__(self):
        self.matrix = None

    def calibrate(self, workspace_points, robot_points):
        """
        workspace_points: [(x, y), ...]
        robot_points: [(X, Y), ...]

        Both lists must contain 3 corresponding points.
        """
        if len(workspace_points) != 3:
            raise ValueError(
                "Exactly 3 workspace points are required."
            )

        if len(robot_points) != 3:
            raise ValueError(
                "Exactly 3 robot points are required."
            )

        workspace = np.float32(workspace_points)
        robot = np.float32(robot_points)

        self.matrix = cv2.getAffineTransform(
            workspace,
            robot
        )

    def workspace_to_robot(self, x, y):
        if self.matrix is None:
            raise RuntimeError(
                "Robot calibration has not been performed."
            )

        point = np.array([x, y, 1.0])

        result = self.matrix @ point

        return float(result[0]), float(result[1])
