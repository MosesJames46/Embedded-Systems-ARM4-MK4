import cv2
import numpy as np

class ArUcoCalibration:
    """Maps camera pixels to a measured planar marker coordinate system (mm)."""
    def __init__(self, marker_centers_mm, dictionary_id=cv2.aruco.DICT_4X4_50):
        if len(marker_centers_mm) < 4:
            raise ValueError('Provide at least four measured marker centers')
        self.marker_centers_mm = {int(k): tuple(map(float, v)) for k, v in marker_centers_mm.items()}
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())
        self.H = None
        self.visible_ids = []

    def update(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        self.H = None
        self.visible_ids = []
        if ids is None:
            return False
        image_points = []
        plane_points = []
        for corner, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in self.marker_centers_mm:
                continue
            center = corner.reshape(4, 2).mean(axis=0)
            image_points.append(center)
            plane_points.append(self.marker_centers_mm[marker_id])
            self.visible_ids.append(marker_id)
        if len(image_points) < 4:
            return False
        image_points = np.asarray(image_points, dtype=np.float32)
        plane_points = np.asarray(plane_points, dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(image_points[:4], plane_points[:4]) if len(image_points) == 4 else cv2.findHomography(image_points, plane_points)[0]
        return self.H is not None

    def pixel_to_workspace(self, pixel):
        if self.H is None:
            raise RuntimeError('No valid marker calibration for current frame')
        point = np.asarray([[pixel]], dtype=np.float32)
        xy = cv2.perspectiveTransform(point, self.H)[0, 0]
        return float(xy[0]), float(xy[1])

