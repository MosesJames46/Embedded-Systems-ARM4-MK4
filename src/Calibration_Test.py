import cv2
from PlaneCalibration import PlaneCalibration

camera = cv2.VideoCapture(3)

success , frame = camera.read()

if not success:
    raise RuntimeError("Could not captuer camera frame.")

calibration = PlaneCalibration()

points = calibration.select_image_points(frame)

print("Selected image points:", points)

camera.release()