import cv2
from ArUcoCalibration import ArUcoCalibration

marker_positions = {
    0: (0, 0),
    1: (200, 0),
    2: (200, 200),
    3: (0, 200)
}

calibration = ArUcoCalibration(marker_positions)

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    
    if not sucess:
        break