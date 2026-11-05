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

    if not success:
        break

    # Detect markers and update calibration
    calibrated = calibration.update(frame)

    if calibrated:
        cv2.putText(
            frame,
            "CALIBRATED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("ArUco Calibration", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()