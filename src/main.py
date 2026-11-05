import cv2
from ColorPicker import ColorPicker
from ColorPicker import pygame
from ColorPicker import pygame_widgets
from Camera import Camera
from ArUcoCalibration import ArUcoCalibration

detector = ColorDetector(ColorPicker(), Camera(3))

cv2.namedWindow(detector.window_name)
cv2.setMouseCallback(detector.window_name, detector.mouse_callback)

# Your existing Camera, ColorPicker, and
# ColorDetector initialization goes above this.

marker_positions = {
    4: (0, 0),
    5: (200, 0),
    6: (200, 200),
    7: (0, 200)
}

calibration = ArUcoCalibration(marker_positions)

# Select the red cube's color once.
detector.color_range = detector.calibrate_color()

if detector.color_range is None:
    raise RuntimeError("Color calibration cancelled")

try:
    while True:
        # Read ONE frame from the existing camera.
        frame = detector.camera.read()

        if frame is None:
            print("Camera frame unavailable")
            break

        # Step 1: Find ArUco markers.
        calibrated = calibration.update(frame)

        # Step 2: Find the cube.
        display_frame = detector.process_frame(frame)

        # Step 3: Get cube's pixel coordinates.
        center = detector.get_tracked_center()

        # Step 4: Convert pixels to workspace mm.
        if (
            calibrated
            and center is not None
            and detector.is_target_stable()
        ):
            x_mm, y_mm = calibration.pixel_to_workspace(
                center
            )

            cv2.putText(
                display_frame,
                f"X: {x_mm:.1f} mm  Y: {y_mm:.1f} mm",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Vision + Calibration", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    detector.camera.release()
    cv2.destroyAllWindows()
