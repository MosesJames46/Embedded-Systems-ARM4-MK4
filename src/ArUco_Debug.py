
import cv2

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

detector = cv2.aruco.ArucoDetector(
    dictionary,
    cv2.aruco.DetectorParameters()
)

camera = cv2.VideoCapture(3)

previous_ids = None

try:
    if not camera.isOpened():
        raise RuntimeError("Camera could not be opened")

    while True:
        success, frame = camera.read()

        if not success or frame is None:
            print("Camera frame unavailable")
            break

        corners, ids, rejected = detector.detectMarkers(frame)

        current_ids = (
            tuple(sorted(ids.flatten().tolist()))
            if ids is not None else ()
        )

        if current_ids != previous_ids:
            print("Detected IDs:", current_ids)
            previous_ids = current_ids

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(
                frame, corners, ids
            )

        status = f"Detected: {list(current_ids)}"

        color = (
            (0, 255, 0)
            if len(current_ids) == 4
            else (0, 0, 255)
        )

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.imshow("ArUco Debug", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped")

finally:
    camera.release()
    cv2.destroyAllWindows()
