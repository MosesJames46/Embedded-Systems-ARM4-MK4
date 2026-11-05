"""Milestone 4 entry point: one command for every step.

Run from the src folder:

    python main.py check     1. Find the webcam and confirm all 4 ArUco markers are visible
    python main.py vision    2. Original vision-only test (colour + board mm, no robot)
    python main.py teach     3. Teach board-to-robot calibration (jog over markers 4-7)
    python main.py dry       4. Full pipeline with safety checks, NO robot motion
    python main.py run       5. Live demo: G = hover over target, SPACE = stop, P = park, Q = quit
                                In the video window: + / - change hover clearance, ] / [ target height
    python main.py park         Move the arm straight up, then to the saved park pose

Extra options are passed through, e.g.
    python main.py check --camera 2
    python main.py run --firmware 6

Settings (marker distances, camera index, heights, speed) live in vision_config.py.
Full procedure: MILESTONE4_RUNBOOK.md in the repo root.
"""
import os
import runpy
import sys

import vision_config as cfg


def check(argv):
    """Probe camera indices, then show a live view with detected marker IDs."""
    import argparse
    import cv2

    ap = argparse.ArgumentParser(prog="main.py check")
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args(argv)

    if args.camera is None:
        print("Probing camera indices 0-5 ...")
        found = []
        for i in range(6):
            cap = cv2.VideoCapture(i)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            if ok:
                found.append(i)
                print(f"  camera {i}: OK ({frame.shape[1]}x{frame.shape[0]})")
        if not found:
            raise SystemExit("No working camera found. Is the webcam plugged in?")
        index = cfg.CAMERA_INDEX if cfg.CAMERA_INDEX in found else found[-1]
        print(f"Showing camera {index}. If it is the wrong one, run: "
              f"python main.py check --camera N  (then set CAMERA_INDEX in vision_config.py)")
    else:
        index = args.camera

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())
    wanted = set(cfg.MARKER_POSITIONS_MM)
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}")
    print("Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame unavailable")
            break
        corners, ids, _ = detector.detectMarkers(frame)
        seen = set() if ids is None else {int(i) for i in ids.flatten()}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        missing = sorted(wanted - seen)
        text = "ALL 4 MARKERS VISIBLE" if not missing else f"missing markers: {missing}"
        colour = (0, 200, 0) if not missing else (0, 0, 255)
        cv2.putText(frame, f"camera {index}: {text}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.imshow("Camera check (Q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def park(argv):
    """Rise to a safe height, then joint-move to the park pose saved by teach."""
    import argparse
    import json

    from robot_backend import connect

    ap = argparse.ArgumentParser(prog="main.py park")
    ap.add_argument("--port", default=None)
    ap.add_argument("--firmware", choices=["8", "6"], default=None)
    args = ap.parse_args(argv)
    if not os.path.exists(cfg.CALIB_FILE):
        raise SystemExit("robot_calib.json not found - run: python main.py teach")
    with open(cfg.CALIB_FILE) as f:
        calib = json.load(f)
    if "park" not in calib.get("taught", {}):
        raise SystemExit("No park pose saved - run teach and type: save park, then w")
    p = calib["taught"]["park"]
    fw = args.firmware or calib.get("firmware", cfg.FIRMWARE)
    arm = connect(fw, args.port, cfg.LOOP_MODE)
    try:
        cur = arm.pose()
        safe = calib["board_z"] + cfg.SAFE_Z_MM
        if cur["z"] < safe - 1:
            print("Rising to safe height ...")
            arm.move_linear(cur["x"], cur["y"], safe, cur["rot"], cfg.SPEED_PCT)
        print("Moving to park ...")
        r = arm.move_joint(p["x"], p["y"], p["z"], p["rot"], cfg.SPEED_PCT)
        print(f"Parked at X {r['x']:.1f}  Y {r['y']:.1f}  Z {r['z']:.1f}")
    finally:
        arm.close()


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return
    step, rest = sys.argv[1], sys.argv[2:]
    if step == "check":
        check(rest)
    elif step == "park":
        park(rest)
    elif step == "vision":
        sys.argv = ["vision_test.py"] + rest
        runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_test.py"),
                       run_name="__main__")
    elif step == "teach":
        import teach_robot
        sys.argv = ["teach_robot.py"] + rest
        teach_robot.main()
    elif step in ("dry", "run"):
        import vision_move
        sys.argv = ["vision_move.py"] + (["--dry"] if step == "dry" else []) + rest
        vision_move.main()
    else:
        print(f"Unknown step '{step}'.\n")
        print(__doc__)


if __name__ == "__main__":
    main()
