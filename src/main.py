"""Milestone 4 entry point: one command for every step.

Run from the src folder:

    python main.py check     1. Find the webcam and confirm all 4 ArUco markers are visible
    python main.py vision    2. Original vision-only test (colour + board mm, no robot)
    python main.py teach     3. Teach board-to-robot calibration (jog over the 4 markers)
    python main.py dry       4. Full pipeline with safety checks, NO robot motion
    python main.py run       5. Live demo: G = hover over target, SPACE = stop, P = park, Q = quit
                                In the video window: C new target, M multi-target mode (1-9 / click
                                to select, A = visit all), + / - hover clearance, ] / [ target height
    python main.py speed 15     Set the arm speed for every program (1-25 %); `speed 15 accel 20`
                                Any command also takes --speed N / --accel N for one run
    python main.py park         Move the arm straight up, then to the saved park pose
    python main.py limits       Show / set joint limits (e.g. limits J3 -89 70) to match the HMI
    python main.py jog          Keyboard jog window (WASD = X/Y, SPACE/CTRL = up/down);
                                saves marker/park points and the 4 board corners (key B)
    python main.py jointtest    Motor check, one joint at a time: direction, amount, return
    python main.py warmup       Pre-session self-test: axis moves, repeatability, wrist, gripper
    python main.py capture      Classifier data: set known positions, every square auto-labelled
    python main.py train        Train the piece classifier (accuracy + confusion matrix)
    python main.py recognize    Live: every square labelled with piece type, colour, confidence
    python main.py board        Type a square (e4): the arm hovers over it
    python main.py boardview    Camera view with the 64 squares drawn; click = which square
                                (board also does: move e2 e4, pick, place, drop, grip, open/close)
    python main.py gripper      Find/test the gripper board (Nano), set open/closed angles, save
    python main.py tool 45      Fingers now 45 mm longer (sticks): raise every board height by 45 mm
    python main.py straightdown Show / fix (`straightdown fix`) the gripper's tilt from straight down
    python main.py relabel      Fix corner names without re-teaching (mirror | rotate | ranks)
    python main.py armcheck     Controller X/Y/Z readout vs the joint angles (diagnostic)
    python main.py piecetest    One piece, legal moves, log where each one lands (finds the cause of misses)
    python main.py gripteach    Keyboard window to set each piece's grip height / open / close live
    python main.py boardteach   Keyboard window to teach every square (W/A/S/D nudge, ENTER save + next)
    python main.py camboard     Camera setup for move detection: click 4 board corners, test detection
    python main.py play         Chess vs the arm, typed moves (--robot white|black, --nogripper, --dry)

Extra options are passed through, e.g.
    python main.py check --camera 2
    python main.py run --firmware 6

Settings (marker distances, camera index, heights, speed) live in vision_config.py.
Full procedure: MILESTONE4_RUNBOOK.md in the repo root.
"""
import json
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
        raise SystemExit("No park pose saved - in python main.py jog, move there and press P")
    p = calib["taught"]["park"]
    print(f"Park pose from {cfg.CALIB_FILE}: X {p['x']:.1f}  Y {p['y']:.1f}  Z {p['z']:.1f}")
    fw = args.firmware or calib.get("firmware", cfg.FIRMWARE)
    arm = connect(fw, args.port, cfg.LOOP_MODE)
    try:
        cur = arm.pose()
        # clearance over the pieces: from the taught board if there is one
        import chess_board as cb
        from robot_backend import rise_to
        board = cb.ChessBoard.load()
        if board is not None:
            top = board.max_z + 80.0            # ~90 mm above the board: clears a king
        elif "board_z" in calib:
            top = calib["board_z"] + cfg.SAFE_Z_MM
        else:
            top = cur["z"] + 100.0
        safe = min(top, p["z"])                 # never higher than the park pose itself
        if cur["z"] < safe - 1:
            print("Rising to safe height ...")
            floor = (board.max_z + 40.0) if board is not None else cur["z"] + 30.0
            rise_to(arm, safe, max(floor, cur["z"] + 10.0), cfg.SPEED_PCT, cur["rot"])
        print("Moving to park ...")
        r = arm.move_joint(p["x"], p["y"], p["z"], p["rot"], cfg.SPEED_PCT)
        print(f"Parked at X {r['x']:.1f}  Y {r['y']:.1f}  Z {r['z']:.1f}")
    finally:
        arm.close()


SPEED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speed.json")
MAX_ACCEL = 30


def _load_speed():
    try:
        with open(SPEED_FILE) as f:
            d = json.load(f)
        return float(d.get("speed", cfg.SPEED_PCT)), float(d.get("accel", 10))
    except (OSError, ValueError):
        return float(cfg.SPEED_PCT), 10.0


def _apply_speed(speed, accel):
    from robot_backend import MAX_SPEED
    import ar4link
    if not 1 <= speed <= MAX_SPEED:
        raise SystemExit(f"speed must be 1-{MAX_SPEED} percent")
    if not 1 <= accel <= MAX_ACCEL:
        raise SystemExit(f"accel must be 1-{MAX_ACCEL} percent")
    cfg.SPEED_PCT = speed
    ar4link.ACCEL_PCT = accel


def speed_cmd(argv):
    """python main.py speed            show the saved speed / acceleration
       python main.py speed 15         save 15 % as the speed for every program
       python main.py speed 15 accel 20   ... and acceleration 20 %"""
    sp, ac = _load_speed()
    if argv:
        try:
            sp = float(argv[0])
            if len(argv) >= 3 and argv[1] == "accel":
                ac = float(argv[2])
            elif len(argv) != 1:
                raise ValueError
        except ValueError:
            print(speed_cmd.__doc__)
            return
        _apply_speed(sp, ac)
        with open(SPEED_FILE, "w") as f:
            json.dump({"speed": sp, "accel": ac}, f, indent=2)
        print(f"Saved: speed {sp:g} %, acceleration {ac:g} % - every program uses this now.")
    else:
        print(f"Speed {sp:g} %, acceleration {ac:g} %  ({SPEED_FILE if os.path.exists(SPEED_FILE) else 'defaults'})")
        print(speed_cmd.__doc__)


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return
    step, rest = sys.argv[1], sys.argv[2:]
    # speed for THIS run: --speed N / --accel N on any command, else the saved default
    sp, ac = _load_speed()
    for flag in ("--speed", "--accel"):
        if flag in rest:
            i = rest.index(flag)
            try:
                v = float(rest[i + 1])
            except (IndexError, ValueError):
                raise SystemExit(f"{flag} needs a number, e.g. {flag} 15")
            if flag == "--speed":
                sp = v
            else:
                ac = v
            del rest[i:i + 2]
    _apply_speed(sp, ac)
    if step == "speed":
        speed_cmd(rest)
        return
    if step not in ("check", "vision", "relabel", "camboard", "train", "recognize"):
        print(f"[speed {sp:g} %, accel {ac:g} %]")
    if step == "check":
        check(rest)
    elif step == "park":
        park(rest)
    elif step == "gripper":
        import gripper_tool
        gripper_tool.run(rest)
    elif step == "tool":
        import chess_board
        chess_board.tool_cmd(rest)
    elif step == "straightdown":
        import chess_board
        chess_board.straightdown_cmd(rest)
    elif step == "relabel":
        import chess_board
        chess_board.relabel(rest)
    elif step == "armcheck":
        import arm_check
        arm_check.run(rest)
    elif step == "piecetest":
        import piece_test
        piece_test.run(rest)
    elif step == "gripteach":
        import grip_teach
        grip_teach.run(rest)
    elif step == "boardteach":
        import board_teach
        board_teach.run(rest)
    elif step == "camboard":
        import cam_moves
        cam_moves.setup(rest)
    elif step == "play":
        import play
        play.run(rest)
    elif step == "limits":
        import limits_tool
        limits_tool.run(rest)
    elif step == "jointtest":
        import jointtest
        jointtest.run(rest)
    elif step == "warmup":
        import warmup
        warmup.run(rest)
    elif step in ("capture", "train", "recognize"):
        import piece_tools
        getattr(piece_tools, step)(rest)
    elif step == "board":
        import board_tool
        board_tool.visit(rest)
    elif step == "boardview":
        import board_tool
        board_tool.view(rest)
    elif step == "jog":
        import jog_pygame
        jog_pygame.main(rest)
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
