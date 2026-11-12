"""Verify the chessboard mapping, with the arm and with the camera.

    python main.py board        type a square (e4) and the arm hovers over it
    python main.py boardview    live camera with the 64-square grid drawn on it;
                                click anywhere to see which square it is

`board` commands (type then Enter):
    e4          hover over e4 (rise -> travel at safe height -> descend)
    corners     visit a1, h1, h8, a8 in turn (Enter between each)
    h 30        set the hover height above the board to 30 mm
    p           park        q   quit
"""
import argparse
import json
import os

import cv2
import numpy as np

import chess_board as cb
import vision_config as cfg
import workspace as ws
from robot_backend import connect, RobotFault

DEFAULT_HOVER_MM = 30


def load_calib():
    if not os.path.exists(cfg.CALIB_FILE):
        return None
    with open(cfg.CALIB_FILE) as f:
        return json.load(f)


def require_board():
    board = cb.ChessBoard.load()
    if board is None:
        raise SystemExit("chess_board.json not found - teach the 4 corner squares first:\n"
                         "  python main.py jog   (gripper over a1 centre, press B; then h1, h8, a8)")
    return board


class Hover:
    def __init__(self, arm, board, calib):
        self.arm, self.board, self.calib = arm, board, calib
        self.hover_mm = DEFAULT_HOVER_MM
        self.safe_z = board.surface_z + cfg.SAFE_Z_MM

    def rise(self):
        p = self.arm.pose()
        if p["z"] < self.safe_z - 1:
            self.arm.move_linear(p["x"], p["y"], self.safe_z, p["rot"], cfg.SPEED_PCT)

    def go(self, square):
        x, y = self.board.square_to_robot(square)
        z = self.board.surface_z + self.hover_mm
        print(f"{square}: robot X {x:.1f}  Y {y:.1f}  hover Z {z:.1f}")
        self.rise()
        self.arm.move_linear(x, y, self.safe_z, self.board.tool_rot, cfg.SPEED_PCT)
        r = self.arm.move_linear(x, y, z, self.board.tool_rot, cfg.SPEED_PCT)
        print(f"  arrived; robot reports X {r['x']:.1f}  Y {r['y']:.1f}  Z {r['z']:.1f}")

    def park(self):
        park = (self.calib or {}).get("taught", {}).get("park")
        self.rise()
        if park is None:
            print("No park pose saved (jog window key P) - stayed at safe height")
            return
        self.arm.move_joint(park["x"], park["y"], park["z"], park["rot"], cfg.SPEED_PCT)
        print("Parked")


def visit(argv):
    ap = argparse.ArgumentParser(prog="main.py board")
    ap.add_argument("--port", default=None)
    ap.add_argument("--firmware", choices=["8", "6"], default=None)
    args = ap.parse_args(argv)
    board = require_board()
    lines, warn, _ = board.report()
    for line in lines + ["WARNING: " + w for w in warn]:
        print(line)
    calib = load_calib()
    fw = args.firmware or board.firmware or (calib or {}).get("firmware", cfg.FIRMWARE)
    arm = connect(fw, args.port, cfg.LOOP_MODE)
    h = Hover(arm, board, calib)
    print(__doc__[__doc__.index("`board` commands"):])
    try:
        while True:
            try:
                cmd = input("board> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd:
                continue
            try:
                if cmd == "q":
                    break
                elif cmd == "p":
                    h.park()
                elif cmd.startswith("h "):
                    h.hover_mm = max(10.0, min(float(cmd.split()[1]), cfg.SAFE_Z_MM - 20))
                    print(f"Hover height {h.hover_mm:.0f} mm above the board")
                elif cmd == "corners":
                    for sq in cb.CORNERS:
                        h.go(sq)
                        input("  check it, then press Enter for the next corner ")
                    h.rise()
                else:
                    h.go(cmd)
            except (RobotFault, ValueError) as e:
                print("ERROR:", e)
    finally:
        arm.close()


def view(argv):
    from ArUcoCalibration import ArUcoCalibration

    ap = argparse.ArgumentParser(prog="main.py boardview")
    ap.add_argument("--camera", type=int, default=cfg.CAMERA_INDEX)
    args = ap.parse_args(argv)
    board = require_board()
    plane = ws.taught_plane(load_calib(), cfg.MARKER_POSITIONS_MM)
    if plane is None:
        raise SystemExit("All 4 ArUco markers must be taught first (jog window: marker ID keys, "
                         "P, ENTER) so the camera can map to robot coordinates.")
    aruco = ArUcoCalibration(plane)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    window = "Board view (click a square, Q = quit)"
    clicks = []
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, lambda ev, x, y, *_: clicks.append((x, y))
                         if ev == cv2.EVENT_LBUTTONDOWN else None)
    last = "click a square"
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame unavailable")
            break
        if aruco.update(frame):
            Hinv = np.linalg.inv(aruco.H)  # robot mm -> pixel

            def px(xy):
                p = cv2.perspectiveTransform(np.float32([[xy]]), Hinv)[0, 0]
                return int(round(p[0])), int(round(p[1]))

            for f in range(8):
                for r in range(8):
                    pts = np.array([px(c) for c in board.square_outline(f, r)], np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 255), 1)
                    cx, cy = px(board.grid_to_robot(f, r))
                    for colour, thick in (((0, 0, 0), 3), ((0, 255, 255), 1)):  # outlined label
                        cv2.putText(frame, cb.square_name(f, r), (cx - 9, cy + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, thick)
            while clicks:
                u, v = clicks.pop()
                x, y = aruco.pixel_to_workspace((u, v))
                sq = board.robot_to_square(x, y)
                last = f"{sq or 'not on a square'}   robot X {x:.1f}  Y {y:.1f}"
                print(last)
            status, colour = last, (0, 255, 0)
        else:
            clicks.clear()
            status, colour = f"markers visible: {sorted(aruco.visible_ids)} - need all 4", (0, 0, 255)
        cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
