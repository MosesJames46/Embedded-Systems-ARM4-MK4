"""Verify the chessboard mapping, with the arm and with the camera.

    python main.py board        type a square (e4) and the arm hovers over it
    python main.py boardview    live camera with the 64-square grid drawn on it;
                                click anywhere to see which square it is

`board` commands (type then Enter):
    e4          hover over e4 (rise -> travel at safe height -> descend)
    corners     visit a1, h1, h8, a8 in turn (Enter between each)
    h 30        set the hover height above the board to 30 mm
    move e2 e4  pick the piece on e2 and put it on e4
    pick e2     pick up only            place e4   put down only
    drop e5     take the piece on e5 off the board (drop zone past the h-file)
    grip 15     gripping height: fingertips this many mm above the board
    piece k     the next moves are this piece type (p n b r q k): uses its grip settings
    tune k height 30   grip a king 30 mm above the board (saved, the game uses it)
    tune k close 25    close to servo angle 25 on a king
    tune k open 50     open only to servo angle 50 around a king      profile   show all
    open / close   the gripper
    fix a square that is off-centre (after hovering over it, e.g. `e4`):
    n a 3       nudge 3 mm toward the a-file      n h 3    toward the h-file
    n far 3     nudge 3 mm toward rank 8           n near 3 toward rank 1
    save        store this square's corrected position (squares nearby follow)
    walk        teach EVERY square in order a1, b1 ... h8: nudge, Enter saves and moves on
    walk e4     start the walk at e4            lift 50   lift between walk squares (mm)
    forget e4   remove a saved correction          points   list saved corrections
    p           park        q   quit
Options: --nogripper (you open/close by hand when asked), --gripper-port /dev/ttyUSB0
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
        self.last = None
        self.safe_z = board.max_z + min(cfg.SAFE_Z_MM, 90.0)   # ~100 mm above the board

    def rise(self):
        from robot_backend import rise_to
        return rise_to(self.arm, self.safe_z, self.board.max_z + 40.0, cfg.SPEED_PCT, self.board.tool_rot)

    def go(self, square):
        self.last = cb.square_name(*cb.square_index(square))
        x, y = self.board.square_to_robot(square)
        z = self.board.square_z(square) + self.hover_mm
        print(f"{square}: robot X {x:.1f}  Y {y:.1f}  hover Z {z:.1f}")
        zt = self.rise()
        while True:                       # cross at the height reached; lower if the far side needs it
            try:
                self.arm.move_linear(x, y, zt, self.board.tool_rot, cfg.SPEED_PCT)
                break
            except RobotFault as e:
                if "refused" not in str(e).lower() or zt - 20 < self.board.max_z + 40.0:
                    raise
                p = self.arm.pose()
                zt -= 20.0
                self.arm.move_linear(p["x"], p["y"], zt, self.board.tool_rot, cfg.SPEED_PCT)
        r = self.arm.move_linear(x, y, z, self.board.tool_rot, cfg.SPEED_PCT)
        print(f"  arrived; robot reports X {r['x']:.1f}  Y {r['y']:.1f}  Z {r['z']:.1f}")

    def hop(self, square, lift_mm):
        """Go to a square at hover height, lifting only lift_mm above the board on the way
        (quicker than the full safe height, still clears the pieces)."""
        self.last = cb.square_name(*cb.square_index(square))
        x, y = self.board.square_to_robot(square)
        z = self.board.square_z(square) + self.hover_mm
        lift = self.board.max_z + lift_mm
        p = self.arm.pose()
        if p["z"] < lift - 1:
            self.arm.move_linear(p["x"], p["y"], lift, self.board.tool_rot, cfg.SPEED_PCT)
        self.arm.move_linear(x, y, max(lift, z), self.board.tool_rot, cfg.SPEED_PCT)
        self.arm.move_linear(x, y, z, self.board.tool_rot, cfg.SPEED_PCT)

    def walk(self, start, firmware, lift_mm):
        """Guided teach of every square: hop, nudge, Enter saves and goes on."""
        order = [f + r for r in "12345678" for f in "abcdefgh"]
        i = order.index(cb.square_name(*cb.square_index(start)))
        print(f"\nWALK from {order[i]}: hover {self.hover_mm:.0f} mm above the board, lifting "
              f"{lift_mm:.0f} mm between squares.\n"
              "  nudge: a 2 / h 2 / near 2 / far 2 (mm)   Enter = save + next   s = skip   "
              "b = back one   q = stop walking")
        done = 0
        while 0 <= i < len(order):
            sq = order[i]
            reached = False
            for lift in (lift_mm, min(lift_mm, 40.0), min(lift_mm, 25.0)):
                try:
                    self.hop(sq, lift)
                    reached = True
                    if lift < lift_mm:
                        print(f"  (lifted only {lift:.0f} mm here - arm limit; keep tall pieces "
                              "out of the way)")
                    break
                except RobotFault as e:
                    last = e
            if not reached:
                print(f"ERROR: cannot reach {sq}: {last}")
                ans = input("  s = skip it, q = stop walking > ").strip().lower()
                if ans == "q":
                    print(f"Walk stopped - {done} squares saved this walk.")
                    return
                i += 1
                continue
            moved = 0.0
            while True:
                cmd = input(f"walk {sq} ({i + 1}/64)> ").strip().lower()
                parts = cmd.split()
                try:
                    if cmd == "":
                        self.save_fix(firmware)
                        done += 1
                        i += 1
                        break
                    if cmd == "s":
                        i += 1
                        break
                    if cmd == "b":
                        i = max(0, i - 1)
                        break
                    if cmd == "q":
                        print(f"Walk stopped - {done} squares saved this walk.")
                        return
                    if len(parts) == 2 and parts[0] in ("a", "h", "near", "far"):
                        self.nudge(parts[0], float(parts[1]))
                        moved += abs(float(parts[1]))
                    else:
                        print("  a/h/near/far + mm, Enter = save, s = skip, b = back, q = stop")
                except (RobotFault, ValueError) as e:
                    print("ERROR:", e)
        print(f"Walk finished - {done} squares saved.")

    def nudge(self, direction, mm):
        if self.last is None:
            raise ValueError("hover over a square first (type e.g. e4)")
        if abs(mm) > 25:
            raise ValueError("nudge at most 25 mm at a time")
        f, r = cb.square_index(self.last)
        here = np.array(self.board.grid_to_robot(f, r))
        axis = {"a": (-1, 0), "h": (1, 0), "near": (0, -1), "far": (0, 1)}.get(direction)
        if axis is None:
            raise ValueError("direction is a, h, near or far")
        v = np.array(self.board.grid_to_robot(f + axis[0], r + axis[1])) - here
        v = v / np.linalg.norm(v) * mm
        p = self.arm.pose()
        self.arm.move_linear(p["x"] + v[0], p["y"] + v[1], p["z"], self.board.tool_rot, cfg.SPEED_PCT)
        q = self.arm.pose()
        off = np.array([q["x"], q["y"]]) - here
        print(f"  {self.last}: now {np.linalg.norm(off):.1f} mm from the computed centre - "
              "`save` when the gripper is centred")

    def save_fix(self, firmware):
        if self.last is None:
            raise ValueError("hover over a square first")
        p = self.arm.pose()
        self.board.set_point(self.last, p["x"], p["y"])
        lines, warn = self.board.save(firmware)
        print(f"  saved {self.last} at X {p['x']:.1f} Y {p['y']:.1f} (chess_board.json updated)")
        for w in warn:
            print("  WARNING:", w)

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
    ap.add_argument("--nogripper", action="store_true")
    ap.add_argument("--gripper-port", default=None)
    ap.add_argument("--adjust", action="store_true", help="pause above each pick/place to centre it")
    ap.add_argument("--adjust-text", action="store_true", help="--adjust with typed nudges, no window")
    args = ap.parse_args(argv)
    board = require_board()
    lines, warn, _ = board.report()
    for line in lines + ["WARNING: " + w for w in warn]:
        print(line)
    calib = load_calib()
    fw = args.firmware or board.firmware or (calib or {}).get("firmware", cfg.FIRMWARE)
    arm = connect(fw, args.port, cfg.LOOP_MODE)
    h = Hover(arm, board, calib)
    import pickplace
    grip = pickplace.make_gripper(args.nogripper, args.gripper_port)
    pp = pickplace.PickPlace(arm, board, grip)
    pp.adjust, pp.firmware = args.adjust or args.adjust_text, fw
    pp.adjust_gui = not args.adjust_text
    cur_piece = None
    walk_lift = 50.0
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
                elif cmd.startswith("n ") or cmd.startswith("nudge "):
                    parts = cmd.split()
                    h.nudge(parts[1], float(parts[2]))
                elif cmd == "save":
                    h.save_fix(fw)
                elif cmd == "walk" or cmd.startswith("walk "):
                    parts = cmd.split()
                    h.walk(parts[1] if len(parts) > 1 else "a1", fw, walk_lift)
                elif cmd.startswith("lift "):
                    walk_lift = max(20.0, min(float(cmd.split()[1]), 150.0))
                    print(f"  walk lift {walk_lift:.0f} mm above the board")
                elif cmd.startswith("forget "):
                    board.forget(cmd.split()[1])
                    board.save(fw)
                    print("  removed; chess_board.json updated")
                elif cmd == "points":
                    print("  " + ("\n  ".join(board.extra_report()) or "no extra points saved"))
                elif cmd.startswith("piece "):
                    t = cmd.split()[1][:1]
                    if t not in pickplace.PIECES:
                        raise ValueError("piece is one of p n b r q k")
                    cur_piece = t
                    pr = pp.profile[t]
                    print(f"  {pickplace.PIECES[t]}: grip {pr['height']} mm above the board, close "
                          f"{pr['close'] if pr['close'] is not None else 'default'}")
                elif cmd.startswith("tune "):
                    _, t, what, val = cmd.split()
                    if t not in pickplace.PIECES or what not in ("height", "close", "open"):
                        raise ValueError("tune <p|n|b|r|q|k> height|close|open <number>")
                    v = float(val)
                    if what == "height" and not 3 <= v <= 80:
                        raise ValueError("height 3-80 mm")
                    if what in ("close", "open") and not 0 <= v <= 180:
                        raise ValueError("servo angle 0-180")
                    pp.profile[t][what] = int(v) if what in ("close", "open") else v
                    pickplace.save_profile(pp.profile)
                    cur_piece = t
                    print(f"  saved: {pickplace.PIECES[t]} {what} = {pp.profile[t][what]} (grip_profile.json)")
                elif cmd == "profile":
                    for t, n in pickplace.PIECES.items():
                        pr = pp.profile[t]
                        d = lambda v: v if v is not None else "default"
                        print(f"  {n:7s} height {pr['height']:5.1f} mm   open {d(pr.get('open'))}   "
                              f"close {d(pr['close'])}")
                elif cmd.startswith("move "):
                    a, b = cmd.split()[1:3]
                    pp.move(a, b, cur_piece)
                elif cmd.startswith("pick "):
                    pp.pick(cmd.split()[1], cur_piece)
                elif cmd.startswith("place "):
                    pp.place(cmd.split()[1])
                elif cmd.startswith("drop "):
                    pp.remove(cmd.split()[1], cur_piece)
                elif cmd.startswith("grip "):
                    pp.grip_mm = max(3.0, min(float(cmd.split()[1]), 80.0))
                    print(f"Gripping {pp.grip_mm:.0f} mm above the board")
                elif cmd == "open":
                    grip.open()
                elif cmd == "close":
                    grip.close()
                elif cmd == "corners":
                    for sq in cb.CORNERS:
                        h.go(sq)
                        input("  check it, then press Enter for the next corner ")
                    h.rise()
                else:
                    h.go(cmd)
            except (RobotFault, ValueError, IndexError) as e:
                print("ERROR:", e)
    finally:
        grip.shutdown()
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
