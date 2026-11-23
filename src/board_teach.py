"""Keyboard window for teaching the board square by square (like jog, for the walk).

    python main.py boardteach --firmware 6          start at a1
    python main.py boardteach --firmware 6 --start e4

The arm goes to each square in turn (a1, b1 ... h1, a2 ... h8) at the hover height, with
the gripper in the same direction the game uses. Centre it with the keys, ENTER saves the
square and goes to the next. Every saved square also pulls its neighbours closer, so the
nudges get smaller as you go.

    W / S        nudge toward rank 8 / toward rank 1          (along the board, not the robot)
    A / D        nudge toward the a-file / the h-file          F swaps A and D if they feel reversed
    UP / DOWN    nudge step  0.5  1  2  5 mm
    ENTER        save this square, go to the next
    RIGHT / LEFT next / previous square WITHOUT saving
    J + square   jump to a square, e.g. J then e then 4
    + / -        hover height for going square to square, up / down 2 mm
    PGUP / PGDN  lift between squares up / down 10 mm (clears pieces on the way)

  Grip testing (the same as gripteach, on whatever square you are on):
    1-6          piece type: 1 pawn 2 knight 3 bishop 4 rook 5 queen 6 king (loads its grip)
    CTRL / SPACE lower / raise the gripper by the step (fingertip height shown)
    O / C        open / close the jaws        G toggles them
    [ / ]        open angle -/+ 2             , / .   close angle -/+ 2   (jaws move live)
    M            save height + open + close for this piece type (the game uses them)
    T            test: full pick-and-place on this square with the saved grip
    U            undo: forget the saved point on this square
    V            point the gripper exactly straight down (keeps the finger direction)
    Z / X        turn the fingers about the vertical (J6) by the step in degrees
    I / K        tilt (J5 direction): gripper TIP swings away from / toward the robot base
                 by the step in degrees - for when "straight down" does not LOOK vertical
                 V, Z and X move the arm right away and are SAVED - every program uses them
    P            park          Q / ESC   quit
"""
import argparse
import json
import os

import numpy as np

import chess_board as cb
import vision_config as cfg
from robot_backend import RobotFault, connect

ORDER = [f + r for r in "12345678" for f in "abcdefgh"]
STEPS = [0.5, 1.0, 2.0, 5.0]
SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jog_settings.json")
BG, FG, DIM = (24, 26, 32), (230, 230, 230), (130, 130, 140)
GREEN, RED, AMBER, BLUE = (90, 200, 120), (235, 90, 90), (240, 180, 70), (110, 160, 240)


def _load_flip():
    try:
        with open(SETTINGS) as f:
            return bool(json.load(f).get("boardteach_flip", False))
    except (OSError, ValueError):
        return False


def _save_flip(flip):
    try:
        with open(SETTINGS) as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}
    d["boardteach_flip"] = flip
    with open(SETTINGS, "w") as f:
        json.dump(d, f, indent=2)


def run(argv):
    import pygame
    ap = argparse.ArgumentParser(prog="main.py boardteach")
    ap.add_argument("--firmware", choices=["8", "6"], default=None)
    ap.add_argument("--port", default=None)
    ap.add_argument("--start", default="a1")
    ap.add_argument("--hover", type=float, default=10.0, help="mm above the board (default 10)")
    ap.add_argument("--lift", type=float, default=50.0, help="mm lift between squares")
    ap.add_argument("--no-gripper", action="store_true")
    args = ap.parse_args(argv)

    import board_tool
    board = board_tool.require_board()
    calib = board_tool.load_calib()
    fw = args.firmware or board.firmware or (calib or {}).get("firmware", cfg.FIRMWARE)
    arm = connect(fw, args.port, cfg.LOOP_MODE)
    h = board_tool.Hover(arm, board, calib)
    h.hover_mm = args.hover
    grip = None
    if not args.no_gripper:
        try:
            from gripper import Gripper
            grip = Gripper()
        except Exception as e:
            print("Gripper not connected:", e)

    state = {"i": ORDER.index(cb.square_name(*cb.square_index(args.start))), "step": 1,
             "lift": args.lift, "flip": _load_flip(), "msg": "", "col": FG, "saved_now": 0}

    def say(m, col=FG):
        state["msg"], state["col"] = m, col
        print(m)

    import kinematics as K
    import pickplace
    state["yaw0"] = 0.0
    state["tilt0"] = 0.0
    TYPES = {"1": "p", "2": "n", "3": "b", "4": "r", "5": "q", "6": "k"}
    prof = pickplace.load_profile()
    base_open = grip.open_pos if grip else 45
    base_close = grip.closed_pos if grip else 0

    def load_piece(t):
        state["t"] = t
        pr = prof[t]
        state["open"] = int(pr.get("open") if pr.get("open") is not None else base_open)
        state["close"] = int(pr["close"] if pr["close"] is not None else base_close)

    load_piece("p")
    state["jaw"] = None
    state["tip"] = None

    def to_tip(mm):
        mm = max(3.0, min(mm, 150.0))
        sq = ORDER[state["i"]]
        p = arm.pose()
        z = board.square_z(sq) - pickplace.TAUGHT_MM + mm
        arm.move_linear(p["x"], p["y"], z, board.tool_rot, cfg.SPEED_PCT)
        state["tip"] = mm

    def jaw_to(angle, which):
        if grip is None:
            say("No gripper board connected", AMBER)
            return
        grip.servo(int(angle))
        state["jaw"] = which

    def set_rot(new_rv, what):
        board.corners["a1"]["rot"] = [float(v) for v in new_rv]
        board.tool_rot = board.corners["a1"]["rot"]
        board.save(fw)
        p = arm.pose()
        arm.move_linear(p["x"], p["y"], p["z"], board.tool_rot, cfg.SPEED_PCT)
        q = arm.pose()["joints"]
        say(f"{what} - saved. Tilt {K.tilt_from_down(board.tool_rot):.2f} deg from model-vertical; "
            f"J5 {q[4]:.1f}  J6 {q[5]:.1f}", GREEN)

    def go_current():
        sq = ORDER[state["i"]]
        for lift in (state["lift"], min(state["lift"], 40.0), min(state["lift"], 25.0)):
            try:
                h.hop(sq, lift)
                state["tip"] = pickplace.TAUGHT_MM + h.hover_mm
                say(f"At {sq}. Centre it, then ENTER." +
                    (f" (lifted only {lift:.0f} mm - arm limit)" if lift < state["lift"] else ""))
                return True
            except RobotFault as e:
                last = e
        say(f"Cannot reach {sq}: {last}. RIGHT = skip it.", RED)
        return False

    def offset():
        try:
            p = arm.pose()
        except RobotFault:
            return None
        f, r = cb.square_index(ORDER[state["i"]])
        cx, cy = board.grid_to_robot(f, r)
        here = np.array(board.grid_to_robot(f, r))
        # express the offset along the board's file / rank directions
        u = np.array(board.grid_to_robot(f + 1, r)) - here
        v = np.array(board.grid_to_robot(f, r + 1)) - here
        d = np.array([p["x"] - cx, p["y"] - cy])
        return float(d @ u / np.linalg.norm(u)), float(d @ v / np.linalg.norm(v))

    pygame.init()
    screen = pygame.display.set_mode((960, 580))
    pygame.display.set_caption("Board teach")
    big = pygame.font.SysFont("monospace", 34, bold=True)
    mid = pygame.font.SysFont("monospace", 20)
    small = pygame.font.SysFont("monospace", 16)
    jump = None

    def draw():
        screen.fill(BG)
        sq = ORDER[state["i"]]
        y = 15
        screen.blit(big.render(f"{sq}   ({state['i'] + 1}/64)", True, FG), (20, y)); y += 50
        off = offset()
        if off is not None:
            col = GREEN if abs(off[0]) < 0.6 and abs(off[1]) < 0.6 else FG
            screen.blit(mid.render(f"moved {off[0]:+.1f} mm toward h, {off[1]:+.1f} mm toward rank 8",
                                   True, col), (20, y))
        y += 30
        pr = prof[state["t"]]
        tip = "?" if state["tip"] is None else f"{state['tip']:.1f}"
        screen.blit(mid.render(f"{pickplace.PIECES[state['t']]} (1-6): tips {tip} mm up  jaws "
                               f"{state['jaw'] or '?'}  open {state['open']}  close {state['close']}  "
                               f"| saved h {pr['height']:.0f}", True, BLUE), (20, y))
        y += 28
        saved = set(board.extra) | set(cb.CORNERS)
        screen.blit(mid.render(f"step {STEPS[state['step']]} mm   hover {h.hover_mm:.0f} mm   lift "
                               f"{state['lift']:.0f} mm   saved {len(saved)}/64 squares", True, FG),
                    (20, y)); y += 30
        screen.blit(small.render(f"gripper tilt {K.tilt_from_down(board.tool_rot):.2f} deg from straight down"
                                 f"   tilted {state['tilt0']:+.1f} (I/K)  turned {state['yaw0']:+.1f} (Z/X) this session",
                                 True, GREEN if K.tilt_from_down(board.tool_rot) < 0.3 else AMBER), (20, y))
        y += 22
        ad = "A = toward h, D = toward a" if state["flip"] else "A = toward a, D = toward h"
        screen.blit(small.render(f"W = toward rank 8   S = toward rank 1   {ad}   (F swaps A/D)",
                                 True, BLUE), (20, y)); y += 24
        for line in ["ENTER save + next   RIGHT/LEFT next/prev (no save)   J+square jump   U undo",
                     "UP/DOWN step   +/- hover   PGUP/PGDN lift   P park   Q quit",
                     "GRIP: 1-6 piece  CTRL/SPACE down/up  O/C G jaws  [ ] open  , . close  M save  T test",
                     "V straight down   I/K tilt tip away/toward robot (J5)   Z/X turn fingers (J6)  - saved"]:
            screen.blit(small.render(line, True, DIM), (20, y)); y += 22
        if jump is not None:
            screen.blit(mid.render(f"jump to: {jump}_", True, AMBER), (20, y))
        y += 30
        screen.blit(mid.render(state["msg"][:80], True, state["col"]), (20, y))
        if len(state["msg"]) > 80:
            screen.blit(mid.render(state["msg"][80:160], True, state["col"]), (20, y + 24))
        # mini board: rank 8 at the top, saved squares green, current yellow
        s, x0, y0 = 32, 620, 160
        for r in range(8):
            for f in range(8):
                name = cb.square_name(f, r)
                rect = pygame.Rect(x0 + f * s, y0 + (7 - r) * s, s - 2, s - 2)
                colr = (60, 62, 70) if (f + r) % 2 else (85, 88, 98)
                if name in saved:
                    colr = (50, 120, 70)
                if name == sq:
                    colr = (200, 160, 40)
                pygame.draw.rect(screen, colr, rect)
        screen.blit(small.render("a" + " " * 26 + "h", True, DIM), (x0 + 10, y0 + 8 * s + 2))
        pygame.display.flip()

    def keep_last():
        """After a nudge: keys pressed while the arm was moving - keep only the LAST one, so
        a press is never lost but a held key cannot pile up moves."""
        pending = pygame.event.get(pygame.KEYDOWN)
        if pending:
            pygame.event.post(pending[-1])

    def drain():
        """After going to another square: drop keys pressed on the way (a double ENTER
        must not save the next square before you have centred it)."""
        pygame.event.clear(pygame.KEYDOWN)

    try:
        go_current()
        draw()
        running = True
        while running:
            ev = pygame.event.wait()
            if ev.type == pygame.QUIT:
                break
            if ev.type != pygame.KEYDOWN:
                continue
            k, ch = ev.key, ev.unicode.lower()
            try:
                if jump is not None:
                    if ch and ch in "abcdefgh12345678":
                        jump += ch
                        if len(jump) == 2:
                            if jump[0] in "abcdefgh" and jump[1] in "12345678":
                                state["i"] = ORDER.index(jump)
                                jump = None
                                go_current()
                            else:
                                say("type a file letter then a rank number, e.g. e4", AMBER)
                                jump = None
                    elif k == pygame.K_ESCAPE:
                        jump = None
                elif k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k in (pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d):
                    mm = STEPS[state["step"]]
                    say("MOVING ...", AMBER)
                    draw()
                    if k == pygame.K_w:
                        h.nudge("far", mm)
                    elif k == pygame.K_s:
                        h.nudge("near", mm)
                    else:
                        toward_a = (k == pygame.K_a) != state["flip"]
                        h.nudge("a" if toward_a else "h", mm)
                    keep_last()
                    say(f"nudged {mm} mm")
                elif k == pygame.K_UP:
                    state["step"] = min(state["step"] + 1, len(STEPS) - 1)
                elif k == pygame.K_DOWN:
                    state["step"] = max(state["step"] - 1, 0)
                elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    h.last = ORDER[state["i"]]
                    h.save_fix(fw)
                    state["saved_now"] += 1
                    if state["i"] < 63:
                        state["i"] += 1
                        go_current()
                    else:
                        say("Saved h8 - all squares done. Q to quit.", GREEN)
                    drain()
                elif k == pygame.K_RIGHT:
                    state["i"] = min(state["i"] + 1, 63)
                    go_current()
                    drain()
                elif k == pygame.K_LEFT:
                    state["i"] = max(state["i"] - 1, 0)
                    go_current()
                    drain()
                elif k == pygame.K_j:
                    jump = ""
                elif ch in ("+", "="):
                    h.hover_mm = min(h.hover_mm + 2, 60)
                    go_current()
                elif ch == "-":
                    h.hover_mm = max(h.hover_mm - 2, 4)
                    go_current()
                elif k == pygame.K_PAGEDOWN:
                    state["lift"] = max(state["lift"] - 10, 20)
                elif k == pygame.K_PAGEUP:
                    state["lift"] = min(state["lift"] + 10, 150)
                elif ch in TYPES:
                    load_piece(TYPES[ch])
                    pr = prof[state["t"]]
                    say(f"{pickplace.PIECES[state['t']]}: saved grip height {pr['height']:.1f} mm, open "
                        f"{state['open']}, close {state['close']}")
                elif k in (pygame.K_LCTRL, pygame.K_RCTRL, pygame.K_SPACE):
                    if state["tip"] is None:
                        state["tip"] = pickplace.TAUGHT_MM + h.hover_mm
                    d = STEPS[state["step"]] * (-1 if k != pygame.K_SPACE else 1)
                    say("MOVING ...", AMBER); draw()
                    to_tip(state["tip"] + d)
                    say(f"fingertips {state['tip']:.1f} mm above the board")
                    keep_last()
                elif k == pygame.K_o:
                    jaw_to(state["open"], "open")
                elif k == pygame.K_c:
                    jaw_to(state["close"], "closed")
                elif ch == "[":
                    state["open"] = max(0, state["open"] - 2)
                    if state["jaw"] == "open":
                        jaw_to(state["open"], "open")
                elif ch == "]":
                    state["open"] = min(180, state["open"] + 2)
                    if state["jaw"] == "open":
                        jaw_to(state["open"], "open")
                elif ch == ",":
                    state["close"] = max(0, state["close"] - 2)
                    if state["jaw"] == "closed":
                        jaw_to(state["close"], "closed")
                elif ch == ".":
                    state["close"] = min(180, state["close"] + 2)
                    if state["jaw"] == "closed":
                        jaw_to(state["close"], "closed")
                elif k == pygame.K_m:
                    t = state["t"]
                    if state["tip"] is None or state["tip"] >= pickplace.TAUGHT_MM + h.hover_mm - 0.1:
                        say("Lower to the gripping height first (CTRL), then M", AMBER)
                    else:
                        prof[t]["height"] = round(state["tip"], 1)
                        prof[t]["open"], prof[t]["close"] = int(state["open"]), int(state["close"])
                        pickplace.save_profile(prof)
                        say(f"Saved {pickplace.PIECES[t]} grip: height {state['tip']:.1f} mm, open "
                            f"{state['open']}, close {state['close']}. T to test.", GREEN)
                elif k == pygame.K_t:
                    if grip is None:
                        say("Test needs the gripper board connected", AMBER)
                    else:
                        sq = ORDER[state["i"]]
                        pp = pickplace.PickPlace(arm, board, grip)
                        say(f"Testing {pickplace.PIECES[state['t']]} pick-and-place on {sq} ...", AMBER)
                        draw()
                        pp.move(sq, sq, state["t"])
                        go_current()
                        state["jaw"] = "open"
                        say("Test done - adjust and M again, or ENTER to save the square.", GREEN)
                        drain()
                elif k == pygame.K_f:
                    state["flip"] = not state["flip"]
                    _save_flip(state["flip"])
                    say("A/D swapped (remembered)")
                elif k == pygame.K_u:
                    sq = ORDER[state["i"]]
                    if sq in cb.CORNERS:
                        say("Corners cannot be forgotten - re-centre and ENTER to update", AMBER)
                    else:
                        board.forget(sq)
                        board.save(fw)
                        say(f"forgot {sq}; going to its computed position")
                        go_current()
                elif k == pygame.K_g:
                    if state["jaw"] == "open":
                        jaw_to(state["close"], "closed")
                    else:
                        jaw_to(state["open"], "open")
                elif k == pygame.K_v:
                    import chess_board as _cb
                    set_rot(_cb.straight_down_rot(board.tool_rot), "Pointed straight down")
                    keep_last()
                elif k in (pygame.K_z, pygame.K_x):
                    deg = STEPS[state["step"]] * (1 if k == pygame.K_z else -1)
                    set_rot(K.rotate_about_vertical(board.tool_rot, deg), f"Fingers turned {deg:+g} deg")
                    state["yaw0"] += deg
                    keep_last()
                elif k in (pygame.K_i, pygame.K_k):
                    deg = STEPS[state["step"]] * (1 if k == pygame.K_i else -1)
                    p = arm.pose()
                    set_rot(K.tilt_toward(board.tool_rot, (p["x"], p["y"]), deg),
                            f"Tip tilted {abs(deg):g} deg {'away from' if deg > 0 else 'toward'} the robot")
                    state["tilt0"] += deg
                    keep_last()
                elif k == pygame.K_p:
                    h.park()
                    say("Parked. ENTER/RIGHT/LEFT to continue.")
            except (RobotFault, ValueError) as e:
                say(f"ERROR: {e}", RED)
                drain()
            draw()
    finally:
        print(f"{state['saved_now']} squares saved this session (chess_board.json).")
        if grip is not None:
            grip.shutdown()
        arm.close()
        pygame.quit()
