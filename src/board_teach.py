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
    + / -        hover height up / down 2 mm (only for looking; it is not saved)
    [ / ]        lift between squares down / up 10 mm (clears pieces on the way)
    G            gripper open / close (straddle a centred pawn to check)
    U            undo: forget the saved point on this square
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

    def go_current():
        sq = ORDER[state["i"]]
        for lift in (state["lift"], min(state["lift"], 40.0), min(state["lift"], 25.0)):
            try:
                h.hop(sq, lift)
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
    screen = pygame.display.set_mode((900, 470))
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
        saved = set(board.extra) | set(cb.CORNERS)
        screen.blit(mid.render(f"step {STEPS[state['step']]} mm   hover {h.hover_mm:.0f} mm   lift "
                               f"{state['lift']:.0f} mm   saved {len(saved)}/64 squares", True, FG),
                    (20, y)); y += 30
        ad = "A = toward h, D = toward a" if state["flip"] else "A = toward a, D = toward h"
        screen.blit(small.render(f"W = toward rank 8   S = toward rank 1   {ad}   (F swaps A/D)",
                                 True, BLUE), (20, y)); y += 24
        for line in ["ENTER save + next   RIGHT/LEFT next/prev (no save)   J+square jump   U undo",
                     "UP/DOWN step   +/- hover (look only)   [ ] lift   G gripper   P park   Q quit"]:
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
                elif ch == "[":
                    state["lift"] = max(state["lift"] - 10, 20)
                elif ch == "]":
                    state["lift"] = min(state["lift"] + 10, 150)
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
                    if grip is None:
                        say("No gripper board connected", AMBER)
                    elif getattr(grip, "state", None) == "open":
                        grip.close()
                    else:
                        grip.open()
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
