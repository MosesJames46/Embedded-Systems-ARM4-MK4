"""Keyboard window for tuning how each piece type is gripped.

    python main.py gripteach --firmware 6

The arm hovers safely above the piece's starting square. Lower it with the keys until the
fingers are where you want to hold the piece, set the jaw opening and closing live, ENTER
saves all three for that piece type (grip_profile.json - the game uses them).

    1-6          piece type: 1 pawn 2 knight 3 bishop 4 rook 5 queen 6 king
                 (goes to its starting square: e2 g1 f1 h1 d1 e1)
    J + square   use another square, e.g. J then d then 8
    CTRL / SPACE lower / raise the gripper   UP / DOWN step 0.5 1 2 5 mm
    O / C        open / close the jaws at the current angles
    [ / ]        open angle -/+ 2 (jaws move right away if open)
    , / .        close angle -/+ 2 (jaws move right away if closed)
    ENTER        save height + open + close for this piece type
    T            test: lift up, then a full pick-and-place on the same square with the
                 saved settings
    H            back up to the safe hover        P park        Q / ESC quit
"""
import argparse

import chess_board as cb
import pickplace
import vision_config as cfg
from robot_backend import RobotFault, connect

TYPES = {"1": "p", "2": "n", "3": "b", "4": "r", "5": "q", "6": "k"}
START = {"p": "e2", "n": "g1", "b": "f1", "r": "h1", "q": "d1", "k": "e1"}
STEPS = [0.5, 1.0, 2.0, 5.0]
HOVER_MM = 60.0       # safe hover above the board while choosing
MIN_MM = 3.0          # fingertips never closer to the board than this
BG, FG, DIM = (24, 26, 32), (230, 230, 230), (130, 130, 140)
GREEN, RED, AMBER, BLUE = (90, 200, 120), (235, 90, 90), (240, 180, 70), (110, 160, 240)


def run(argv):
    import pygame
    ap = argparse.ArgumentParser(prog="main.py gripteach")
    ap.add_argument("--firmware", choices=["8", "6"], default=None)
    ap.add_argument("--port", default=None)
    ap.add_argument("--no-gripper", action="store_true")
    args = ap.parse_args(argv)

    import board_tool
    board = board_tool.require_board()
    calib = board_tool.load_calib()
    fw = args.firmware or board.firmware or (calib or {}).get("firmware", cfg.FIRMWARE)
    arm = connect(fw, args.port, cfg.LOOP_MODE)
    hover = board_tool.Hover(arm, board, calib)
    hover.hover_mm = HOVER_MM - pickplace.TAUGHT_MM     # Hover heights count from the taught Z
    grip = None
    if not args.no_gripper:
        try:
            from gripper import Gripper
            grip = Gripper()
        except Exception as e:
            print("Gripper not connected:", e)
    prof = pickplace.load_profile()
    base_open = grip.open_pos if grip else 45
    base_close = grip.closed_pos if grip else 0
    pp = pickplace.PickPlace(arm, board, grip or pickplace.ManualGripper())

    st = {"t": "k", "sq": "e1", "step": 2, "mm": HOVER_MM, "jaw": None,
          "msg": "", "col": FG}
    st["open"] = prof["k"].get("open") if prof["k"].get("open") is not None else base_open
    st["close"] = prof["k"]["close"] if prof["k"]["close"] is not None else base_close

    def say(m, col=FG):
        st["msg"], st["col"] = m, col
        print(m)

    def z_for(mm):
        return board.square_z(st["sq"]) - pickplace.TAUGHT_MM + mm

    def to_height(mm):
        mm = max(MIN_MM, min(mm, 150.0))
        x, y = board.square_to_robot(st["sq"])
        arm.move_linear(x, y, z_for(mm), board.tool_rot, cfg.SPEED_PCT)
        st["mm"] = mm

    def select(t, sq=None):
        st["t"] = t
        st["sq"] = sq or START[t]
        pr = prof[t]
        st["open"] = pr.get("open") if pr.get("open") is not None else base_open
        st["close"] = pr["close"] if pr["close"] is not None else base_close
        last = None
        for lift in (HOVER_MM, 40.0, 25.0):
            hover.hover_mm = lift - pickplace.TAUGHT_MM
            try:
                hover.hop(st["sq"], lift)
                st["mm"] = st["start"] = lift
                break
            except RobotFault as e:
                last = e
        else:
            raise last
        hover.hover_mm = HOVER_MM - pickplace.TAUGHT_MM
        if grip:
            grip.servo(int(st["open"]))
            st["jaw"] = "open"
        say(f"{pickplace.PIECES[t]} on {st['sq']}: saved height {pr['height']:.1f} mm. Lower with "
            "CTRL until the fingers are where you want to hold it.")

    def jaw(angle_key, delta):
        st[angle_key] = max(0, min(180, st[angle_key] + delta))
        if grip and st["jaw"] == ("open" if angle_key == "open" else "closed"):
            grip.servo(int(st[angle_key]))

    pygame.init()
    screen = pygame.display.set_mode((880, 400))
    pygame.display.set_caption("Grip teach")
    big = pygame.font.SysFont("monospace", 32, bold=True)
    mid = pygame.font.SysFont("monospace", 20)
    small = pygame.font.SysFont("monospace", 16)
    jump = None

    def draw():
        screen.fill(BG)
        t = st["t"]
        pr = prof[t]
        y = 15
        screen.blit(big.render(f"{pickplace.PIECES[t].upper()}  on {st['sq']}", True, FG), (20, y)); y += 50
        screen.blit(mid.render(f"fingertips {st['mm']:.1f} mm above the board   (step "
                               f"{STEPS[st['step']]} mm)", True, GREEN if st["mm"] < HOVER_MM else FG),
                    (20, y)); y += 30
        screen.blit(mid.render(f"jaws {st['jaw'] or '?'}   open angle {st['open']}   close angle "
                               f"{st['close']}", True, FG), (20, y)); y += 30
        sv = lambda v: "default" if v is None else v
        screen.blit(mid.render(f"saved: height {pr['height']:.1f}  open {sv(pr.get('open'))}  close "
                               f"{sv(pr['close'])}", True, BLUE), (20, y)); y += 36
        for line in ["1-6 piece (1 pawn 2 knight 3 bishop 4 rook 5 queen 6 king)   J+square other square",
                     "CTRL down  SPACE up  UP/DOWN step   O open  C close   [ ] open angle   , . close angle",
                     "ENTER save this piece   T test pick+place   H hover   P park   Q quit"]:
            screen.blit(small.render(line, True, DIM), (20, y)); y += 22
        if jump is not None:
            screen.blit(mid.render(f"square: {jump}_", True, AMBER), (20, y))
        y += 34
        screen.blit(mid.render(st["msg"][:78], True, st["col"]), (20, y))
        if len(st["msg"]) > 78:
            screen.blit(mid.render(st["msg"][78:156], True, st["col"]), (20, y + 24))
        pygame.display.flip()

    try:
        try:
            select("k")
        except RobotFault as e:
            say(f"ERROR: {e}  - press another piece key (1-6) or J + square", RED)
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
                                sq, jump = jump, None
                                select(st["t"], sq)
                            else:
                                jump = None
                                say("type a file letter then a rank number", AMBER)
                    elif k == pygame.K_ESCAPE:
                        jump = None
                elif k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ch in TYPES:
                    select(TYPES[ch])
                elif k == pygame.K_j:
                    jump = ""
                elif k in (pygame.K_LCTRL, pygame.K_RCTRL):
                    to_height(st["mm"] - STEPS[st["step"]])
                    pygame.event.clear(pygame.KEYDOWN)
                    if st["mm"] <= MIN_MM:
                        say(f"Lowest allowed ({MIN_MM:.0f} mm above the board)", AMBER)
                elif k == pygame.K_SPACE:
                    to_height(st["mm"] + STEPS[st["step"]])
                    pygame.event.clear(pygame.KEYDOWN)
                elif k == pygame.K_UP:
                    st["step"] = min(st["step"] + 1, len(STEPS) - 1)
                elif k == pygame.K_DOWN:
                    st["step"] = max(st["step"] - 1, 0)
                elif k == pygame.K_o:
                    if grip:
                        grip.servo(int(st["open"]))
                    st["jaw"] = "open"
                elif k == pygame.K_c:
                    if grip:
                        grip.servo(int(st["close"]))
                    st["jaw"] = "closed"
                elif ch == "[":
                    jaw("open", -2)
                elif ch == "]":
                    jaw("open", +2)
                elif ch == ",":
                    jaw("close", -2)
                elif ch == ".":
                    jaw("close", +2)
                elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if st["mm"] >= st.get("start", HOVER_MM) - 0.1:
                        say("Lower to the gripping height first (CTRL), then ENTER", AMBER)
                    else:
                        t = st["t"]
                        prof[t]["height"] = round(st["mm"], 1)
                        prof[t]["open"] = int(st["open"])
                        prof[t]["close"] = int(st["close"])
                        pickplace.save_profile(prof)
                        pp.profile = pickplace.load_profile()
                        say(f"Saved {pickplace.PIECES[t]}: height {st['mm']:.1f} mm, open {st['open']}, "
                            f"close {st['close']}. T to test.", GREEN)
                elif k == pygame.K_t:
                    pp.profile = pickplace.load_profile()
                    if grip:
                        grip.servo(int(st["open"]))
                    to_height(HOVER_MM)
                    say("Testing: pick up, lift, put back ...")
                    draw()
                    pp.move(st["sq"], st["sq"], st["t"])
                    to_height(HOVER_MM)
                    st["jaw"] = "open"
                    say("Test done. Adjust and ENTER again, or pick another piece (1-6).", GREEN)
                    pygame.event.clear(pygame.KEYDOWN)
                elif k == pygame.K_h:
                    to_height(HOVER_MM)
                elif k == pygame.K_p:
                    hover.park()
                    say("Parked. Pick a piece (1-6) to continue.")
            except (RobotFault, ValueError) as e:
                say(f"ERROR: {e}", RED)
                pygame.event.clear(pygame.KEYDOWN)
            draw()
    finally:
        if grip is not None:
            grip.shutdown()
        arm.close()
        pygame.quit()
