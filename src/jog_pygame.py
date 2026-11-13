"""Keyboard jog controller for the AR4 (pygame window).

    python main.py jog                 (firmware from robot_calib.json, else vision_config)
    python main.py jog --firmware 6

Hold a key to keep moving; each step is one straight-line move of the current step size.
The robot frame: X = away from the base, Y = to the robot's left, Z = up.

    W / S        +X / -X  (away from / toward the base)
    A / D        +Y / -Y  (robot's left / right)
    SPACE        +Z  (up)
    CTRL         -Z  (down)
    UP / DOWN    step size  1 2 5 10 20 50 mm
    R            re-lock the gripper orientation to the current pose (it is locked
                 when the window opens, so jogging never rotates the gripper)
    V            point the gripper STRAIGHT DOWN (Ry 180) in place - needs Z >= 120 mm.
                 Straight down keeps J4/J5 still when the base turns (no 180 flips)
    Z / X        turn the gripper about the vertical axis -5 / +5 deg (finger direction)
    LEFT / RIGHT speed  5 - 25 %
    G            gripper open / close (if the Nano gripper board is connected)
    0-9          save the tool position as the ArUco marker with that ID
                 (only IDs listed in MARKER_POSITIONS_MM in vision_config.py)
    P            save the park pose
    B            save the next chessboard corner square (a1 -> h1 -> h8 -> a8);
                 after the 4th, chess_board.json is written with all 64 squares
    N            start the corner sequence again
    ENTER        write robot_calib.json (same file as `main.py teach`) and show fit errors
    ESC          STOP the current move (firmware 8.1; on 6.x use the red E-stop)
    Q            quit

Safety: one move at a time, speed capped at 25 %, steps capped at 50 mm, and when a
calibration exists the tool may not go below the taught mat height minus 3 mm.
The HMI must be closed (one program per serial port).
"""
import argparse
import json
import os
import threading
import time

import pygame

import teach_robot as teach
import chess_board as cb
import vision_config as cfg
from robot_backend import connect, RobotFault, MAX_SPEED

STEPS = [1, 2, 5, 10, 20, 50]
V_MIN_Z = 120.0   # mm: re-orient the gripper only well above the table
FLOOR_MARGIN_MM = 3.0

BG, FG, DIM = (24, 26, 32), (230, 232, 236), (140, 146, 158)
GREEN, RED, AMBER, BLUE = (90, 200, 120), (235, 90, 90), (240, 180, 70), (110, 160, 240)


def jog_vector(keys):
    """Direction from held keys: (dx, dy, dz) in units of one step."""
    dx = (keys[pygame.K_w] - keys[pygame.K_s])
    dy = (keys[pygame.K_a] - keys[pygame.K_d])
    dz = keys[pygame.K_SPACE] - (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])
    return dx, dy, dz


class Jogger:
    def __init__(self, arm, floor_z):
        self.arm = arm
        self.floor_z = floor_z
        self.pose = arm.pose()
        # Orientation is captured ONCE and sent unchanged with every jog. Re-sending the
        # read-back orientation lets small reporting/rounding errors accumulate, so the
        # gripper slowly rotates and never comes back.
        self.hold_rot = list(self.pose["rot"])
        self.worker = None
        self.message = "Ready"
        self.msg_colour = GREEN
        self.lock = threading.Lock()

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def say(self, text, colour=FG):
        self.message, self.msg_colour = text, colour

    def start(self, fn, *a):
        self.arm.clear_stop()
        self.worker = threading.Thread(target=self._run, args=(fn,) + a, daemon=True)
        self.worker.start()

    def _run(self, fn, *a):
        try:
            fn(*a)
        except (RobotFault, ValueError) as e:
            self.say(f"Refused/stopped: {e}", RED)
            try:
                self.pose = self.arm.pose()
            except Exception:
                pass

    def step(self, dx, dy, dz, step_mm, speed):
        p = self.pose
        x, y, z = p["x"] + dx * step_mm, p["y"] + dy * step_mm, p["z"] + dz * step_mm
        if self.floor_z is not None and z < self.floor_z:
            self.say(f"Refused: Z {z:.1f} is below the mat floor ({self.floor_z:.1f})", RED)
            return
        self.pose = self.arm.move_linear(x, y, z, self.hold_rot, speed)
        self.say(f"Moved to X {x:.1f}  Y {y:.1f}  Z {z:.1f}", FG)


def draw(screen, fonts, j, step_mm, speed, grip, saved, fw, fit_lines, corners, board_lines):
    big, mid, small = fonts
    screen.fill(BG)
    p = j.pose
    y = 18

    def text(s, font, colour, x=20):
        nonlocal y
        screen.blit(font.render(s, True, colour), (x, y))
        y += font.get_linesize()

    text(f"AR4 jog  (firmware {fw})", mid, DIM)
    text(f"X {p['x']:7.1f}   Y {p['y']:7.1f}   Z {p['z']:7.1f}  mm", big, FG)
    j5 = p["joints"][4]
    hr = j.hold_rot
    text(f"orientation locked at {hr[0]:.1f} {hr[1]:.1f} {hr[2]:.1f}   "
         f"(robot reports {p['rot'][0]:.1f} {p['rot'][1]:.1f} {p['rot'][2]:.1f})   R = re-lock",
         small, DIM)
    text("J " + "  ".join(f"{a:6.1f}" for a in p["joints"]), small, DIM)
    if abs(j5) < 5:
        text("J5 near 0: wrist singularity zone - the wrist may flip (8.1 fixes this)", small, AMBER)
    if abs(p["joints"][3]) > 90:
        text(f"J4 at {p['joints'][3]:.0f} deg: wrist twisted - press V (gripper straight down) "
             "at a safe height", small, AMBER)
    y += 6
    state = "MOVING" if j.busy() else "idle"
    text(f"step {step_mm} mm    speed {speed}%    {state}    gripper: {grip}", mid,
         AMBER if j.busy() else GREEN)
    if j.floor_z is not None:
        text(f"floor Z {j.floor_z:.1f} (taught mat height - {FLOOR_MARGIN_MM:.0f})", small, DIM)
    text(j.message, mid, j.msg_colour)
    y += 6
    ids = sorted(cfg.MARKER_POSITIONS_MM)
    names = [str(k) for k in ids if k in saved] + (["park"] if "park" in saved else [])
    missing = [str(k) for k in ids if k not in saved]
    if missing:
        names.append("(still to teach: " + ", ".join(missing) + ")")
    text(f"saved: {', '.join(names) or 'none'}", small, BLUE)
    for line in fit_lines:
        text(line, small, AMBER if "Large difference" in line else BLUE)
    done = [c for c in cb.CORNERS if c in corners]
    nxt = next((c for c in cb.CORNERS if c not in corners), None)
    text("board corners: " + (", ".join(done) or "none") +
         (f"   next: {nxt} (press B over its centre)" if nxt else "   - all taught"), small, BLUE)
    for line in board_lines:
        text(line, small, AMBER if line.startswith("!") else BLUE)
    y += 8
    for line in ["W/S +X/-X   A/D +Y/-Y   SPACE up   CTRL down   (hold to repeat)",
                 "UP/DOWN step  LEFT/RIGHT speed  G gripper  R re-lock  V straight down  Z/X turn  ESC stop  Q quit",
                 f"{' '.join(str(k) for k in sorted(cfg.MARKER_POSITIONS_MM))} save marker   "
                 "P save park   ENTER write calibration   B board corner   N restart corners"]:
        text(line, small, DIM)
    pygame.display.flip()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="main.py jog")
    ap.add_argument("--port", default=None)
    ap.add_argument("--firmware", choices=["8", "6"], default=None)
    ap.add_argument("--gripper-port", default=None, help="Nano port, e.g. /dev/ttyUSB0")
    ap.add_argument("--no-gripper", action="store_true")
    args = ap.parse_args(argv)

    calib = None
    if os.path.exists(cfg.CALIB_FILE):
        with open(cfg.CALIB_FILE) as f:
            calib = json.load(f)
    fw = args.firmware or (calib or {}).get("firmware", cfg.FIRMWARE)
    floor_z = calib["board_z"] - FLOOR_MARGIN_MM if calib else None

    arm = connect(fw, args.port, cfg.LOOP_MODE)
    j = Jogger(arm, floor_z)
    saved = teach.load_saved()
    fit_lines = []
    corners, board_lines = {}, []

    grip, grip_state = None, "not connected"
    if not args.no_gripper:
        try:
            from gripper import Gripper
            grip = Gripper(args.gripper_port)
            grip_state = f"ready ({grip.protocol}, {grip.port})"
        except Exception as e:
            print("Gripper not available:", e)

    pygame.display.init()   # no pygame.init(): skips audio (avoids ALSA errors)
    pygame.font.init()
    screen = pygame.display.set_mode((900, 430))
    pygame.display.set_caption("AR4 jog - click here so the keys go to this window")
    fonts = (pygame.font.SysFont("monospace", 30, bold=True),
             pygame.font.SysFont("monospace", 20),
             pygame.font.SysFont("monospace", 16))
    clock = pygame.time.Clock()
    step_i, speed = STEPS.index(10), min(cfg.SPEED_PCT, MAX_SPEED)
    running = True
    try:
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k == pygame.K_q:
                        running = False
                    elif k == pygame.K_ESCAPE:
                        if j.busy():
                            arm.stop()
                            j.say("STOP sent" if arm.supports_stop
                                  else "No software stop on 6.x - use the red E-stop", AMBER)
                    elif k == pygame.K_UP:
                        step_i = min(step_i + 1, len(STEPS) - 1)
                    elif k == pygame.K_DOWN:
                        step_i = max(step_i - 1, 0)
                    elif k == pygame.K_RIGHT:
                        speed = min(speed + 5, MAX_SPEED)
                    elif k == pygame.K_LEFT:
                        speed = max(speed - 5, 5)
                    elif j.busy():
                        continue
                    elif k == pygame.K_g:
                        if grip is None:
                            j.say("No gripper board connected", AMBER)
                        else:
                            try:
                                grip.toggle()
                                grip_state = grip.state
                                j.say(f"Gripper {grip.state}", FG)
                            except Exception as e:
                                j.say(f"Gripper error: {e}", RED)
                    elif (k == pygame.K_p or (pygame.K_0 <= k <= pygame.K_9)):
                        name = "park" if k == pygame.K_p else k - pygame.K_0
                        if name != "park" and name not in cfg.MARKER_POSITIONS_MM:
                            j.say(f"Marker {name} is not in MARKER_POSITIONS_MM "
                                  f"{sorted(cfg.MARKER_POSITIONS_MM)}", AMBER)
                            continue
                        try:
                            saved[name] = arm.pose()
                            saved[name]["rot"] = list(j.hold_rot)
                            j.pose = saved[name]
                            j.say(f"Saved {name} at X {j.pose['x']:.1f} Y {j.pose['y']:.1f} "
                                  f"Z {j.pose['z']:.1f}", GREEN)
                        except RobotFault as e:
                            j.say(f"Could not read pose: {e}", RED)
                    elif k == pygame.K_r:
                        try:
                            j.pose = arm.pose()
                            j.hold_rot = list(j.pose["rot"])
                            j.say("Orientation re-locked to the current pose", GREEN)
                        except RobotFault as e:
                            j.say(f"Could not read pose: {e}", RED)
                    elif k in (pygame.K_v, pygame.K_z, pygame.K_x):
                        p = j.pose
                        if k == pygame.K_v:
                            if p["z"] < V_MIN_Z:
                                j.say(f"Raise above Z {V_MIN_Z:.0f} first (SPACE), then press V", AMBER)
                                continue
                            new_rot = [0.0, 180.0, 0.0]
                            what = "Gripper set straight down"
                        else:
                            new_rot = list(j.hold_rot)
                            new_rot[0] += -5.0 if k == pygame.K_z else 5.0
                            what = f"Gripper turned to Rz {new_rot[0]:.0f}"

                        def reorient(rot=new_rot, msg=what, x=p["x"], y=p["y"], z=p["z"]):
                            j.pose = arm.move_linear(x, y, z, rot, min(speed, 10))
                            j.hold_rot = list(rot)
                            j.say(msg + " - check it by eye; if wrong, press the red E-stop", GREEN)

                        j.start(reorient)
                    elif k == pygame.K_n:
                        corners, board_lines = {}, []
                        j.say("Corner sequence restarted: go to a1 and press B", FG)
                    elif k == pygame.K_b:
                        nxt = next((c for c in cb.CORNERS if c not in corners), None)
                        if nxt is None:
                            j.say("All 4 corners taught - press N to start again", AMBER)
                            continue
                        try:
                            corners[nxt] = arm.pose()
                            corners[nxt]["rot"] = list(j.hold_rot)
                        except RobotFault as e:
                            j.say(f"Could not read pose: {e}", RED)
                            continue
                        j.pose = corners[nxt]
                        j.say(f"Saved corner {nxt} at X {j.pose['x']:.1f} Y {j.pose['y']:.1f}", GREEN)
                        if len(corners) == 4:
                            board = cb.ChessBoard(corners)
                            lines, warn = board.save(fw)
                            board_lines = lines + ["! " + w for w in warn]
                            j.say("Wrote chess_board.json (64 squares)" +
                                  (" - check the warnings" if warn else ""), AMBER if warn else GREEN)
                    elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if "park" not in saved:
                            j.say("Save a park pose first (key P), out of the camera view", AMBER)
                            continue
                        try:
                            ids, err = teach.write_calib(saved, fw)
                            fit_lines = teach.calib_report(ids, err)
                            j.say(f"Wrote {os.path.basename(cfg.CALIB_FILE)}", GREEN)
                        except ValueError as e:
                            j.say(str(e), RED)
            if not j.busy():
                dx, dy, dz = jog_vector(pygame.key.get_pressed())
                if dx or dy or dz:
                    j.start(j.step, dx, dy, dz, STEPS[step_i], speed)
            draw(screen, fonts, j, STEPS[step_i], speed, grip_state, saved, fw, fit_lines,
                 corners, board_lines)
            clock.tick(30)
    finally:
        if j.busy():
            arm.stop()
            j.worker.join(timeout=5)
        arm.close()
        if grip is not None:
            grip.shutdown()
        pygame.quit()


if __name__ == "__main__":
    main()
