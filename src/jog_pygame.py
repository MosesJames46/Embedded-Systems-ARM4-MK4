"""Keyboard jog controller for the AR4 (pygame window).

    python main.py jog                 (firmware from robot_calib.json, else vision_config)
    python main.py jog --firmware 6

Hold a key to keep moving; each step is one straight-line move of the current step size.
The robot frame: X = away from the base, Y = to the robot's left, Z = up.

    M            switch mode:  BASE (default) -> XYZ -> JOINT -> ROTATE -> BASE
    ROTATE mode  the gripper turns in place (the tool point stays put):
                 W / S  tilt the tip away from / toward the robot
                 A / D  tilt the tip to the robot's left / right
                 SPACE / CTRL  turn about the vertical
                 UP / DOWN  step 0.5 1 2 5 10 deg.  The window always shows how many
                 degrees the gripper is from straight down (V snaps it to 0)
    JOINT mode   TAB  choose the joint (J1..J6),  W / S  move it + / -
                 UP / DOWN  step 0.5 1 2 5 10 deg.  Only that joint moves; the gripper may
                 tilt - press V (above Z 120) to point it straight down again
    BASE mode    A / D  turn the base (J1 ONLY) left / right - the arm sweeps sideways
                 W / S  reach out / pull in along the arm's current direction
                 SPACE / CTRL  up / down
    XYZ mode     straight lines in robot X / Y (several joints move together):
    W / S        forward / back    }  as seen from where YOU stand - press O to tell the
    A / D        left / right      }  window where that is (behind, opposite, left, right
                                      of the robot); the window shows which robot axis
                                      each key drives. Default: behind the robot, i.e.
                                      W = X+ (away from the base), A = Y+ (robot's left)
    O            change where you stand (saved for next time)
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

import numpy as np
import pygame

import teach_robot as teach
import chess_board as cb
import vision_config as cfg
from robot_backend import connect, RobotFault, MAX_SPEED

STEPS = [1, 2, 5, 10, 20, 50]
V_MIN_Z = 120.0   # mm: re-orient the gripper only well above the table
J1_LIMITS = (-165.0, 165.0)
# Annin defaults (same as the HMI sliders); the firmware enforces its own configured limits too
JOINT_LIMITS = [(-165, 165), (-40, 88), (-87, 50), (-178, 178), (-103, 103), (-178, 178)]
JOINT_STEPS = [0.5, 1, 2, 5, 10]
FLOOR_MARGIN_MM = 3.0

BG, FG, DIM = (24, 26, 32), (230, 232, 236), (140, 146, 158)
GREEN, RED, AMBER, BLUE = (90, 200, 120), (235, 90, 90), (240, 180, 70), (110, 160, 240)


# Where the operator stands, so W = "away from me" and A = "my left" wherever that is.
# heading = direction the operator faces, in the robot frame (deg, X+ = 0, Y+ = 90).
VIEWS = [("BEHIND the robot (facing the board)", 0),
         ("OPPOSITE the robot (facing it)", 180),
         ("on the robot's RIGHT side", 90),
         ("on the robot's LEFT side", 270)]
VIEW_FILE = os.path.join(cfg.HERE, "jog_settings.json")


def load_view():
    try:
        with open(VIEW_FILE) as f:
            return int(json.load(f).get("view", 0)) % len(VIEWS)
    except (OSError, ValueError):
        return 0


def save_view(i):
    try:
        with open(VIEW_FILE, "w") as f:
            json.dump({"view": i}, f)
    except OSError:
        pass


def jog_vector(keys, view=0):
    """Held keys -> (dx, dy, dz) in the ROBOT frame, in units of one step.
    W/S = forward/back and A/D = left/right as seen from where the operator stands."""
    fwd = keys[pygame.K_w] - keys[pygame.K_s]
    left = keys[pygame.K_a] - keys[pygame.K_d]
    dz = keys[pygame.K_SPACE] - (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])
    th = np.radians(VIEWS[view][1])
    dx = fwd * np.cos(th) - left * np.sin(th)
    dy = fwd * np.sin(th) + left * np.cos(th)
    return round(float(dx), 6), round(float(dy), 6), dz


def key_meaning(view):
    """Text: what W and A do in robot coordinates for this view."""
    names = {(1, 0): "X+ (away from robot)", (-1, 0): "X- (toward robot)",
             (0, 1): "Y+ (robot's left)", (0, -1): "Y- (robot's right)"}
    th = np.radians(VIEWS[view][1])
    w = (round(np.cos(th)), round(np.sin(th)))
    a = (round(-np.sin(th)), round(np.cos(th)))
    return f"W = {names[w]}   A = {names[a]}"


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

    def turn_base(self, sign, step_mm, speed):
        """Rotate J1 ONLY, so the gripper sweeps sideways by about step_mm along an arc."""
        p = self.pose
        q = list(p["joints"])
        r = max(float(np.hypot(p["x"], p["y"])), 100.0)
        dtheta = float(np.degrees(step_mm / r)) * sign
        lo, hi = J1_LIMITS
        if not (lo <= q[0] + dtheta <= hi):
            self.say(f"Refused: J1 would be at {q[0] + dtheta:.0f} deg (limit {lo:.0f}..{hi:.0f})", RED)
            return
        q[0] += dtheta
        self.pose = self.arm.move_joints(q, speed)
        # the whole arm turned about the vertical, so the held gripper direction turns with it
        if getattr(self.arm, "local_ik", False):
            import kinematics as K
            self.hold_rot = K.rotate_about_vertical(self.hold_rot, dtheta)
        else:
            self.hold_rot = list(self.pose["rot"])
        self.say(f"Base turned {dtheta:+.1f} deg (J1 = {q[0]:.1f})", FG)

    def move_one_joint(self, index, sign, step_deg, speed):
        """Move ONE joint by step_deg. The gripper orientation follows (re-locked after)."""
        p = self.pose
        q = list(p["joints"])
        lo, hi = JOINT_LIMITS[index]
        target = q[index] + sign * step_deg
        if not (lo <= target <= hi):
            self.say(f"Refused: J{index + 1} would be at {target:.1f} deg (limit {lo:.0f}..{hi:.0f})", RED)
            return
        q[index] = target
        new = self.arm.move_joints(q, speed)
        if self.floor_z is not None and new["z"] < self.floor_z:
            self.say(f"Warning: tool now below the mat floor (Z {new['z']:.0f}) - raise it", AMBER)
        self.pose = new
        self.hold_rot = list(new["rot"])     # keep whatever orientation this left us in
        self.say(f"J{index + 1} -> {target:.1f} deg", FG)

    def rotate(self, axis, sign, step_deg, speed):
        """Tilt / turn the gripper about a world axis; the tool point stays where it is."""
        import kinematics as K
        p = self.pose
        new_rot = K.rotate_about_world(self.hold_rot, axis, sign * step_deg)
        self.pose = self.arm.move_linear(p["x"], p["y"], p["z"], new_rot, speed)
        self.hold_rot = list(new_rot)
        self.say(f"Gripper {K.tilt_from_down(new_rot):.1f} deg from straight down", FG)

    def step(self, dx, dy, dz, step_mm, speed):
        p = self.pose
        x, y, z = p["x"] + dx * step_mm, p["y"] + dy * step_mm, p["z"] + dz * step_mm
        if self.floor_z is not None and z < self.floor_z:
            self.say(f"Refused: Z {z:.1f} is below the mat floor ({self.floor_z:.1f})", RED)
            return
        self.pose = self.arm.move_linear(x, y, z, self.hold_rot, speed)
        self.say(f"Moved to X {x:.1f}  Y {y:.1f}  Z {z:.1f}", FG)


def draw(screen, fonts, j, step_mm, speed, grip, saved, fw, fit_lines, corners, board_lines, view, mode,
         sel_joint=0, jstep_i=2):
    big, mid, small = fonts
    screen.fill(BG)
    p = j.pose
    y = 18

    def text(s, font, colour, x=20):
        nonlocal y
        screen.blit(font.render(s, True, colour), (x, y))
        y += font.get_linesize()

    text(f"AR4 jog  (firmware {fw}{', laptop kinematics - no wrist flips' if getattr(j.arm, 'local_ik', False) else ''})",
         mid, DIM)
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
    if getattr(j.arm, "local_ik", False):
        import kinematics as K
        tilt = K.tilt_from_down(j.hold_rot)
        text(f"gripper {tilt:.1f} deg from straight down" + ("  (pointing down)" if tilt < 1 else ""),
             small, GREEN if tilt < 1 else AMBER)
    if mode == "rotate":
        text(f"ROTATE mode: W/S tilt away/toward robot  A/D tilt left/right  SPACE/CTRL turn   "
             f"step {JOINT_STEPS[jstep_i]} deg   (M = BASE mode)", small, GREEN)
    elif mode == "joint":
        text(f"JOINT mode: J{sel_joint + 1} selected ({p['joints'][sel_joint]:.1f} deg)  W/S move it  "
             f"TAB next joint  step {JOINT_STEPS[jstep_i]} deg   (M = BASE mode)", small, GREEN)
    elif mode == "base":
        text(f"BASE mode: A/D turn the base (J1 only)  W/S reach out/in  SPACE/CTRL up/down   "
             f"(M = XYZ mode, O = where you stand: {VIEWS[view][0].split(' (')[0]})", small, GREEN)
    else:
        text(f"XYZ mode - you stand {VIEWS[view][0]}:  {key_meaning(view)}   (M = BASE mode, O = change)",
             small, GREEN)
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
    for line in ["BASE: A/D turn base  W/S reach out/in   XYZ: straight lines   M = switch mode   SPACE up  CTRL down",
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
    floor_z = calib["board_z"] - FLOOR_MARGIN_MM if calib and "board_z" in calib else None

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
    screen = pygame.display.set_mode((980, 450))
    pygame.display.set_caption("AR4 jog - click here so the keys go to this window")
    fonts = (pygame.font.SysFont("monospace", 30, bold=True),
             pygame.font.SysFont("monospace", 20),
             pygame.font.SysFont("monospace", 16))
    clock = pygame.time.Clock()
    step_i, speed = STEPS.index(10), min(cfg.SPEED_PCT, MAX_SPEED)
    view = load_view()
    mode = "base"
    sel_joint, jstep_i = 1, 2          # JOINT mode: J2 selected, 2 deg steps
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
                    elif k == pygame.K_m:
                        mode = {"base": "xyz", "xyz": "joint", "joint": "rotate", "rotate": "base"}[mode]
                        if mode == "rotate" and not getattr(arm, "local_ik", False):
                            mode = "base"
                            j.say("ROTATE mode needs the 6.x laptop kinematics", AMBER)
                            continue
                        j.say({"xyz": "XYZ mode: straight lines in robot X/Y",
                               "rotate": "ROTATE mode: WASD tilt, SPACE/CTRL turn - the gripper turns in place",
                               "joint": "JOINT mode: TAB picks the joint, W/S move only that joint",
                               "base": "BASE mode: A/D turn J1 only, W/S reach out/in"}[mode], GREEN)
                    elif k == pygame.K_TAB and mode == "joint":
                        sel_joint = (sel_joint + 1) % 6
                        j.say(f"J{sel_joint + 1} selected", FG)
                    elif k == pygame.K_o:
                        view = (view + 1) % len(VIEWS)
                        save_view(view)
                        j.say(f"You stand {VIEWS[view][0]} - {key_meaning(view)}", GREEN)
                    elif k == pygame.K_UP:
                        if mode in ("joint", "rotate"):
                            jstep_i = min(jstep_i + 1, len(JOINT_STEPS) - 1)
                        else:
                            step_i = min(step_i + 1, len(STEPS) - 1)
                    elif k == pygame.K_DOWN:
                        if mode in ("joint", "rotate"):
                            jstep_i = max(jstep_i - 1, 0)
                        else:
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
                            teach.save_points({name: saved[name]}, fw)   # written to disk now
                            j.pose = saved[name]
                            j.say(f"Saved {name} at X {j.pose['x']:.1f} Y {j.pose['y']:.1f} "
                                  f"Z {j.pose['z']:.1f} -> {os.path.basename(cfg.CALIB_FILE)}", GREEN)
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
                            if not getattr(arm, "local_ik", False):
                                j.say("Z/X turning needs the 6.x laptop kinematics", AMBER)
                                continue
                            import kinematics as K
                            new_rot = K.rotate_about_vertical(j.hold_rot, -5.0 if k == pygame.K_z else 5.0)
                            what = "Gripper turned 5 deg"

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
                keys = pygame.key.get_pressed()
                if mode == "rotate":
                    tilt_fwd = keys[pygame.K_w] - keys[pygame.K_s]
                    tilt_side = keys[pygame.K_a] - keys[pygame.K_d]
                    turn = keys[pygame.K_SPACE] - (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])
                    deg = JOINT_STEPS[jstep_i]
                    # checked numerically: tip away from robot (+X) = negative turn about Y,
                    # tip to the robot's left (+Y) = positive turn about X
                    if tilt_fwd:
                        j.start(j.rotate, "y", -tilt_fwd, deg, speed)
                    elif tilt_side:
                        j.start(j.rotate, "x", tilt_side, deg, speed)
                    elif turn:
                        j.start(j.rotate, "z", turn, deg, speed)
                elif mode == "joint":
                    sgn = keys[pygame.K_w] - keys[pygame.K_s]
                    if sgn:
                        j.start(j.move_one_joint, sel_joint, sgn, JOINT_STEPS[jstep_i], speed)
                elif mode == "base":
                    flip = -1 if VIEWS[view][1] == 180 else 1   # facing the robot: mirror
                    turn = (keys[pygame.K_a] - keys[pygame.K_d]) * flip
                    reach = (keys[pygame.K_w] - keys[pygame.K_s]) * flip
                    dz = keys[pygame.K_SPACE] - (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])
                    if turn:
                        j.start(j.turn_base, turn, STEPS[step_i], speed)
                    elif reach or dz:
                        phi = np.arctan2(j.pose["y"], j.pose["x"])   # along the arm's direction
                        j.start(j.step, reach * np.cos(phi), reach * np.sin(phi), dz,
                                STEPS[step_i], speed)
                else:
                    dx, dy, dz = jog_vector(keys, view)
                    if dx or dy or dz:
                        j.start(j.step, dx, dy, dz, STEPS[step_i], speed)
            draw(screen, fonts, j, STEPS[step_i], speed, grip_state, saved, fw, fit_lines,
                 corners, board_lines, view, mode, sel_joint, jstep_i)
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
