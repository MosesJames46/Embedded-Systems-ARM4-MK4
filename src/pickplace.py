"""Pick up and put down chess pieces on the taught board (chess_board.json).

Heights are relative to what you TAUGHT: at each corner you jogged the gripper tip to
about 10 mm above the board, and that height is interpolated for every square.
    GRIP_MM   tip height above the board when the fingers close (default 15)
    TAUGHT_MM how high above the board you taught the corners (default 10)
A pick is:  rise to travel height -> over the square -> open -> down to grip height
            -> close -> back up to travel height.
A place is the same with open instead of close.

No gripper board (Nano) connected? Use --nogripper: the program pauses and you open or
close the gripper yourself (HMI or by hand) and press Enter - good for testing the motion.
"""
import vision_config as cfg
from robot_backend import RobotFault

TAUGHT_MM = 10.0       # tip height above the board when the 4 corners were taught
GRIP_MM = 15.0         # tip height above the board when gripping a piece
TRAVEL_MM = 120.0      # travel height above the highest corner (clears a king held in the gripper)
DROP_GRID = (9.0, 4.5)  # captured pieces: 2 squares past the h-file, level with rank 5

# Per piece type: fingertip height above the board when gripping, and the servo angle to
# close to (None = the gripper's saved closed angle). Tune with `board` -> tune k height 30
import json as _json
import os as _os
PROFILE_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "grip_profile.json")
PIECES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}


def load_profile():
    prof = {t: {"height": GRIP_MM, "close": None, "open": None} for t in PIECES}
    try:
        with open(PROFILE_FILE) as f:
            for t, v in _json.load(f).items():
                if t in prof:
                    prof[t].update(v)
    except (OSError, ValueError):
        pass
    return prof


def save_profile(prof):
    with open(PROFILE_FILE, "w") as f:
        _json.dump(prof, f, indent=2)


class ManualGripper:
    """Stand-in when the Nano is not connected: asks the operator."""
    state = None

    def open(self, pos=None):
        input("    >> OPEN the gripper, then press Enter ")
        self.state = "open"

    def close(self, pos=None):
        input("    >> CLOSE the gripper on the piece, then press Enter ")
        self.state = "closed"

    def shutdown(self):
        pass


def make_gripper(nogripper=False, port=None):
    if nogripper:
        return ManualGripper()
    from gripper import Gripper
    try:
        g = Gripper(port=port)
        print(f"Gripper board on {g.port} ({g.protocol} protocol)")
        return g
    except Exception as e:
        print(f"Gripper not available ({e}).\n  -> run `python main.py gripper` to find and test it."
              "\n  Falling back to manual open/close prompts for now.")
        return ManualGripper()


class PickPlace:
    def __init__(self, arm, board, gripper, speed=None, grip_mm=GRIP_MM, travel_mm=TRAVEL_MM):
        self.arm, self.board, self.grip = arm, board, gripper
        self.speed = speed or cfg.SPEED_PCT
        self.grip_mm, self.travel_mm = grip_mm, travel_mm
        self.rot = board.tool_rot           # straight down, as taught at a1
        self.z_now = self.travel_z
        self.profile = load_profile()
        self.held = None        # height (mm above board) the held piece was gripped at

    @property
    def travel_z(self):
        return self.board.max_z - TAUGHT_MM + self.travel_mm

    def _z_at(self, f, r, above_mm):
        return self.board.grid_z(f, r) - TAUGHT_MM + above_mm

    def rise(self):
        p = self.arm.pose()
        if p["z"] < self.travel_z - 1:
            self.arm.move_linear(p["x"], p["y"], self.travel_z, self.rot, self.speed)
        self.z_now = max(p["z"], self.travel_z)

    def _over(self, f, r):
        """Get over square (f, r) at travel height. Squares at the edge of reach (h8) cannot
        be reached that high, so step the travel height down 20 mm at a time (a refused
        move is planned first and never starts, so retrying is safe)."""
        x, y = self.board.grid_to_robot(f, r)
        last = None
        for drop in (0, 20, 40, 60):
            tz = self.travel_z - drop
            try:
                p = self.arm.pose()
                if abs(p["z"] - tz) > 1:
                    self.arm.move_linear(p["x"], p["y"], tz, self.rot, self.speed)
                self.arm.move_linear(x, y, tz, self.rot, self.speed)
                if drop:
                    print(f"    (travel lowered {drop} mm here - edge of reach; watch tall pieces)")
                self.z_now = tz
                return x, y
            except RobotFault as e:
                if "refused" not in str(e).lower():
                    raise
                last = e
        raise last

    def _grip_for(self, piece):
        if piece:
            pr = self.profile[piece.lower()]
            return float(pr["height"]), pr["close"], pr.get("open")
        return self.grip_mm, None, None

    adjust = False          # --adjust: pause above each grip/release to centre it
    firmware = "6"
    CHECK_MM = 25.0         # pause this far above the grip height to check centring

    def _adjust_here(self, f, r, x, y, height):
        """Stop just above the piece; the operator nudges until centred; the corrected
        spot is saved to the board for that square. Returns the (possibly new) x, y."""
        import numpy as np
        import chess_board as cb
        name = cb.square_name(int(round(f)), int(round(r)))
        z = self._z_at(f, r, height + self.CHECK_MM)
        self.arm.move_linear(x, y, z, self.rot, self.speed)
        moved = False
        here = np.array([x, y], float)
        if getattr(self, "adjust_gui", True):
            here, moved = self._adjust_window(name, f, r, here, z)
        while not getattr(self, "adjust_gui", True):
            ans = input(f"    {name}: centred? Enter = go | a/h/near/far + mm (e.g. h 2) > ").strip().lower()
            if ans == "":
                break
            parts = ans.split()
            dirs = {"a": (-1, 0), "h": (1, 0), "near": (0, -1), "far": (0, 1)}
            if len(parts) != 2 or parts[0] not in dirs:
                print("    e.g.  a 2   h 1   near 1.5   far 3")
                continue
            try:
                mm = float(parts[1])
            except ValueError:
                continue
            if abs(mm) > 15:
                print("    at most 15 mm at a time")
                continue
            d = dirs[parts[0]]
            v = np.array(self.board.grid_to_robot(f + d[0], r + d[1])) - np.array(self.board.grid_to_robot(f, r))
            here = here + v / np.linalg.norm(v) * mm
            try:
                self.arm.move_linear(float(here[0]), float(here[1]), z, self.rot, self.speed)
                moved = True
            except RobotFault as e:
                print("    ", e)
                here = here - v / np.linalg.norm(v) * mm
        if moved and float(f).is_integer() and float(r).is_integer() and 0 <= f <= 7 and 0 <= r <= 7:
            self.board.set_point(name, float(here[0]), float(here[1]))
            self.board.save(self.firmware)
            print(f"    saved the corrected centre of {name}")
        return float(here[0]), float(here[1])

    def _adjust_window(self, name, f, r, here, z):
        """Small keyboard window for centring: W/S toward rank 8/1, A/D toward a/h,
        UP/DOWN step, ENTER go. Returns (xy, moved)."""
        import json
        import numpy as np
        import pygame
        steps = [0.5, 1.0, 2.0, 5.0]
        si = 1
        try:
            with open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "jog_settings.json")) as fh:
                flip = bool(json.load(fh).get("boardteach_flip", False))
        except (OSError, ValueError):
            flip = False
        start = here.copy()
        u = np.array(self.board.grid_to_robot(f + 1, r)) - np.array(self.board.grid_to_robot(f, r))
        v = np.array(self.board.grid_to_robot(f, r + 1)) - np.array(self.board.grid_to_robot(f, r))
        u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((640, 230))
        pygame.display.set_caption(f"Centre {name} - ENTER to go")
        big = pygame.font.SysFont("monospace", 30, bold=True)
        mid = pygame.font.SysFont("monospace", 19)
        msg = ""

        def draw():
            d = here - start
            screen.fill((24, 26, 32))
            screen.blit(big.render(f"{name}: centre the gripper", True, (230, 230, 230)), (20, 15))
            screen.blit(mid.render(f"moved {d @ u:+.1f} mm toward h   {d @ v:+.1f} mm toward rank 8",
                                   True, (90, 200, 120)), (20, 65))
            screen.blit(mid.render(f"step {steps[si]} mm (UP/DOWN)", True, (230, 230, 230)), (20, 95))
            ad = "A toward h / D toward a" if flip else "A toward a / D toward h"
            screen.blit(mid.render(f"W toward rank 8 / S toward rank 1   {ad}", True, (110, 160, 240)), (20, 125))
            screen.blit(mid.render("ENTER = go     BACKSPACE = undo all nudges", True, (130, 130, 140)), (20, 155))
            screen.blit(mid.render(msg[:60], True, (240, 180, 70)), (20, 190))
            pygame.display.flip()

        moved = False
        try:
            draw()
            while True:
                ev = pygame.event.wait()
                if ev.type == pygame.QUIT:
                    break
                if ev.type != pygame.KEYDOWN:
                    continue
                k = ev.key
                if k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    break
                vec = None
                if k == pygame.K_w:
                    vec = v
                elif k == pygame.K_s:
                    vec = -v
                elif k in (pygame.K_a, pygame.K_d):
                    toward_a = (k == pygame.K_a) != flip
                    vec = -u if toward_a else u
                elif k == pygame.K_UP:
                    si = min(si + 1, len(steps) - 1)
                elif k == pygame.K_DOWN:
                    si = max(si - 1, 0)
                elif k == pygame.K_BACKSPACE:
                    vec = start - here
                    if np.linalg.norm(vec) < 1e-6:
                        vec = None
                    else:
                        vec = vec / steps[si]          # scaled back below
                msg = ""
                if vec is not None:
                    msg = "MOVING ..."
                    draw()
                    msg = ""
                    target = here + vec * steps[si]
                    if np.linalg.norm(target - start) > 15:
                        msg = "at most 15 mm from the computed centre"
                    else:
                        try:
                            self.arm.move_linear(float(target[0]), float(target[1]), z, self.rot, self.speed)
                            here = target
                            moved = bool(np.linalg.norm(here - start) > 1e-6)
                        except RobotFault as e:
                            msg = str(e)
                    pending = pygame.event.get(pygame.KEYDOWN)     # keep only the last press
                    if pending:
                        pygame.event.post(pending[-1])
                draw()
        finally:
            pygame.display.quit()
        return here, moved

    def _down_up(self, f, r, x, y, action, height):
        if self.adjust and float(f).is_integer() and 0 <= f <= 7 and 0 <= r <= 7:
            x, y = self._adjust_here(f, r, x, y, height)
        z = self._z_at(f, r, height)
        self.arm.move_linear(x, y, z, self.rot, self.speed)
        action()
        self.arm.move_linear(x, y, self.z_now, self.rot, self.speed)

    def pick_grid(self, f, r, piece=None):
        height, close, opn = self._grip_for(piece)
        x, y = self._over(f, r)
        self.grip.open(pos=opn)
        self._down_up(f, r, x, y, lambda: self.grip.close(pos=close), height)
        self.held, self.held_open = height, opn

    def place_grid(self, f, r):
        x, y = self._over(f, r)
        # put it down at the height it was picked at (+1 mm so it does not press)
        height = (self.held if self.held is not None else self.grip_mm) + 1.0
        opn = getattr(self, "held_open", None)
        self._down_up(f, r, x, y, lambda: self.grip.open(pos=opn), height)
        self.held = self.held_open = None

    def pick(self, square, piece=None):
        print(f"  pick  {square}" + (f" ({PIECES[piece.lower()]})" if piece else ""))
        self.pick_grid(*_idx(square), piece)

    def place(self, square):
        print(f"  place {square}")
        self.place_grid(*_idx(square))

    def move(self, src, dst, piece=None):
        self.pick(src, piece)
        self.place(dst)

    def remove(self, square, piece=None):
        """Take the piece on square off the board to the drop zone."""
        self.pick(square, piece)
        print("  drop  captured piece")
        f, r = DROP_GRID
        x, y = self._over(f, r)
        # release a little higher than a normal place: the drop zone is not taught
        z = self._z_at(f, r, self.held + 15)
        self.arm.move_linear(x, y, z, self.rot, self.speed)
        self.grip.open(pos=getattr(self, "held_open", None))
        self.held = self.held_open = None
        self.arm.move_linear(x, y, self.z_now, self.rot, self.speed)


def _idx(square):
    import chess_board as cb
    return cb.square_index(square)
