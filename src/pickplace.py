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

    def _down_up(self, f, r, x, y, action, height):
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
