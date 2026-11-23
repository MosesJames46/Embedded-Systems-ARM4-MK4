"""Chessboard geometry: the 64 square centres in robot coordinates.

Taught by jogging the gripper over the centres of the 4 corner squares (a1, h1, h8, a8)
in the jog window (key B). A perspective map from (file, rank) grid indices to robot X/Y
is fitted through those 4 points, so the board may sit at any angle to the robot and
small skews are absorbed. Nothing is measured with a ruler.

    a1 = White's queen-side rook square (dark corner on White's LEFT)
    h1 = White's RIGHT corner,  h8 / a8 = the corners on Black's side
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD_FILE = os.path.join(HERE, "chess_board.json")
CORNERS = ["a1", "h1", "h8", "a8"]
FILES = "abcdefgh"
GRID = {"a1": (0, 0), "h1": (7, 0), "h8": (7, 7), "a8": (0, 7)}


def square_index(name):
    name = name.strip().lower()
    if len(name) != 2 or name[0] not in FILES or name[1] not in "12345678":
        raise ValueError(f"'{name}' is not a square (use a1 .. h8)")
    return FILES.index(name[0]), int(name[1]) - 1


def square_name(f, r):
    return f"{FILES[f]}{r + 1}"


class ChessBoard:
    def __init__(self, corners, extra=None):
        """corners: {"a1": pose, "h1": pose, "h8": pose, "a8": pose}; pose has x, y, z, rot.
        extra: {square: {"x", "y", optional "z"}} - more taught or nudged squares. The map
        goes exactly through every one of them; squares in between are corrected smoothly."""
        self.corners = corners
        self.extra = dict(extra or {})
        grid = np.float32([GRID[c] for c in CORNERS])
        robot = np.float32([[corners[c]["x"], corners[c]["y"]] for c in CORNERS])
        self.H = cv2.getPerspectiveTransform(grid, robot)          # grid -> robot
        self.Hinv = np.linalg.inv(self.H)                          # robot -> grid (first guess)
        self._c = {c: np.array([corners[c]["x"], corners[c]["y"]], float) for c in CORNERS}
        self.surface_z = float(np.mean([corners[c]["z"] for c in CORNERS]))
        # taught tool height at each corner; squares in between are interpolated, because
        # the reported Z for the same real height grows with reach (arm sag / calibration)
        self.corner_z = {c: float(corners[c]["z"]) for c in CORNERS}
        self._max_z0 = max(self.corner_z.values())
        # extra length added to the fingers AFTER the board was taught (popsicle sticks):
        # every height rises by this much so the new tips stay where the old tips were
        self.tool_ext = getattr(self, "tool_ext", 0.0)
        self.tool_rot = corners["a1"]["rot"]
        self._build_corrections()

    # --- extra taught points ---------------------------------------------------------
    def _build_corrections(self):
        """Residual of each extra point against the 4-corner map; the corners are anchors
        with zero residual. Corrections in between: inverse-distance weighting."""
        self._anchors = [(GRID[c], np.zeros(2), 0.0) for c in CORNERS]
        for sq, p in self.extra.items():
            f, r = square_index(sq)
            dxy = np.array([p["x"], p["y"]], float) - self._bilinear(f / 7.0, r / 7.0)
            dz = (float(p["z"]) - self._base_z(f, r)) if p.get("z") is not None else None
            self._anchors.append(((f, r), dxy, dz))

    def _correction(self, f, r):
        if len(self._anchors) == 4:
            return np.zeros(2), 0.0
        wxy, wz, sxy, sz = 0.0, 0.0, np.zeros(2), 0.0
        for (af, ar), dxy, dz in self._anchors:
            d2 = (f - af) ** 2 + (r - ar) ** 2
            if d2 < 1e-9:
                return dxy, (dz if dz is not None else self._z_corr_only(f, r))
            w = 1.0 / d2
            wxy += w
            sxy += w * dxy
            if dz is not None:
                wz += w
                sz += w * dz
        return sxy / wxy, (sz / wz if wz else 0.0)

    def _z_corr_only(self, f, r):
        wz = sz = 0.0
        for (af, ar), _, dz in self._anchors:
            d2 = (f - af) ** 2 + (r - ar) ** 2
            if dz is None or d2 < 1e-9:
                continue
            wz += 1.0 / d2
            sz += dz / d2
        return sz / wz if wz else 0.0

    def set_point(self, square, x, y, z=None):
        """Teach / nudge one square: the arm was centred over it at robot x, y (and, when
        given, z = tool height at the 10 mm teach height)."""
        square = square_name(*square_index(square))
        if square in CORNERS:
            self.corners[square]["x"], self.corners[square]["y"] = float(x), float(y)
            if z is not None:
                self.corners[square]["z"] = float(z)
            keep = self.extra
            self.__init__(self.corners, keep)
            return
        old = self.extra.get(square, {})
        self.extra[square] = {"x": float(x), "y": float(y),
                              "z": float(z) if z is not None else old.get("z")}
        self._build_corrections()

    def forget(self, square):
        self.extra.pop(square_name(*square_index(square)), None)
        self._build_corrections()

    def extra_report(self):
        out = []
        for (f, r), dxy, dz in self._anchors[4:]:
            out.append(f"{square_name(f, r)} corrects {np.linalg.norm(dxy):.1f} mm"
                       + (f", Z {dz:+.1f}" if dz is not None else ""))
        return out

    # --- conversions -------------------------------------------------------------
    def grid_to_robot(self, f, r):
        p = self._bilinear(f / 7.0, r / 7.0) + self._correction(f, r)[0]
        return float(p[0]), float(p[1])

    def _base_xy(self, f, r):
        """Bilinear between the 4 taught corners: squares evenly spaced along every edge.
        (Not a perspective map - the robot has no foreshortening. Its reported X/Y error
        grows with reach, so the taught corners form a slight trapezoid; bilinear follows
        that without bunching the squares toward one side.)"""
        p = self._bilinear(f / 7.0, r / 7.0)
        return float(p[0]), float(p[1])

    def _bilinear(self, u, v):
        c = self._c
        return ((1 - u) * (1 - v) * c["a1"] + u * (1 - v) * c["h1"]
                + u * v * c["h8"] + (1 - u) * v * c["a8"])

    def square_to_robot(self, name):
        return self.grid_to_robot(*square_index(name))

    @property
    def max_z(self):
        return self._max_z0 + self.tool_ext

    def grid_z(self, f, r):
        """Taught tool Z (the height you jogged to) at grid (f, r), plus any finger
        extension added since (tool_extension_mm)."""
        return self._base_z(f, r) + self._correction(f, r)[1] + self.tool_ext

    def _base_z(self, f, r):
        """Bilinear between the 4 corners."""
        u, v = min(max(f / 7.0, 0.0), 1.0), min(max(r / 7.0, 0.0), 1.0)
        z = self.corner_z
        return float((1 - u) * (1 - v) * z["a1"] + u * (1 - v) * z["h1"]
                     + u * v * z["h8"] + (1 - u) * v * z["a8"])

    def square_z(self, name):
        return self.grid_z(*square_index(name))

    def robot_to_grid(self, x, y):
        g = cv2.perspectiveTransform(np.float32([[[x, y]]]), self.Hinv)[0, 0]
        u, v = g[0] / 7.0, g[1] / 7.0          # perspective guess, refined by Newton
        c, t = self._c, np.array([x, y], float)
        for _ in range(20):
            e = self._bilinear(u, v) - t
            if np.linalg.norm(e) < 1e-4:
                break
            du = (1 - v) * (c["h1"] - c["a1"]) + v * (c["h8"] - c["a8"])
            dv = (1 - u) * (c["a8"] - c["a1"]) + u * (c["h8"] - c["h1"])
            step = np.linalg.solve(np.column_stack([du, dv]), e)
            u, v = u - step[0], v - step[1]
        f, r = u * 7.0, v * 7.0
        if len(self._anchors) > 4:          # refine on the corrected map (numeric Newton)
            for _ in range(30):
                e = np.array(self.grid_to_robot(f, r)) - t
                if np.linalg.norm(e) < 1e-3:
                    break
                h = 1e-3
                J = np.column_stack([(np.array(self.grid_to_robot(f + h, r)) - np.array(self.grid_to_robot(f, r))) / h,
                                     (np.array(self.grid_to_robot(f, r + h)) - np.array(self.grid_to_robot(f, r))) / h])
                d = np.linalg.solve(J, e)
                f, r = f - d[0], r - d[1]
        return float(f), float(r)

    def robot_to_square(self, x, y, tolerance=0.45):
        """Square under a robot X/Y, or None if off the board or too close to a line.
        tolerance is the max distance from the square centre, in squares (0.5 = edge)."""
        f, r = self.robot_to_grid(x, y)
        fi, ri = int(round(f)), int(round(r))
        if not (0 <= fi <= 7 and 0 <= ri <= 7):
            return None
        if abs(f - fi) > tolerance or abs(r - ri) > tolerance:
            return None
        return square_name(fi, ri)

    def all_squares(self):
        return {square_name(f, r): self.grid_to_robot(f, r) for r in range(8) for f in range(8)}

    def square_outline(self, f, r):
        """Robot X/Y of the 4 corners of square (f, r)."""
        return [self.grid_to_robot(f + df, r + dr)
                for df, dr in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))]

    # --- quality report ------------------------------------------------------------
    def report(self):
        c = {k: np.array([self.corners[k]["x"], self.corners[k]["y"]]) for k in CORNERS}
        sides = {"a1-h1": c["h1"] - c["a1"], "a8-h8": c["h8"] - c["a8"],
                 "a1-a8": c["a8"] - c["a1"], "h1-h8": c["h8"] - c["h1"]}
        lens = {k: float(np.linalg.norm(v)) for k, v in sides.items()}
        sq = np.mean(list(lens.values())) / 7.0
        v1, v2 = sides["a1-h1"], sides["a1-a8"]
        angle = float(np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))))
        lines = [f"square size {sq:.1f} mm (from corner spacing / 7)",
                 "edges: " + "  ".join(f"{k} {v:.0f}" for k, v in lens.items()) + " mm",
                 f"corner angle at a1: {angle:.1f} deg (90 = square board)"]
        spread = max(lens.values()) - min(lens.values())
        warn = []
        if spread > 1.0 * sq:
            warn.append(f"edges differ by {spread:.0f} mm (more than a whole square) - a corner "
                        "is probably on the wrong square; re-teach")
        elif spread > 5:
            lines.append(f"edges differ by {spread:.0f} mm: the far edge reads longer because the "
                         "robot's X/Y error grows with reach. Handled - squares are interpolated "
                         "between your corners. Check middle squares (e4, d5, a4, h5) are centred.")
        if abs(angle - 90) > 10:
            warn.append(f"corner angle {angle:.0f} deg is far from 90 - wrong corner order?")
        zs = [self.corners[k]["z"] for k in CORNERS]
        if max(zs) - min(zs) > 40:
            warn.append(f"corner heights differ by {max(zs) - min(zs):.0f} mm - check the tip was "
                        "the same height above each corner")
        elif max(zs) - min(zs) > 5:
            lines.append(f"corner heights differ by {max(zs) - min(zs):.0f} mm - used per square, "
                         "interpolated between the taught points")
        if self.extra:
            lines.append("extra points: " + "; ".join(self.extra_report()))
            big = [l for l in self.extra_report() if float(l.split("corrects ")[1].split(" mm")[0]) > 0.6 * sq]
            if big:
                warn.append("an extra point corrects by more than half a square - wrong square taught? "
                            + "; ".join(big))
        return lines, warn, sq

    # --- persistence ---------------------------------------------------------------
    def save(self, firmware, path=BOARD_FILE):
        lines, warn, sq = self.report()
        data = {"firmware": firmware, "corners": self.corners, "surface_z": self.surface_z,
                "tool_rot": self.tool_rot, "square_mm": sq,
                "heights": {k: round(self.square_z(k), 1) for k in self.all_squares()},
                "extra": self.extra,
                "tool_extension_mm": self.tool_ext,
                "squares": {k: [round(x, 2), round(y, 2)] for k, (x, y) in self.all_squares().items()}}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return lines, warn

    @classmethod
    def load(cls, path=BOARD_FILE):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        b = cls(data["corners"], data.get("extra"))
        b.tool_ext = float(data.get("tool_extension_mm", 0.0))
        b.firmware = data.get("firmware")
        return b


# --- fixing corner labels without re-teaching --------------------------------------------
RELABEL = {
    # new name: old name
    "mirror": {"a1": "h1", "h1": "a1", "h8": "a8", "a8": "h8"},   # a <-> h files swapped
    "rotate": {"a1": "h8", "h1": "a8", "h8": "a1", "a8": "h1"},   # board turned 180 deg
    "ranks":  {"a1": "a8", "h1": "h8", "h8": "h1", "a8": "a1"},   # 1 <-> 8 ranks swapped
}


def relabel(argv):
    """python main.py relabel mirror|rotate|ranks  - rename the 4 taught corners."""
    import shutil
    if len(argv) != 1 or argv[0] not in RELABEL:
        print("usage: python main.py relabel mirror|rotate|ranks\n"
              "  mirror  you taught the a-file corners where the h-file is (left/right swapped)\n"
              "  rotate  you taught as if sitting on the other side of the board\n"
              "  ranks   ranks 1 and 8 swapped")
        return
    board = ChessBoard.load()
    if board is None:
        raise SystemExit("chess_board.json not found")
    shutil.copy(BOARD_FILE, BOARD_FILE + ".bak")
    m = RELABEL[argv[0]]
    flip = {"mirror": lambda f, r: (7 - f, r), "rotate": lambda f, r: (7 - f, 7 - r),
            "ranks": lambda f, r: (f, 7 - r)}[argv[0]]
    extra = {square_name(*flip(*square_index(sq))): p for sq, p in board.extra.items()}
    new = ChessBoard({k: board.corners[m[k]] for k in CORNERS}, extra)
    new.tool_ext = board.tool_ext
    lines, warn = new.save(getattr(board, "firmware", None) or "6")
    print(f"Relabelled ({argv[0]}). Old file kept as chess_board.json.bak")
    for k in CORNERS:
        c = new.corners[k]
        print(f"  {k}: X {c['x']:.1f}  Y {c['y']:.1f}")
    for k in sorted(new.extra):
        print(f"  {k}: X {new.extra[k]['x']:.1f}  Y {new.extra[k]['y']:.1f}  (extra point)")
    for line in lines + ["WARNING: " + w for w in warn]:
        print(line)


# --- finger extension (sticks) and straight-down gripper ---------------------------------
def tool_cmd(argv):
    """python main.py tool          show the finger extension in use
       python main.py tool 45       the fingertips now reach 45 mm LOWER than when the board
                                    was taught (sticks added) - every height rises 45 mm
       python main.py tool 0        back to the original fingers
    Re-teaching the corners in jog (N, B x4) is done WITH the new fingers, so it resets this to 0."""
    import shutil
    board = ChessBoard.load()
    if board is None:
        raise SystemExit("chess_board.json not found")
    if not argv:
        print(f"Finger extension: {board.tool_ext:g} mm")
        print(tool_cmd.__doc__)
        return
    try:
        mm = float(argv[0])
    except ValueError:
        print(tool_cmd.__doc__)
        return
    if not 0 <= mm <= 150:
        raise SystemExit("extension 0-150 mm")
    shutil.copy(BOARD_FILE, BOARD_FILE + ".bak")
    old = board.tool_ext
    board.tool_ext = mm
    board.save(getattr(board, "firmware", None) or "6")
    print(f"Finger extension {old:g} -> {mm:g} mm: every board height is now {mm - old:+g} mm "
          f"(old file kept as chess_board.json.bak).")
    print("Check before playing: `python main.py board --firmware 6`, `h 10`, `e4` - the NEW tips "
          "should be about 10 mm above the board.")


def straight_down_rot(rv):
    """Same finger direction (turn about the vertical), but pointing exactly straight down."""
    import kinematics as K
    R = K.rotvec_to_R(rv)
    down = K.rotvec_to_R([0, 180, 0])
    a, b = R[:, 0][:2], down[:, 0][:2]
    yaw = np.arctan2(a[1], a[0]) - np.arctan2(b[1], b[0])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return [float(v) for v in K.R_to_rotvec(Rz @ down)]


def straightdown_cmd(argv):
    """python main.py straightdown          show how far the gripper is from straight down
       python main.py straightdown fix      make it exactly straight down (keeps the finger
                                            direction); every move uses this"""
    import shutil
    import kinematics as K
    board = ChessBoard.load()
    if board is None:
        raise SystemExit("chess_board.json not found")
    rv = board.corners["a1"]["rot"]
    tilt = K.tilt_from_down(rv)
    print(f"Gripper direction used for every move (saved at a1): {tilt:.2f} deg from straight down")
    for L in (100, 150, 200):
        print(f"   at {L} mm finger length that puts the tips {L * np.sin(np.radians(tilt)):.1f} mm sideways")
    if not argv or argv[0] != "fix":
        if tilt > 0.2:
            print("Run `python main.py straightdown fix` to make it exactly straight down.")
        return
    new = straight_down_rot(rv)
    shutil.copy(BOARD_FILE, BOARD_FILE + ".bak")
    board.corners["a1"]["rot"] = [float(v) for v in new]
    nb = ChessBoard(board.corners, board.extra)
    nb.tool_ext = board.tool_ext
    nb.save(getattr(board, "firmware", None) or "6")
    print(f"Now {K.tilt_from_down(new):.2f} deg from straight down, same finger direction "
          "(old file kept as chess_board.json.bak).")
    if tilt > 0.5:
        print("The tilt was big enough to move the tips a little - check a1, e4 and h8 with "
              "`board` and nudge + save any that are off.")
