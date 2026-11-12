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
    def __init__(self, corners):
        """corners: {"a1": pose, "h1": pose, "h8": pose, "a8": pose}; pose has x, y, z, rot."""
        self.corners = corners
        grid = np.float32([GRID[c] for c in CORNERS])
        robot = np.float32([[corners[c]["x"], corners[c]["y"]] for c in CORNERS])
        self.H = cv2.getPerspectiveTransform(grid, robot)          # grid -> robot
        self.Hinv = np.linalg.inv(self.H)                          # robot -> grid
        self.surface_z = float(np.mean([corners[c]["z"] for c in CORNERS]))
        self.tool_rot = corners["a1"]["rot"]

    # --- conversions -------------------------------------------------------------
    def grid_to_robot(self, f, r):
        p = cv2.perspectiveTransform(np.float32([[[f, r]]]), self.H)[0, 0]
        return float(p[0]), float(p[1])

    def square_to_robot(self, name):
        return self.grid_to_robot(*square_index(name))

    def robot_to_grid(self, x, y):
        g = cv2.perspectiveTransform(np.float32([[[x, y]]]), self.Hinv)[0, 0]
        return float(g[0]), float(g[1])

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
        if spread > 0.25 * sq:
            warn.append(f"edges differ by {spread:.0f} mm (> a quarter square) - re-teach the corners")
        if abs(angle - 90) > 5:
            warn.append(f"corner angle {angle:.0f} deg is far from 90 - wrong corner order?")
        zs = [self.corners[k]["z"] for k in CORNERS]
        if max(zs) - min(zs) > 10:
            warn.append(f"corner heights differ by {max(zs) - min(zs):.0f} mm - board not level, "
                        "or the tip was not at the same height above each corner")
        return lines, warn, sq

    # --- persistence ---------------------------------------------------------------
    def save(self, firmware, path=BOARD_FILE):
        lines, warn, sq = self.report()
        data = {"firmware": firmware, "corners": self.corners, "surface_z": self.surface_z,
                "tool_rot": self.tool_rot, "square_mm": sq,
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
        b = cls(data["corners"])
        b.firmware = data.get("firmware")
        return b
