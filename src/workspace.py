"""Workspace frame helpers.

When all ArUco markers have been taught (jogged over and saved), their ROBOT X/Y are
used directly as the homography plane points: camera pixels map straight to robot mm.
No tape-measure values are needed; MARKER_POSITIONS_MM then only lists which IDs exist.
Without a full teach, the measured MARKER_POSITIONS_MM + affine fit is used (old way).
"""
import cv2
import numpy as np


def taught_plane(calib, ids):
    """{id: (robot_x, robot_y)} if every marker id has been taught, else None."""
    if not calib:
        return None
    taught = calib.get("taught", {})
    plane = {}
    for i in ids:
        p = taught.get(str(i))
        if p is None:
            return None
        plane[int(i)] = (float(p["x"]), float(p["y"]))
    return plane


def make_inside(points, margin_mm):
    """Return inside(x, y) -> bool: point lies inside the marker polygon, at least
    margin_mm from its edges. Works for any convex quad in any frame."""
    pts = np.array(list(points), dtype=np.float32)
    centre = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
    poly = pts[order].reshape(-1, 1, 2)

    def inside(x, y):
        return cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= margin_mm

    return inside
