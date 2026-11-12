"""Multi-target detection and tracking for same-colour objects.

The team's ColorDetector follows ONE blob (largest first, then nearest to the last
position), so with several same-colour objects the choice is arbitrary. This module
finds EVERY blob of the selected colour, converts each to board mm, and tracks them
across frames so each object keeps a stable identity and gets its own stability check.

Uses the team's ColorDetector.mask_colors / clean_mask, so colour selection is unchanged.
"""
from collections import deque

import cv2
import numpy as np

MATCH_MM = 20.0        # a detection within this distance of a track continues that track
MAX_MISSED = 5         # frames a track may go unseen before it is dropped
MERGE_RATIO = 1.8      # a blob this many times the median area may be two touching objects


def find_blobs(detector, frame):
    """All blobs of the selected colour: [{"pixel": (u, v), "area": a, "bbox": (x, y, w, h)}]."""
    if detector.color_range is None:
        return []
    lower, upper = detector.color_range
    mask = detector.clean_mask(detector.mask_colors(lower, upper, frame))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < detector.min_contour_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        # area centroid (more accurate than the bounding-box centre for irregular shapes)
        blobs.append({"pixel": (m["m10"] / m["m00"], m["m01"] / m["m00"]),
                      "area": area, "bbox": cv2.boundingRect(c)})
    return blobs


class Track:
    _next_id = 1

    def __init__(self, blob, board_xy, samples):
        self.id = Track._next_id
        Track._next_id += 1
        self.history = deque(maxlen=samples)
        self.missed = 0
        self.update(blob, board_xy)

    def update(self, blob, board_xy):
        self.history.append(board_xy)
        self.pixel = blob["pixel"]
        self.area = blob["area"]
        self.bbox = blob["bbox"]
        self.missed = 0

    @property
    def board(self):
        return tuple(np.median(np.array(self.history), axis=0))


class MultiTracker:
    def __init__(self, samples, spread_mm, limits):
        self.samples = samples
        self.spread_mm = spread_mm
        self.limits = limits  # inside(x, y) -> bool, or (x0, x1, y0, y1)
        self.tracks = []

    def clear(self):
        self.tracks = []

    def update(self, blobs, to_board):
        """blobs from find_blobs; to_board(pixel) -> (x, y) board mm."""
        dets = [(b, to_board(b["pixel"])) for b in blobs]
        # greedy nearest matching, closest pairs first
        pairs = sorted(
            ((np.hypot(t.board[0] - xy[0], t.board[1] - xy[1]), ti, di)
             for ti, t in enumerate(self.tracks) for di, (_, xy) in enumerate(dets)),
            key=lambda p: p[0])
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if dist > MATCH_MM or ti in used_t or di in used_d:
                continue
            self.tracks[ti].update(*dets[di])
            used_t.add(ti)
            used_d.add(di)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED]
        for di, (b, xy) in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(b, xy, self.samples))

    def status(self, t):
        """(ok, reason) for one track."""
        if t.missed:
            return False, "not seen"
        if len(t.history) < self.samples:
            return False, "settling"
        pts = np.array(t.history)
        spread = float(np.max(np.linalg.norm(pts - np.median(pts, axis=0), axis=1)))
        if spread > self.spread_mm:
            return False, "moving"
        bx, by = t.board
        if callable(self.limits):
            on_board = self.limits(bx, by)
        else:
            x0, x1, y0, y1 = self.limits
            on_board = x0 <= bx <= x1 and y0 <= by <= y1
        if not on_board:
            return False, "off board"
        others = [o.area for o in self.tracks if o is not t and not o.missed]
        if others and t.area > MERGE_RATIO * float(np.median(others)):
            return False, "merged? separate objects"
        return True, "ready"

    def numbered(self):
        """Visible tracks with display numbers 1..N, ordered along board X then Y."""
        vis = [t for t in self.tracks if not t.missed]
        vis.sort(key=lambda t: (round(t.board[0] / 20), t.board[1]))
        return list(enumerate(vis, start=1))

    def ready(self):
        return [(n, t) for n, t in self.numbered() if self.status(t)[0]]

    def by_id(self, track_id):
        for t in self.tracks:
            if t.id == track_id and not t.missed:
                return t
        return None

    def nearest_to_pixel(self, u, v, max_px=40):
        best = None
        for n, t in self.numbered():
            d = np.hypot(t.pixel[0] - u, t.pixel[1] - v)
            if d <= max_px and (best is None or d < best[0]):
                best = (d, t)
        return None if best is None else best[1]


def draw(frame, tracker, selected_id):
    for n, t in tracker.numbered():
        ok, reason = tracker.status(t)
        colour = (0, 200, 0) if ok else (0, 0, 255)
        box = (0, 200, 0) if ok else (255, 255, 255)  # white box shows on any target colour
        x, y, w, h = t.bbox
        thick = 3 if t.id == selected_id else 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), box, thick)
        u, v = int(t.pixel[0]), int(t.pixel[1])
        cv2.circle(frame, (u, v), 4, colour, -1)
        label = f"#{n}" + (" SEL" if t.id == selected_id else "") + ("" if ok else f" {reason}")
        cv2.putText(frame, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 2)
    return frame
