"""Chess piece perception: per-square crops from the live camera + a trained classifier.

Pipeline:
  ArUco cards (taught)  -> pixel <-> robot mm
  chess_board.json      -> each square's outline in robot mm -> in pixels
  per square            -> crop the square plus the space ABOVE it in the image (a piece
                           stands up, so its body appears above its base in an angled view)
  crop                  -> features (HOG shape + colour)  -> classifier -> label + confidence

Labels use python-chess symbols: "." empty, P N B R Q K white, p n b r q k black.
"""
import json
import os

import cv2
import numpy as np

import chess_board as cb
import vision_config as cfg
import workspace as ws

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")
MODEL_FILE = os.path.join(HERE, "piece_model.joblib")

LABELS = [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
NAMES = {".": "empty", "P": "white_pawn", "N": "white_knight", "B": "white_bishop",
         "R": "white_rook", "Q": "white_queen", "K": "white_king", "p": "black_pawn",
         "n": "black_knight", "b": "black_bishop", "r": "black_rook", "q": "black_queen",
         "k": "black_king"}
DIR_TO_LABEL = {v: k for k, v in NAMES.items()}

CROP_W, CROP_H = 64, 112   # normalised crop size (portrait: pieces are tall)
EXTEND_UP = 1.6            # crop extends this many square-heights above the square
CAMERA_SIZE = (1280, 720)  # requested capture resolution


# --- camera + geometry --------------------------------------------------------------
def open_camera(index=None, size=CAMERA_SIZE):
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX if index is None else index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    return cap


class BoardCamera:
    """Maps the taught board into the current camera frame."""

    def __init__(self):
        from ArUcoCalibration import ArUcoCalibration
        board = cb.ChessBoard.load()
        if board is None:
            raise SystemExit("chess_board.json not found - teach the board corners first "
                             "(python main.py jog, key B).")
        calib = None
        if os.path.exists(cfg.CALIB_FILE):
            with open(cfg.CALIB_FILE) as f:
                calib = json.load(f)
        plane = ws.taught_plane(calib, cfg.MARKER_POSITIONS_MM)
        if plane is None:
            raise SystemExit("All 4 ArUco cards must be taught first (jog window: ID keys, P, ENTER).")
        self.board = board
        self.aruco = ArUcoCalibration(plane)
        self.Hinv = None

    def update(self, frame):
        """Locate the cards in this frame. False if not all 4 are visible."""
        if not self.aruco.update(frame):
            self.Hinv = None
            return False
        self.Hinv = np.linalg.inv(self.aruco.H)
        return True

    def to_px(self, xy):
        p = cv2.perspectiveTransform(np.float32([[xy]]), self.Hinv)[0, 0]
        return float(p[0]), float(p[1])

    def square_quad(self, f, r):
        return np.array([self.to_px(c) for c in self.board.square_outline(f, r)], np.float32)

    def square_box(self, f, r, frame_shape):
        """Crop box for a square: its outline, widened a little and extended upward."""
        q = self.square_quad(f, r)
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        w, h = x1 - x0, y1 - y0
        x0, x1 = x0 - 0.15 * w, x1 + 0.15 * w
        y0 = y0 - EXTEND_UP * h
        H, W = frame_shape[:2]
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(W, x1)), int(min(H, y1 + 0.05 * h))
        return x0, y0, x1, y1

    def crops(self, frame):
        """{square: normalised crop} for all 64 squares (call update() first)."""
        out = {}
        for r in range(8):
            for f in range(8):
                x0, y0, x1, y1 = self.square_box(f, r, frame.shape)
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                out[cb.square_name(f, r)] = cv2.resize(frame[y0:y1, x0:x1], (CROP_W, CROP_H),
                                                       interpolation=cv2.INTER_AREA)
        return out

    def draw_grid(self, frame, labels=None, conf=None, min_conf=0.6):
        for r in range(8):
            for f in range(8):
                sq = cb.square_name(f, r)
                q = self.square_quad(f, r).astype(np.int32)
                cv2.polylines(frame, [q], True, (0, 255, 255), 1)
                cx, cy = q.mean(axis=0).astype(int)
                text, colour = sq, (0, 255, 255)
                if labels is not None and sq in labels:
                    text = labels[sq] if labels[sq] != "." else ""
                    ok = conf is None or conf.get(sq, 1.0) >= min_conf
                    colour = (0, 255, 0) if ok else (0, 0, 255)
                    if not ok and not text:
                        text = "?"
                for c, t in (((0, 0, 0), 3), (colour, 1)):
                    cv2.putText(frame, text, (cx - 8, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, t)
        return frame


# --- features ---------------------------------------------------------------------
_HOG = cv2.HOGDescriptor((CROP_W, CROP_H), (16, 16), (8, 8), (8, 8), 9)
_HOG_LOW = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


def features(crop):
    """Shape (HOG on the whole crop + a second HOG on its lower part, where THIS square's
    piece stands, less disturbed by pieces on the squares behind) + colour histograms."""
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    grey = cv2.equalizeHist(grey)
    hog = _HOG.compute(grey).ravel()
    hog_low = _HOG_LOW.compute(cv2.resize(grey[CROP_H // 3:], (64, 64))).ravel() * 1.5
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    body = hsv[CROP_H // 3:, CROP_W // 5: CROP_W - CROP_W // 5]
    v_hist = cv2.calcHist([body], [2], None, [16], [0, 256]).ravel()
    s_hist = cv2.calcHist([body], [1], None, [8], [0, 256]).ravel()
    h_hist = cv2.calcHist([body], [0], None, [12], [0, 180]).ravel()
    colour = np.concatenate([v_hist, s_hist, h_hist])
    colour /= colour.sum() + 1e-6
    stats = np.array([body[..., 2].mean() / 255, body[..., 2].std() / 255,
                      body[..., 1].mean() / 255], np.float32)
    return np.concatenate([hog, hog_low, colour * 4.0, stats * 4.0]).astype(np.float32)


# --- classifier ---------------------------------------------------------------------
class PieceClassifier:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path=None):
        import joblib
        path = path or MODEL_FILE
        if not os.path.exists(path):
            raise SystemExit("piece_model.joblib not found - capture data and train first:\n"
                             "  python main.py capture   then   python main.py train")
        return cls(joblib.load(path))

    def predict(self, crops):
        """crops {square: img} -> (labels {sq: symbol}, conf {sq: probability})."""
        squares = list(crops)
        X = np.stack([features(crops[s]) for s in squares])
        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        best = proba.argmax(axis=1)
        labels = {s: classes[b] for s, b in zip(squares, best)}
        conf = {s: float(proba[i, b]) for i, (s, b) in enumerate(zip(squares, best))}
        self.last_proba = {s: dict(zip(classes, proba[i])) for i, s in enumerate(squares)}
        return labels, conf


def board_to_labels(board):
    """python-chess Board -> {square: symbol} for all 64 squares."""
    import chess
    out = {}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        out[chess.square_name(sq)] = p.symbol() if p else "."
    return out
