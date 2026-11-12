"""Piece classifier tools: capture labelled data, train, recognise live.

    python main.py capture      set up known positions, press SPACE: all 64 squares are
                                saved with their correct labels automatically
    python main.py train        train the classifier, print accuracy + confusion matrix
    python main.py recognize    live view: every square labelled with piece + confidence

CAPTURE keys (video window focused):
    S  standard starting position (32 pieces, labels from the rules)
    N  next random placement: all 32 pieces scattered (the layout is printed in the
       terminal - set the pieces exactly like that, then capture)
    E  empty board          F  type a FEN position in the terminal
    SPACE  capture 1 frame  B  burst: 5 frames (move the lights/your hand between bursts)
    Q  quit
Aim for: start position x 10 frames, then 8-10 random placements x 5 frames, in your
real demo lighting. Each frame gives 64 labelled crops.

RECOGNIZE keys:  SPACE print the board   C compare with the starting position   Q quit
"""
import argparse
import glob
import os
import random
import time
from collections import Counter, deque

import cv2
import numpy as np

import perception as pc


def _print_position(board, title):
    import chess
    print(f"\n=== {title} ===  (as seen from White's side: rank 8 at the top)")
    for r in range(7, -1, -1):
        row = []
        for f in range(8):
            p = board.piece_at(chess.square(f, r))
            row.append(p.symbol() if p else ".")
        print(f" {r + 1}  " + " ".join(row))
    print("    a b c d e f g h      (UPPER = white, lower = black)")
    print("    P pawn  N knight  B bishop  R rook  Q queen  K king")


def random_placement(seed):
    """All 32 pieces on random squares (not a legal game - it doesn't need to be)."""
    import chess
    rng = random.Random(seed)
    pieces = list("RNBQKBNR" + "P" * 8 + "rnbqkbnr" + "p" * 8)
    squares = rng.sample(range(64), len(pieces))
    b = chess.Board(None)
    for sym, sq in zip(pieces, squares):
        b.set_piece_at(sq, chess.Piece.from_symbol(sym))
    return b


def capture(argv=None):
    import chess
    ap = argparse.ArgumentParser(prog="main.py capture")
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args(argv)
    bc = pc.BoardCamera()
    cap = pc.open_camera(args.camera)
    position, title, seed = chess.Board(), "starting position", int(time.time())
    _print_position(position, title)
    counts = Counter(os.path.basename(os.path.dirname(p))   # crops already saved, per class
                     for p in glob.glob(os.path.join(pc.DATASET_DIR, "*", "*.png")))
    burst_left, next_shot = 0, 0.0
    window = "Capture (S start  N random  E empty  F fen  SPACE shot  B burst  Q quit)"
    print(__doc__.split("CAPTURE keys")[1].split("RECOGNIZE")[0])
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame unavailable")
            break
        visible = bc.update(frame)
        shown = frame.copy()
        if visible:
            bc.draw_grid(shown, pc.board_to_labels(position))
        status = (f"{title} | saved frames this session: {counts.get('_frames', 0)}"
                  if visible else f"cards visible: {sorted(bc.aruco.visible_ids)} - need all 4")
        cv2.putText(shown, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if visible else (0, 0, 255), 2)
        cv2.imshow(window, shown)
        key = cv2.waitKey(1) & 0xFF

        if burst_left and time.time() >= next_shot and visible:
            _save(bc, frame, position, counts)
            burst_left -= 1
            next_shot = time.time() + 0.4
            if not burst_left:
                _report(counts)
        if key == ord("q"):
            break
        elif key == ord("s"):
            position, title = chess.Board(), "starting position"
            _print_position(position, title)
        elif key == ord("e"):
            position, title = chess.Board(None), "empty board"
            _print_position(position, title)
        elif key == ord("n"):
            seed += 1
            position, title = random_placement(seed), f"random placement #{seed % 1000}"
            _print_position(position, title)
        elif key == ord("f"):
            fen = input("FEN (piece placement part is enough): ").strip()
            try:
                position, title = chess.Board(fen.split()[0] + " w - - 0 1"), "custom position"
                _print_position(position, title)
            except ValueError as e:
                print("Bad FEN:", e)
        elif key == ord(" "):
            if not visible:
                print("Not saved: all 4 cards must be visible")
            else:
                _save(bc, frame, position, counts)
                _report(counts)
        elif key == ord("b"):
            burst_left, next_shot = 5, 0.0
    cap.release()
    cv2.destroyAllWindows()


def _save(bc, frame, position, counts):
    labels = pc.board_to_labels(position)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    for sq, img in bc.crops(frame).items():
        name = pc.NAMES[labels[sq]]
        d = os.path.join(pc.DATASET_DIR, name)
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, f"{stamp}_{sq}.png"), img)
        counts[name] += 1
    counts["_frames"] += 1


def _report(counts):
    parts = [f"{k} {v}" for k, v in sorted(counts.items()) if not k.startswith("_")]
    print(f"saved frame {counts['_frames']} | " + ", ".join(parts))


def load_dataset(augment=False):
    """Features for every saved crop. augment=True adds a mirrored copy of each crop
    (chess pieces look the same mirrored; this doubles the training data)."""
    X, y, groups = [], [], []
    for d in sorted(glob.glob(os.path.join(pc.DATASET_DIR, "*"))):
        label = pc.DIR_TO_LABEL.get(os.path.basename(d))
        if label is None:
            continue
        for path in sorted(glob.glob(os.path.join(d, "*.png"))):
            img = cv2.imread(path)
            if img is None:
                continue
            if img.shape[:2] != (pc.CROP_H, pc.CROP_W):
                img = cv2.resize(img, (pc.CROP_W, pc.CROP_H))
            group = os.path.basename(path).split("_")[0]  # one group per captured frame
            for im in ([img, cv2.flip(img, 1)] if augment else [img]):
                X.append(pc.features(im))
                y.append(label)
                groups.append(group)
    return np.array(X), np.array(y), np.array(groups)


def build_model():
    # Logistic regression: as accurate as an SVM on these features in testing, ~10x
    # faster to train, and gives the per-square probabilities used as confidence.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))


def train(argv=None):
    import joblib
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit

    X, y, groups = load_dataset(augment=True)
    if len(X) == 0:
        raise SystemExit("No data in dataset/ - run: python main.py capture")
    present = [l for l in pc.LABELS if l in set(y)]
    print(f"{len(X) // 2} crops (+ mirrored copies) from {len(set(groups))} frames. Per class: " +
          ", ".join(f"{pc.NAMES[l]} {int((y == l).sum()) // 2}" for l in present))
    missing = [pc.NAMES[l] for l in pc.LABELS if l not in set(y)]
    if missing:
        print("WARNING: no examples of", ", ".join(missing))

    if len(set(groups)) >= 4:
        split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
        tr, te = next(split.split(X, y, groups))   # whole frames held out (no leakage)
        model = build_model().fit(X[tr], y[tr])
        pred = model.predict(X[te])
        acc = float((pred == y[te]).mean())
        occ = y[te] != "."
        pacc = float((pred[occ] == y[te][occ]).mean()) if occ.any() else float("nan")
        print(f"\nHeld-out accuracy (frames never seen in training): {acc * 100:.1f}% of squares, "
              f"{pacc * 100:.1f}% of pieces (type + colour)")
        cm = confusion_matrix(y[te], pred, labels=present)
        print("\nConfusion matrix (rows = truth, columns = prediction):")
        print("      " + " ".join(f"{l:>4}" for l in present))
        for l, row in zip(present, cm):
            print(f"  {l:>3} " + " ".join(f"{v:>4}" for v in row))
        print("\nPer class recall:")
        for l, row in zip(present, cm):
            n = row.sum()
            if n:
                print(f"  {pc.NAMES[l]:<14} {row[present.index(l)] / n * 100:5.1f}%  ({n} crops)")
    else:
        print("Fewer than 4 captured frames - skipping the held-out test (capture more).")

    model = build_model().fit(X, y)
    joblib.dump(model, pc.MODEL_FILE)
    print(f"\nTrained on all data -> {os.path.basename(pc.MODEL_FILE)}")


def recognize(argv=None):
    import chess
    ap = argparse.ArgumentParser(prog="main.py recognize")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--min-conf", type=float, default=0.6)
    args = ap.parse_args(argv)
    bc = pc.BoardCamera()
    clf = pc.PieceClassifier.load()
    cap = pc.open_camera(args.camera)
    history = deque(maxlen=5)
    window = "Recognize (SPACE print board  C compare with start  Q quit)"
    labels = conf = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame unavailable")
            break
        shown = frame.copy()
        if bc.update(frame):
            clf.predict(bc.crops(frame))
            history.append(clf.last_proba)
            labels, conf = _smooth(history)
            bc.draw_grid(shown, labels, conf, args.min_conf)
            unsure = sum(1 for v in conf.values() if v < args.min_conf)
            msg, col = (f"{unsure} uncertain square(s) - red ?" if unsure else "all squares confident",
                        (0, 0, 255) if unsure else (0, 255, 0))
        else:
            history.clear()
            labels = None
            msg, col = f"cards visible: {sorted(bc.aruco.visible_ids)} - need all 4", (0, 0, 255)
        cv2.putText(shown, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.imshow(window, shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if labels is None:
            continue
        if key == ord(" "):
            _print_labels(labels, conf, args.min_conf)
        elif key == ord("c"):
            want = pc.board_to_labels(chess.Board())
            wrong = [(s, want[s], labels[s], conf[s]) for s in sorted(want) if want[s] != labels[s]]
            print(f"\nCompared with the starting position: {64 - len(wrong)}/64 squares match")
            for s, w, g, c in wrong:
                print(f"  {s}: expected {pc.NAMES[w]}, seen {pc.NAMES[g]} ({c * 100:.0f}%)")
    cap.release()
    cv2.destroyAllWindows()


def _smooth(history):
    squares = history[-1].keys()
    labels, conf = {}, {}
    for s in squares:
        acc = Counter()
        for h in history:
            for k, v in h.get(s, {}).items():
                acc[k] += v / len(history)
        best, p = acc.most_common(1)[0]
        labels[s], conf[s] = best, p
    return labels, conf


def _print_labels(labels, conf, min_conf):
    print("\nRecognised board (? = below confidence threshold):")
    for r in range(8, 0, -1):
        row = []
        for f in "abcdefgh":
            s = f"{f}{r}"
            row.append(labels[s] if conf[s] >= min_conf else "?")
        print(f" {r}  " + " ".join(row))
    print("    a b c d e f g h")
    placement = "/".join(
        "".join(labels[f"{f}{r}"] for f in "abcdefgh") for r in range(8, 0, -1))
    import re
    fen = re.sub(r"\.+", lambda m: str(len(m.group())), placement)
    print("FEN placement:", fen)
