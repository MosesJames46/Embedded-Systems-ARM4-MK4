"""Milestone 4: camera-guided AR4 positioning above a coloured target.

Pipeline: webcam -> colour detection (ColorDetector) -> pixel to board mm
(the 4 ArUco markers in vision_config.py) -> board mm to robot mm (robot_calib.json from teach_robot.py)
-> safe approach, hover, retreat, park.

Run:
    python vision_move.py --dry     # vision only, no serial connection, no motion
    python vision_move.py           # live; the arm moves only when you press G

Keys in the video window:
    G      move to hover above the current target (only when the status is READY)
    C      choose a new target (re-select its colour) - only while the arm is idle
    M      switch single / multi-target mode (multi: every same-colour object is numbered)
    1-9    multi mode: select target by number (or click it in the video)
    A      multi mode: visit every READY target in turn, then park
    + / -  raise / lower the hover clearance by 5 mm (saved for next time)
    ] / [  raise / lower the target height by 5 mm (saved for next time)
    SPACE  stop the arm immediately (firmware 8.1 only)
    P      go to the park pose
    Q      quit

Hardware emergency stop: the red button on the base lid (Teensy pin 39) or the
24 V rocker switch. The camera view stays live while the arm moves.
"""
import argparse
import threading
import csv
import json
import os
import time
from collections import deque

import cv2
import numpy as np

from ColorDetector import ColorDetector
from Camera import Camera
from ArUcoCalibration import ArUcoCalibration
import vision_config as cfg
from robot_backend import connect, RobotFault
import multi_target as mt
import workspace as ws

GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 255), (0, 200, 255)


LIVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_settings.json")
CLEARANCE_RANGE = (10, 120)   # mm above the target top
TARGET_RANGE = (0, 120)       # mm, target height above the mat
STEP_MM = 5


def load_live():
    """Heights adjustable from the video window; saved so they persist between runs."""
    live = {"target_height_mm": float(cfg.TARGET_HEIGHT_MM),
            "hover_clearance_mm": float(cfg.HOVER_CLEARANCE_MM)}
    try:
        with open(LIVE_FILE) as f:
            saved = json.load(f)
        for k in live:
            if k in saved:
                live[k] = float(saved[k])
    except (OSError, ValueError):
        pass
    return live


def save_live(live):
    try:
        with open(LIVE_FILE, "w") as f:
            json.dump(live, f, indent=2)
    except OSError as e:
        print("Could not save live settings:", e)


def adjust_live(live, key):
    """Apply a height key. Returns True if something changed."""
    lo_c, hi_c = CLEARANCE_RANGE
    lo_t, hi_t = TARGET_RANGE
    c, h = live["hover_clearance_mm"], live["target_height_mm"]
    if key in (ord("+"), ord("=")):
        c += STEP_MM
    elif key in (ord("-"), ord("_")):
        c -= STEP_MM
    elif key == ord("]"):
        h += STEP_MM
    elif key == ord("["):
        h -= STEP_MM
    else:
        return False
    c = min(max(c, lo_c), hi_c)
    h = min(max(h, lo_t), hi_t)
    # hover must stay well below the travel height
    if h + c > cfg.SAFE_Z_MM - 20:
        print(f"Refused: hover {h + c:.0f} mm would be too close to the travel height "
              f"({cfg.SAFE_Z_MM} mm)")
        return False
    changed = (c, h) != (live["hover_clearance_mm"], live["target_height_mm"])
    live["hover_clearance_mm"], live["target_height_mm"] = c, h
    if changed:
        save_live(live)
        print(f"Hover now {h + c:.0f} mm above the mat (target {h:.0f} + clearance {c:.0f})")
    return changed


def load_calib():
    if not os.path.exists(cfg.CALIB_FILE):
        return None
    with open(cfg.CALIB_FILE) as f:
        c = json.load(f)
    plane = ws.taught_plane(c, cfg.MARKER_POSITIONS_MM)
    if plane is not None:
        # All markers taught: camera maps straight to robot mm, so board == robot frame.
        c["plane"] = plane
        c["M"] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    else:
        c["plane"] = None
        c["M"] = np.array(c["affine_board_to_robot"], dtype=np.float64)
    return c


def board_to_robot(calib, x, y):
    rx, ry = calib["M"] @ np.array([x, y, 1.0])
    return float(rx), float(ry)


def check_target(history, calibrated, detector_stable, inside):
    """Return (ok, reason, (x, y)) using the median of recent readings."""
    if not calibrated:
        return False, "markers not all visible", None
    if not detector_stable:
        return False, "no stable target", None
    if len(history) < cfg.STABLE_SAMPLES:
        return False, "collecting samples", None
    pts = np.array(history)
    med = np.median(pts, axis=0)
    spread = float(np.max(np.linalg.norm(pts - med, axis=1)))
    if spread > cfg.STABLE_SPREAD_MM:
        return False, f"target jittering ({spread:.1f} mm)", tuple(med)
    if not inside(*med):
        return False, "target outside board limits", tuple(med)
    return True, "READY - press G", tuple(med)


def log_row(row):
    new = not os.path.exists(cfg.LOG_FILE)
    with open(cfg.LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def go_park(arm, calib):
    park = calib["taught"]["park"]
    safe = calib["board_z"] + cfg.SAFE_Z_MM
    p = arm.pose()
    if p["z"] < safe - 1:
        arm.move_linear(p["x"], p["y"], safe, p["rot"], cfg.SPEED_PCT)
    # Joint move: avoids the wrist-flip zone that straight-line moves cross on old firmware
    arm.move_joint(park["x"], park["y"], park["z"], park["rot"], cfg.SPEED_PCT)


def hover_over(arm, calib, board_xy, pixel, status, live, park=True):
    rot = calib["tool_rot"]
    bz = calib["board_z"]
    safe = bz + cfg.SAFE_Z_MM
    hover = bz + live["target_height_mm"] + live["hover_clearance_mm"]
    tx, ty = board_to_robot(calib, *board_xy)
    row = {"time": time.strftime("%H:%M:%S"), "pixel_u": round(pixel[0], 1),
           "pixel_v": round(pixel[1], 1), "board_x": round(board_xy[0], 1),
           "board_y": round(board_xy[1], 1), "cmd_x": round(tx, 1), "cmd_y": round(ty, 1),
           "cmd_z": round(hover, 1), "rep_x": "", "rep_y": "", "rep_z": "",
           "result": "", "measured_error_mm": ""}
    print(f"\nTarget board ({board_xy[0]:.1f}, {board_xy[1]:.1f}) mm -> "
          f"robot ({tx:.1f}, {ty:.1f}), hover Z {hover:.1f}")
    try:
        if hover < bz + 20 or hover >= safe:
            raise ValueError(f"Bad heights: hover {hover:.1f}, safe {safe:.1f}, board {bz:.1f}")
        p = arm.pose()
        if p["z"] < safe - 1:
            status["text"] = "rising to safe height"
            arm.move_linear(p["x"], p["y"], safe, p["rot"], cfg.SPEED_PCT)
        status["text"] = "approach at safe height"
        arm.move_linear(tx, ty, safe, rot, cfg.SPEED_PCT)
        status["text"] = "descending to hover"
        r = arm.move_linear(tx, ty, hover, rot, cfg.SPEED_PCT)
        row.update(rep_x=round(r["x"], 1), rep_y=round(r["y"], 1), rep_z=round(r["z"], 1))
        status["text"] = "hovering above target"
        print(f"  hovering; reported X {r['x']:.1f} Y {r['y']:.1f} Z {r['z']:.1f}")
        time.sleep(cfg.HOLD_SECONDS)
        status["text"] = "retreating"
        arm.move_linear(tx, ty, safe, rot, cfg.SPEED_PCT)
        if park:
            status["text"] = "parking"
            go_park(arm, calib)
        row["result"] = "ok"
        status["text"] = "done"
    except (RobotFault, ValueError) as e:
        row["result"] = f"aborted: {e}"
        status["text"] = f"ABORTED: {e}"
        print("  ABORTED:", e)
        if "Stopped by operator" not in str(e):
            try:  # best-effort straight-up retreat after a refused/faulted move
                p = arm.pose()
                if p["z"] < safe - 1:
                    arm.move_linear(p["x"], p["y"], safe, p["rot"], cfg.SPEED_PCT)
            except Exception as e2:
                print("  Retreat failed - use the E-stop if needed:", e2)
    log_row(row)
    return row["result"] == "ok"


def visit_all(arm, calib, targets, status, live):
    """Hover over each target in turn (positions captured before the arm moved), then park."""
    total = len(targets)
    for i, (num, board_xy, pixel) in enumerate(targets, start=1):
        print(f"\n=== Target {i}/{total} (#{num}) ===")
        if not hover_over(arm, calib, board_xy, pixel, status, live, park=False):
            status["text"] = f"sequence stopped at target {i}/{total}: {status['text']}"
            return
    status["text"] = "parking"
    try:
        go_park(arm, calib)
        status["text"] = f"visited {total} targets"
    except Exception as e:
        status["text"] = f"park failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="vision only, no robot connection")
    ap.add_argument("--port", default=None)
    ap.add_argument("--camera", type=int, default=cfg.CAMERA_INDEX)
    ap.add_argument("--firmware", choices=["8", "6"], default=None)
    args = ap.parse_args()

    calib = load_calib()
    if calib is None and not args.dry:
        raise SystemExit("robot_calib.json not found - run teach_robot.py first "
                         "(or use --dry)")
    if calib is not None and "park" not in calib.get("taught", {}):
        raise SystemExit("robot_calib.json has no park pose - re-run teach_robot.py")

    arm = None
    if not args.dry:
        fw = args.firmware or calib.get("firmware", cfg.FIRMWARE)
        arm = connect(fw, args.port, cfg.LOOP_MODE)
        p = arm.pose()
        print(f"Connected (firmware {fw}). Pose X {p['x']:.1f} Y {p['y']:.1f} Z {p['z']:.1f}")
        if not arm.supports_stop:
            print("NOTE: firmware 6.x has no software stop - SPACE does nothing; use the E-stop.")

    # ColorDetector stores its ColorPicker but never uses it; passing None avoids the
    # frozen pygame window ("not responding" popup).
    detector = ColorDetector(None, Camera(args.camera))
    plane = calib["plane"] if calib is not None else None
    if plane is not None:
        print("Using the taught marker positions: camera pixels map straight to robot mm "
              "(no tape measurements needed).")
    else:
        plane = cfg.MARKER_POSITIONS_MM
        print("Using measured MARKER_POSITIONS_MM from vision_config.py"
              + (" with the board-to-robot fit." if calib is not None else "."))
    aruco = ArUcoCalibration(plane)
    inside = ws.make_inside(plane.values(), cfg.BOARD_MARGIN_MM)
    robot_frame = calib is not None and calib["plane"] is not None
    print("Drag a box over the target colour, then press ENTER.")
    detector.color_range = detector.calibrate_color()
    cv2.destroyAllWindows()
    if detector.color_range is None:
        raise SystemExit("Colour calibration cancelled")

    history = deque(maxlen=cfg.STABLE_SAMPLES)
    live = load_live()
    print(f"Hover height: target {live['target_height_mm']:.0f} + clearance "
          f"{live['hover_clearance_mm']:.0f} mm.  Keys: + / - clearance, ] / [ target height")
    worker = None
    status = {"text": ""}

    def busy():
        return worker is not None and worker.is_alive()

    def run_job(fn, *a):
        nonlocal worker
        arm.clear_stop()
        worker = threading.Thread(target=fn, args=a, daemon=True)
        worker.start()
    window = "Vision-guided AR4 (G=go P=park Q=quit)"
    multi = False
    tracker = mt.MultiTracker(cfg.STABLE_SAMPLES, cfg.STABLE_SPREAD_MM, inside)
    selected_id = None
    clicks = []
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, lambda ev, x, y, *_: clicks.append((x, y))
                         if ev == cv2.EVENT_LBUTTONDOWN else None)
    print("M = switch single / multi-target mode")
    try:
        while True:
            frame = detector.camera.read()
            if frame is None:
                print("Camera frame unavailable")
                break
            calibrated = aruco.update(frame)
            if multi:
                display = frame.copy()
                if not calibrated:
                    tracker.clear()
                elif not busy():
                    tracker.update(mt.find_blobs(detector, frame), aruco.pixel_to_workspace)
                center, stable = None, False
                ready = tracker.ready()
                if tracker.by_id(selected_id) is None:
                    selected_id = None
                if selected_id is None and len(ready) == 1:
                    selected_id = ready[0][1].id   # only one candidate: select it
                sel = tracker.by_id(selected_id)
                if sel is not None:
                    ok, reason = tracker.status(sel)
                    med, center = sel.board, sel.pixel
                    reason = "READY - G = go to selected, A = visit all" if ok else f"selected: {reason}"
                else:
                    ok, med = False, None
                    reason = ("click a target or press its number" if ready
                              else "no ready targets")
                if not calibrated:
                    ok, reason = False, "markers not all visible"
                mt.draw(display, tracker, selected_id)
            else:
                display = detector.process_frame(frame)
                center = detector.get_tracked_center()
                stable = detector.is_target_stable()

            if not multi:
                if calibrated and center is not None and stable and not busy():
                    history.append(aruco.pixel_to_workspace(center))
                else:
                    history.clear()
                ok, reason, med = check_target(history, calibrated, stable, inside)

            lines = [(f"markers: {sorted(aruco.visible_ids)}", GREEN if calibrated else RED)]
            if multi:
                sel_num = next((n for n, tr in tracker.numbered() if tr.id == selected_id), None)
                lines.append((f"MULTI: {len(tracker.numbered())} seen, {len(ready)} ready, "
                              f"selected {'#' + str(sel_num) if sel_num else 'none'}  (M = single)",
                              YELLOW))
            if med is not None and robot_frame:
                lines.append((f"robot X {med[0]:.1f}  Y {med[1]:.1f} mm", YELLOW))
            elif med is not None:
                lines.append((f"board X {med[0]:.1f}  Y {med[1]:.1f} mm", YELLOW))
                if calib is not None:
                    rx, ry = board_to_robot(calib, *med)
                    lines.append((f"robot X {rx:.1f}  Y {ry:.1f} mm", YELLOW))
            lines.append((f"hover {live['target_height_mm'] + live['hover_clearance_mm']:.0f} mm "
                          f"= target {live['target_height_mm']:.0f} + clear "
                          f"{live['hover_clearance_mm']:.0f}   (+/- ]/[  C=new target)", (255, 255, 255)))
            if busy():
                lines.append((f"MOVING: {status['text']}  (SPACE = stop)", YELLOW))
            else:
                lines.append((("DRY RUN: " if args.dry else "") + reason, GREEN if ok else RED))
                if status["text"]:
                    lines.append((f"last: {status['text']}", YELLOW))
            for i, (text, colour) in enumerate(lines):
                cv2.putText(display, text, (20, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, colour, 2)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != 255 and not busy() and adjust_live(live, key):
                continue
            if clicks:
                u, v = clicks.pop()
                clicks.clear()
                if multi and not busy():
                    hit = tracker.nearest_to_pixel(u, v)
                    if hit is not None:
                        selected_id = hit.id
            if key == ord("m"):
                if busy():
                    print("Wait for the arm to finish before switching mode")
                    continue
                multi = not multi
                tracker.clear()
                history.clear()
                selected_id = None
                print("Mode:", "MULTI-target" if multi else "single-target")
                continue
            if multi and ord("1") <= key <= ord("9") and not busy():
                want = key - ord("0")
                hit = next((tr for n, tr in tracker.numbered() if n == want), None)
                if hit is not None:
                    selected_id = hit.id
                continue
            if key == ord("c"):
                if busy():
                    print("Wait for the arm to finish before changing the target")
                    continue
                print("Select the new target: drag a box over it, then ENTER (C to cancel)")
                new_range = detector.calibrate_color()
                try:
                    cv2.destroyWindow("Calibration")
                except cv2.error:
                    pass
                if new_range is None:
                    print("Target change cancelled - keeping the previous target")
                else:
                    detector.color_range = new_range
                    history.clear()
                    tracker.clear()
                    selected_id = None
                    status["text"] = "new target selected"
                    print("New target selected")
                continue
            if arm is None:
                continue
            if key == ord(" "):
                if busy():
                    print("STOP requested")
                    arm.stop()
                continue
            if busy():
                continue  # ignore other keys while moving
            if key == ord("g"):
                if not ok:
                    print("Refused:", reason)
                    continue
                history.clear()
                run_job(hover_over, arm, calib, med, center, status, dict(live))
            elif key == ord("a") and multi:
                if not calibrated:
                    print("Refused: markers not all visible")
                    continue
                targets = [(n, tr.board, tr.pixel) for n, tr in tracker.ready()]
                if not targets:
                    print("Refused: no ready targets")
                    continue
                print(f"Visiting {len(targets)} targets: " +
                      ", ".join(f"#{n} ({b[0]:.0f},{b[1]:.0f})" for n, b, _ in targets))
                run_job(visit_all, arm, calib, targets, status, dict(live))
            elif key == ord("p"):
                def park_job():
                    try:
                        status["text"] = "parking"
                        go_park(arm, calib)
                        status["text"] = "parked"
                    except Exception as e:
                        status["text"] = f"park failed: {e}"
                run_job(park_job)
    finally:
        detector.camera.stop()
        cv2.destroyAllWindows()
        if arm is not None:
            if busy():
                arm.stop()
                worker.join(timeout=5)
            arm.close()


if __name__ == "__main__":
    main()
