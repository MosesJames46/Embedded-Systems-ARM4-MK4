"""Milestone 4: camera-guided AR4 positioning above a coloured target.

Pipeline: webcam -> colour detection (ColorDetector) -> pixel to board mm
(ArUco markers 4-7) -> board mm to robot mm (robot_calib.json from teach_robot.py)
-> safe approach, hover, retreat, park.

Run:
    python vision_move.py --dry     # vision only, no serial connection, no motion
    python vision_move.py           # live; the arm moves only when you press G

Keys in the video window:
    G      move to hover above the current target (only when the status is READY)
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

from ColorPicker import ColorPicker
from ColorDetector import ColorDetector
from Camera import Camera
from ArUcoCalibration import ArUcoCalibration
import vision_config as cfg
from robot_backend import connect, RobotFault

GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 255), (0, 200, 255)


def load_calib():
    if not os.path.exists(cfg.CALIB_FILE):
        return None
    with open(cfg.CALIB_FILE) as f:
        c = json.load(f)
    c["M"] = np.array(c["affine_board_to_robot"], dtype=np.float64)
    return c


def board_to_robot(calib, x, y):
    rx, ry = calib["M"] @ np.array([x, y, 1.0])
    return float(rx), float(ry)


def board_limits():
    xs = [p[0] for p in cfg.MARKER_POSITIONS_MM.values()]
    ys = [p[1] for p in cfg.MARKER_POSITIONS_MM.values()]
    m = cfg.BOARD_MARGIN_MM
    return min(xs) + m, max(xs) - m, min(ys) + m, max(ys) - m


def check_target(history, calibrated, detector_stable):
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
    x0, x1, y0, y1 = board_limits()
    if not (x0 <= med[0] <= x1 and y0 <= med[1] <= y1):
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
    arm.move_linear(park["x"], park["y"], max(park["z"], safe), park["rot"], cfg.SPEED_PCT)


def hover_over(arm, calib, board_xy, pixel, status):
    rot = calib["tool_rot"]
    bz = calib["board_z"]
    safe = bz + cfg.SAFE_Z_MM
    hover = bz + cfg.TARGET_HEIGHT_MM + cfg.HOVER_CLEARANCE_MM
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

    detector = ColorDetector(ColorPicker(), Camera(args.camera))
    aruco = ArUcoCalibration(cfg.MARKER_POSITIONS_MM)
    print("Drag a box over the target colour, then press ENTER.")
    detector.color_range = detector.calibrate_color()
    cv2.destroyAllWindows()
    if detector.color_range is None:
        raise SystemExit("Colour calibration cancelled")

    history = deque(maxlen=cfg.STABLE_SAMPLES)
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
    try:
        while True:
            frame = detector.camera.read()
            if frame is None:
                print("Camera frame unavailable")
                break
            calibrated = aruco.update(frame)
            display = detector.process_frame(frame)
            center = detector.get_tracked_center()
            stable = detector.is_target_stable()

            if calibrated and center is not None and stable and not busy():
                history.append(aruco.pixel_to_workspace(center))
            else:
                history.clear()
            ok, reason, med = check_target(history, calibrated, stable)

            lines = [(f"markers: {sorted(aruco.visible_ids)}", GREEN if calibrated else RED)]
            if med is not None:
                lines.append((f"board X {med[0]:.1f}  Y {med[1]:.1f} mm", YELLOW))
                if calib is not None:
                    rx, ry = board_to_robot(calib, *med)
                    lines.append((f"robot X {rx:.1f}  Y {ry:.1f} mm", YELLOW))
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
                run_job(hover_over, arm, calib, med, center, status)
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
