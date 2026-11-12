"""Teach the board-to-robot calibration by jogging the tool over the ArUco markers.

Run (HMI closed, robot already calibrated in the HMI):
    python teach_robot.py                 (firmware 8.1)
    python teach_robot.py --firmware 6    (old 6.x firmware)

Commands (type then Enter):
    x+10  y-5  z+2      jog the tool in the robot base frame (mm)
    p                   print the current pose
    save 0              store the current tool XY as the centre of marker 0 (any ID in
                        MARKER_POSITIONS_MM in vision_config.py)
    save park           store a parking pose that is OUT of the camera view
    s 5                 set jog speed in percent (1-25, default 10)
    w                   write robot_calib.json and report the fit error
    q                   quit

Procedure: jog the tool tip until it sits a few mm above the CENTRE of each marker,
save it, and do this for at least 3 markers (all 4 gives an error check).
The tool orientation of the first saved marker is used for every vision move,
so keep the tool pointing straight down.
"""
import json
import os

import numpy as np
import cv2

import argparse

from robot_backend import connect, RobotFault, MAX_SPEED
from vision_config import MARKER_POSITIONS_MM, CALIB_FILE, LOOP_MODE, FIRMWARE

JOG_LIMIT_MM = 50  # refuse single jogs bigger than this


def fit(points):
    ids = sorted(k for k in points if k in MARKER_POSITIONS_MM)
    if len(ids) < 3:
        raise ValueError("Need at least 3 saved markers")
    src = np.float32([MARKER_POSITIONS_MM[i] for i in ids])
    dst = np.float32([[points[i]["x"], points[i]["y"]] for i in ids])
    if len(ids) == 3:
        M = cv2.getAffineTransform(src, dst)
    else:
        M, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
    pred = (M @ np.hstack([src, np.ones((len(ids), 1), np.float32)]).T).T
    err = np.linalg.norm(pred - dst, axis=1)
    return ids, M, err


def fmt(p):
    r = p["rot"]
    return (f"X {p['x']:.1f}  Y {p['y']:.1f}  Z {p['z']:.1f}  rot {r[0]:.1f} {r[1]:.1f} {r[2]:.1f}"
            f"  | J " + " ".join(f"{a:.1f}" for a in p["joints"]))


def load_saved():
    """Taught points from an existing robot_calib.json (empty dict if none)."""
    if not os.path.exists(CALIB_FILE):
        return {}
    with open(CALIB_FILE) as f:
        old = json.load(f)
    return {int(k) if k.isdigit() else k: v for k, v in old.get("taught", {}).items()}


def write_calib(saved, firmware):
    """Fit board->robot from the saved marker poses and write robot_calib.json."""
    ids, M, err = fit(saved)
    first = saved[ids[0]]
    data = {
        "affine_board_to_robot": M.tolist(),
        "board_z": float(np.mean([saved[i]["z"] for i in ids])),
        "tool_rot": first["rot"],
        "firmware": firmware,
        "markers_used": ids,
        "fit_error_mm": {str(i): float(e) for i, e in zip(ids, err)},
        "taught": {str(k): v for k, v in saved.items()},
    }
    with open(CALIB_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return ids, err


def calib_report(ids, err):
    lines = [f"marker {i}: {e:.2f} mm from your tape measurements" for i, e in zip(ids, err)]
    if max(err) > 15:
        lines.append("Large difference: if MARKER_POSITIONS_MM holds real tape measurements, "
                     "check marker IDs and centring. If you did not measure, ignore this - "
                     "with all markers taught, the taught positions are used directly.")
    if len(ids) == 3:
        lines.append("(3 points fit exactly - teach the 4th marker to get a real error check)")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--firmware", choices=["8", "6"], default=FIRMWARE)
    args = ap.parse_args()
    arm = connect(args.firmware, args.port, LOOP_MODE)
    speed = 10
    saved = load_saved()
    if saved:
        print(f"Loaded existing taught points: {list(saved)}")
    print(__doc__)
    print("Current:", fmt(arm.pose()))

    while True:
        try:
            line = input("teach> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        try:
            if line == "q":
                break
            elif line == "p":
                print(fmt(arm.pose()))
            elif line.startswith("s "):
                speed = max(1, min(MAX_SPEED, int(line.split()[1])))
                print(f"Speed {speed}%")
            elif line[0] in "xyz" and line[1] in "+-":
                d = float(line[1:])
                if abs(d) > JOG_LIMIT_MM:
                    print(f"Refused: jog limit is {JOG_LIMIT_MM} mm")
                    continue
                p = arm.pose()
                x, y, z = p["x"], p["y"], p["z"]
                if line[0] == "x":
                    x += d
                elif line[0] == "y":
                    y += d
                else:
                    z += d
                print(fmt(arm.move_linear(x, y, z, p["rot"], speed=speed)))
            elif line.startswith("save "):
                name = line.split()[1]
                key = int(name) if name.isdigit() else name
                if key != "park" and key not in MARKER_POSITIONS_MM:
                    print(f"Unknown name. Use one of {list(MARKER_POSITIONS_MM)} or park")
                    continue
                saved[key] = arm.pose()
                print(f"Saved {key}: {saved[key]['x']:.1f}, {saved[key]['y']:.1f}, "
                      f"{saved[key]['z']:.1f}")
            elif line == "w":
                if "park" not in saved:
                    print("Save a park pose first (out of camera view): save park")
                    continue
                ids, err = write_calib(saved, args.firmware)
                print(f"Wrote {CALIB_FILE}")
                for line_ in calib_report(ids, err):
                    print(" ", line_)
            else:
                print("Unknown command")
        except (RobotFault, ValueError, IndexError) as e:
            print("ERROR:", e)
    arm.close()


if __name__ == "__main__":
    main()
