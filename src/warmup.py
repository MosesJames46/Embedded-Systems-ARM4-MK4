"""Pre-session warm-up and self-test for the AR4 (no camera, no board needed).

    python main.py warmup                 (firmware from the calibration files, else config)
    python main.py warmup --firmware 6 --cycles 3 --step 20

What it does, at 10 % speed, from wherever the arm is:
  1. Connects, prints the joints, warns if J5 is near 0 (wrist-singularity zone).
  2. Rises 30 mm to get clear, then for each axis X, Y, Z: moves +step, back, -step, back.
     Checks each move arrived (reported position within 1 mm) and that the gripper
     orientation and J6 did not change (the "gripper keeps turning" fault).
  3. Repeats the axis cycle --cycles times and reports how far the return position wanders
     (repeatability, from the controller's reported position).
  4. Opens and closes the gripper twice if the Nano gripper board is connected.
  5. Returns to the start position and prints PASS / FAIL per check.

Watch the arm while it runs: the reported numbers come from the controller, so a slipping
coupling or skipped steps show up only to your eyes. The red E-stop must be within reach.
"""
import argparse
import json
import os
import time

import numpy as np

import vision_config as cfg
from robot_backend import connect, RobotFault

POS_TOL_MM = 1.0
ROT_TOL_DEG = 0.5
J6_TOL_DEG = 1.0
CLEAR_MM = 30


def _firmware_hint():
    for path in (cfg.CALIB_FILE, os.path.join(cfg.HERE, "chess_board.json")):
        if os.path.exists(path):
            with open(path) as f:
                fw = json.load(f).get("firmware")
            if fw:
                return fw
    return cfg.FIRMWARE


def _rot_err(a, b):
    """Largest orientation difference in degrees, wrap-safe."""
    return max(abs((x - y + 180) % 360 - 180) for x, y in zip(a, b))


def run(argv=None):
    ap = argparse.ArgumentParser(prog="main.py warmup")
    ap.add_argument("--port", default=None)
    ap.add_argument("--firmware", choices=["8", "6"], default=None)
    ap.add_argument("--step", type=float, default=20.0, help="mm per axis move (5-50)")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--no-gripper", action="store_true")
    ap.add_argument("--gripper-port", default=None)
    args = ap.parse_args(argv)
    step = max(5.0, min(args.step, 50.0))
    speed = min(cfg.SPEED_PCT, 10)

    fw = args.firmware or _firmware_hint()
    arm = connect(fw, args.port, cfg.LOOP_MODE)
    results = []

    def check(name, ok, detail):
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    try:
        start = arm.pose()
        rot = list(start["rot"])          # locked for the whole test
        j = start["joints"]
        print(f"Connected (firmware {fw}).")
        print(f"Start: X {start['x']:.1f}  Y {start['y']:.1f}  Z {start['z']:.1f}  "
              f"J " + " ".join(f"{a:.1f}" for a in j))
        check("wrist clear of singularity", abs(j[4]) >= 5,
              f"J5 = {j[4]:.1f} deg" + ("" if abs(j[4]) >= 5 else " (near 0: moves may spin the wrist)"))

        base = (start["x"], start["y"], start["z"] + CLEAR_MM)
        print(f"\nRising {CLEAR_MM} mm to get clear ...")
        p = arm.move_linear(*base, rot, speed)
        check("rise", abs(p["z"] - base[2]) <= POS_TOL_MM, f"Z {p['z']:.1f} (target {base[2]:.1f})")
        j6_ref = p["joints"][5]

        returns = []
        for c in range(1, args.cycles + 1):
            print(f"\nAxis cycle {c}/{args.cycles} (step {step:.0f} mm, speed {speed}%)")
            for axis in range(3):
                for sign in (+1, -1):
                    tgt = list(base)
                    tgt[axis] += sign * step
                    p = arm.move_linear(*tgt, rot, speed)
                    err = float(np.linalg.norm(np.subtract((p["x"], p["y"], p["z"]), tgt)))
                    name = f"{'XYZ'[axis]}{'+' if sign > 0 else '-'}{step:.0f}"
                    if err > POS_TOL_MM:
                        check(f"cycle {c} {name}", False, f"arrived {err:.2f} mm off target")
                    p = arm.move_linear(*base, rot, speed)
                    returns.append((p["x"], p["y"], p["z"]))
                    rerr = _rot_err(p["rot"], rot)
                    dj6 = abs((p["joints"][5] - j6_ref + 180) % 360 - 180)
                    if rerr > ROT_TOL_DEG or dj6 > J6_TOL_DEG:
                        check(f"cycle {c} {name} orientation", False,
                              f"orientation off by {rerr:.2f} deg, J6 moved {dj6:.2f} deg")
            print("  cycle done")

        pts = np.array(returns)
        spread = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
        check("axis moves arrive on target", not any(n.startswith("cycle") and "orientation" not in n
                                                    and not ok for n, ok, _ in results),
              f"all {len(returns)} moves within {POS_TOL_MM} mm")
        check("repeatability (return position)", spread <= POS_TOL_MM,
              f"max wander {spread:.2f} mm over {len(returns)} returns")
        check("gripper orientation steady", not any("orientation" in n and not ok
                                                   for n, ok, _ in results),
              f"orientation within {ROT_TOL_DEG} deg, J6 within {J6_TOL_DEG} deg")

        if not args.no_gripper:
            try:
                from gripper import Gripper
                g = Gripper(args.gripper_port)
                print(f"\nGripper ({g.protocol} on {g.port}): open/close x2 - watch it")
                for _ in range(2):
                    g.open()
                    g.close()
                g.open()
                g.shutdown()
                check("gripper responds", True, "commands sent - confirm it moved")
            except Exception as e:
                check("gripper responds", False, str(e))

        print("\nReturning to the start position ...")
        p = arm.move_linear(start["x"], start["y"], start["z"], rot, speed)
        check("return to start", abs(p["z"] - start["z"]) <= POS_TOL_MM, f"Z {p['z']:.1f}")
    except RobotFault as e:
        check("motion", False, f"{e}  (move refused or stopped - is the arm near a limit?)")
    finally:
        arm.close()

    failed = [r for r in results if not r[1]]
    print("\n" + ("WARM-UP PASSED" if not failed else f"WARM-UP FAILED ({len(failed)} checks)"))
    for name, ok, detail in failed:
        print(f"  - {name}: {detail}")
    print("Reported positions come from the controller: also confirm by eye that nothing "
          "slipped or rotated.")
    return not failed


if __name__ == "__main__":
    run()
