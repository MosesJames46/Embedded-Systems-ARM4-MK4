"""Joint-by-joint motor test: does every motor move the right way, the right amount?

    python main.py jointtest                    all six joints, 10 degrees
    python main.py jointtest --joints 6 --deg 15
    python main.py jointtest --firmware 6

This is the AR4 manual's startup check (direction diagram on manual page 366), run one
joint at a time at 10 % speed, with the answers recorded in joint_test_results.csv.

For each joint:  start -> +deg -> start -> -deg -> start
  * The controller's reported angle must change by the commanded amount, and the
    OTHER joints must not move (catches a motor that only turns one way, a swapped
    motor, or a joint that drags another).
  * You confirm by eye: right direction (compare with the diagram), roughly the
    right amount (a digital angle gauge gives a number), and back to the start.

Start with the arm high and clear of the table, people and the board (the test refuses
to start below 150 mm). Keep a hand on the red E-stop. Close the HMI first.
"""
import argparse
import csv
import os
import time

import vision_config as cfg
from robot_backend import connect, RobotFault

# Annin's default joint limits (degrees from zero). Your HMI settings may differ; the
# firmware enforces the real ones and refuses moves beyond them.
LIMITS = {1: (-170, 170), 2: (-42, 90), 3: (-89, 52), 4: (-180, 180), 5: (-105, 105), 6: (-180, 180)}
MARGIN = 5.0
ANGLE_TOL = 0.5     # reported angle must match within this (deg)
OTHER_TOL = 0.5     # other joints may not move more than this (deg)
MIN_START_Z = 150.0
RESULTS = os.path.join(cfg.HERE, "joint_test_results.csv")


def ask(prompt, options="yn"):
    while True:
        a = input(f"{prompt} [{'/'.join(options)}] ").strip().lower()
        if a and a[0] in options:
            return a[0]


def run(argv=None):
    ap = argparse.ArgumentParser(prog="main.py jointtest")
    ap.add_argument("--port", default=None)
    ap.add_argument("--firmware", choices=["8", "6"], default=None)
    ap.add_argument("--joints", default="1,2,3,4,5,6", help="e.g. 1,2,6")
    ap.add_argument("--deg", type=float, default=10.0, help="test angle, 2-20 deg")
    ap.add_argument("--speed", type=int, default=10)
    args = ap.parse_args(argv)
    deg = max(2.0, min(args.deg, 20.0))
    speed = max(1, min(args.speed, 15))
    joints = [int(j) for j in args.joints.split(",") if j.strip()]

    from warmup import _firmware_hint
    fw = args.firmware or _firmware_hint()
    arm = connect(fw, args.port, cfg.LOOP_MODE)
    rows = []
    try:
        start = arm.pose()
        print(f"Connected (firmware {fw}). Joints: " +
              "  ".join(f"J{i + 1} {a:.1f}" for i, a in enumerate(start["joints"])))
        if start["z"] < MIN_START_Z:
            raise SystemExit(f"Tool is at Z {start['z']:.0f} mm. Raise the arm above "
                             f"{MIN_START_Z:.0f} mm and clear of the table first (jog window, SPACE).")
        if ask("Is the arm clear of the table, the board and people, with the E-stop in reach?") != "y":
            raise SystemExit("Stopped - clear the area first.")
        print("\nHave the manual's direction diagram (page 366) in front of you.")

        for jn in joints:
            i = jn - 1
            home = list(arm.pose()["joints"])
            lo, hi = LIMITS[jn]
            print(f"\n===== J{jn} (now {home[i]:.1f} deg, default limits {lo} .. {hi}) =====")
            for sign in (+1, -1):
                target = home[i] + sign * deg
                label = f"J{jn}{'+' if sign > 0 else '-'}{deg:.0f}"
                if not (lo + MARGIN <= target <= hi - MARGIN):
                    print(f"  {label}: skipped - {target:.1f} is too close to the joint limit")
                    rows.append({"joint": jn, "move": label, "result": "skipped (near limit)"})
                    continue
                angles = list(home)
                angles[i] = target
                input(f"  Next: {label}. Watch J{jn}, then press Enter to move ")
                row = {"joint": jn, "move": label, "time": time.strftime("%H:%M:%S")}
                try:
                    p = arm.move_joints(angles, speed)
                except RobotFault as e:
                    print(f"  REFUSED/FAULT: {e}")
                    row["result"] = f"fault: {e}"
                    rows.append(row)
                    break
                moved = p["joints"][i] - home[i]
                others = max((abs(p["joints"][k] - home[k]) for k in range(6) if k != i), default=0)
                ctrl_ok = abs(moved - sign * deg) <= ANGLE_TOL and others <= OTHER_TOL
                print(f"  controller: J{jn} moved {moved:+.2f} deg (commanded {sign * deg:+.0f}), "
                      f"other joints max {others:.2f} deg -> {'OK' if ctrl_ok else 'MISMATCH'}")
                d = ask(f"  Did J{jn} turn in its {'+' if sign > 0 else '-'} direction per the diagram?",
                        "yns")
                gauge = input("  Angle-gauge reading in degrees (Enter to skip): ").strip()
                p = arm.move_joints(home, speed)
                back = max(abs(p["joints"][k] - home[k]) for k in range(6))
                r = ask(f"  Did J{jn} return to where it started?", "yn")
                ok = ctrl_ok and d == "y" and r == "y" and back <= ANGLE_TOL
                if gauge:
                    try:
                        ok = ok and abs(abs(float(gauge)) - deg) <= 1.0
                    except ValueError:
                        pass
                row.update(controller_deg=round(moved, 2), other_joints_deg=round(others, 2),
                           direction_ok={"y": "yes", "n": "NO", "s": "unsure"}[d], gauge_deg=gauge,
                           returned={"y": "yes", "n": "NO"}[r], result="PASS" if ok else "FAIL")
                rows.append(row)
                print(f"  -> {row['result']}")
    except RobotFault as e:
        print("Robot error:", e)
    finally:
        arm.close()

    if rows:
        new = not os.path.exists(RESULTS)
        fields = ["time", "joint", "move", "controller_deg", "other_joints_deg",
                  "direction_ok", "gauge_deg", "returned", "result"]
        with open(RESULTS, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)
        print("\n===== SUMMARY =====")
        for r in rows:
            print(f"  {r['move']:<10} {r.get('result')}")
        fails = [r for r in rows if r.get("result") not in ("PASS",) and "skipped" not in r.get("result", "")]
        print("\nALL JOINTS PASSED" if not fails else f"\n{len(fails)} problem(s) - see the fixes below")
        if fails:
            print("  Wrong direction        -> motor wiring (A+A-B+B-) or the driver's direction setting")
            print("  Moves one way only     -> the DIR wire from the Teensy to that driver")
            print("  Wrong amount (gauge)   -> microstep DIP switches on that driver")
            print("  Other joints moved     -> J1/J3 motor cables swapped, or a loose coupling")
            print("  Did not return         -> skipped steps: check coupling set screws and driver current")
        print(f"Results appended to {os.path.basename(RESULTS)}")


if __name__ == "__main__":
    run()
