"""python main.py limits                      show the joint limits every tool uses
   python main.py limits J3 -89 70            set J3 to -89..70 deg (writes joint_limits.json)
   python main.py limits J3 default           put J3 back to the Annin default
   python main.py limits reset                all joints back to Annin defaults

Use the SAME numbers as the HMI (its joint limit settings). These limits are what
jog, IK and board moves refuse to go past. The firmware/HMI limits still apply on top,
and the real hard stop is the limit switch end - do not set a value the HMI does not allow.
"""
import json
import os

import kinematics as K


def _show(lim):
    print(f"Joint limits ({'from ' + K.LIMITS_FILE if os.path.exists(K.LIMITS_FILE) else 'Annin defaults - no joint_limits.json yet'}):")
    for i, (lo, hi) in enumerate(lim):
        dlo, dhi = K.DEFAULT_LIMITS[i]
        tag = "" if (lo, hi) == (dlo, dhi) else f"   (default {dlo}..{dhi})"
        print(f"  J{i + 1}: {lo:7.1f} .. {hi:6.1f}{tag}")


def run(args):
    lim = K.load_limits()
    if not args:
        _show(lim)
        return
    if args[0].lower() == "reset":
        if os.path.exists(K.LIMITS_FILE):
            os.remove(K.LIMITS_FILE)
        _show(K.load_limits())
        return
    j = args[0].upper().lstrip("J")
    if not j.isdigit() or not 1 <= int(j) <= 6:
        print(__doc__)
        return
    i = int(j) - 1
    if len(args) == 2 and args[1].lower() == "default":
        lim[i] = K.DEFAULT_LIMITS[i]
    elif len(args) == 3:
        lo, hi = float(args[1]), float(args[2])
        if not lo < hi:
            print("Give the low value first, e.g. limits J3 -89 70")
            return
        if lo < -190 or hi > 190:
            print("Refused: limits beyond +-190 deg are not physical for the AR4")
            return
        lim[i] = (lo, hi)
    else:
        print(__doc__)
        return
    data = {f"J{k + 1}": [float(a), float(b)] for k, (a, b) in enumerate(lim)}
    with open(K.LIMITS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    _show(K.load_limits())
