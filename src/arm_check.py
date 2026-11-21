"""python main.py armcheck     compare the controller's X/Y/Z readout with the joint angles

Prints the joint angles, the position the controller reports, and the position those joint
angles really give (the laptop model, used by every program for planning and saving). A
gap means the controller's readout is off - harmless now, since only the joints are used.
"""
import numpy as np

import kinematics as K


def run(argv):
    from ar4link import AR4
    a = AR4(None)
    try:
        p = a.pose()
        xyz, rv = K.pose_of(p.joints)
        gap = np.asarray(xyz) - np.asarray(p.xyz)
        Rd = K.rotvec_to_R(rv).T @ K.rotvec_to_R([p.rz, p.ry, p.rx])
        ang = np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1)))
        print("joints            " + "  ".join(f"J{i + 1} {v:8.3f}" for i, v in enumerate(p.joints)))
        print(f"controller says   X {p.x:8.2f}  Y {p.y:8.2f}  Z {p.z:8.2f}   rot {p.rz:.2f} {p.ry:.2f} {p.rx:.2f}")
        print(f"joints give       X {xyz[0]:8.2f}  Y {xyz[1]:8.2f}  Z {xyz[2]:8.2f}   rot "
              + " ".join(f"{v:.2f}" for v in rv))
        print(f"gap               {np.linalg.norm(gap):.2f} mm  ({gap[0]:+.2f} {gap[1]:+.2f} {gap[2]:+.2f}),"
              f"  {ang:.2f} deg")
        print(f"gripper tilt from straight down: {K.tilt_from_down(rv):.1f} deg")
        print("OK - readout agrees." if np.linalg.norm(gap) <= 1 and ang <= 0.5 else
              "Readout disagrees - harmless for planning (joints are used), but tell Claude these "
              "numbers: points saved by older versions came from this readout.")
    finally:
        a.close()
