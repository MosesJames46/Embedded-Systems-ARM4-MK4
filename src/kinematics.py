"""AR4 kinematics on the laptop: forward kinematics, continuity-preserving inverse
kinematics, and joint-space path planning.

Why: the 6.x firmware's inverse kinematics picks between two wrist solutions with a
test dominated by J6, so near the wrist singularity (J5 ~ 0, e.g. close to the robot
with the gripper pointing down) it jumps to the mirrored wrist: J4/J5/J6 swing ~180 deg
before the move. Here the IK is solved iteratively FROM THE CURRENT JOINTS, so it always
stays on the same wrist branch, and the result is sent as a direct joint move (RJ),
which bypasses the firmware IK entirely.

Model: the firmware's own modified-DH table and conventions (verified against 9 poses
from the real robot: position and orientation match to 0.01 mm / 0.05 deg).
Orientation values are the firmware's ROTATION VECTOR in degrees (axis * angle) - the
same three numbers the controller reports as "rot".
"""
import numpy as np

# Annin 6.x default modified-DH table: (theta offset deg, alpha deg, d mm, a mm)
DH = [(0, 0, 169.77, 0), (-90, -90, 0, 64.2), (0, 0, 0, 305),
      (0, -90, 222.63, 0), (0, 90, 0, 0), (180, -90, 41, 0)]

# Joint limits (deg). Annin defaults below; set YOUR robot's values (the ones in the
# HMI) with `python main.py limits`, which writes joint_limits.json next to this file.
# Every tool (jog, IK, board moves, jointtest) reads them from here.
import json as _json
import os as _os

DEFAULT_LIMITS = [(-170, 170), (-42, 90), (-89, 52), (-180, 180), (-105, 105), (-180, 180)]
LIMITS_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "joint_limits.json")


def load_limits(path=LIMITS_FILE):
    lim = [list(x) for x in DEFAULT_LIMITS]
    if _os.path.exists(path):
        with open(path) as f:
            data = _json.load(f)
        for i in range(6):
            v = data.get(f"J{i + 1}")
            if v is not None:
                lo, hi = float(v[0]), float(v[1])
                if not lo < hi:
                    raise ValueError(f"{path}: J{i + 1} limits {v} must be [low, high]")
                lim[i] = [lo, hi]
    return np.array(lim, dtype=float)


LIMITS = load_limits()

POS_TOL = 0.01      # mm
ROT_TOL = 1e-4      # rad
MAX_STEP_DEG = 12.0   # largest joint change allowed between 5 mm path samples
PATH_DEV_MM = 2.0     # max deviation from the straight line between sent waypoints
SAMPLE_DEG = 1.0      # rotation sampled every 1 deg when planning


class KinematicsError(RuntimeError):
    pass


def _dhm(alpha, a, theta, d):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -sa * d],
                     [st * sa, ct * sa, ca, ca * d],
                     [0, 0, 0, 1]])


def fk(q):
    """Joint angles (deg) -> 4x4 flange pose (mm)."""
    M = np.eye(4)
    for (th0, al, d, a), qi in zip(DH, q):
        M = M @ _dhm(np.radians(al), a, np.radians(th0 + qi), d)
    return M


def rotvec_to_R(rv_deg):
    rv = np.radians(np.asarray(rv_deg, float))
    ang = np.linalg.norm(rv)
    if ang < 1e-9:
        return np.eye(3)
    k = rv / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if np.pi - ang < 1e-6:                      # 180 deg: axis from the diagonal
        k = np.sqrt(np.clip((np.diag(R) + 1) / 2, 0, 1))
        i = int(np.argmax(k))
        k = R[:, i] + np.eye(3)[:, i]
        k /= np.linalg.norm(k)
        return np.degrees(ang) * k
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return np.degrees(ang) * v


def pose_of(q):
    """(xyz, rotvec deg) like the controller reports."""
    M = fk(q)
    return M[:3, 3], R_to_rotvec(M[:3, :3])


def rotate_about_world(rv_deg, axis, deg):
    """Turn an orientation about a world axis ('x', 'y' or 'z') through the tool point."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
         "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
         "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]
    return R_to_rotvec(R @ rotvec_to_R(rv_deg)).tolist()


def tilt_toward(rv_deg, direction_xy, deg):
    """Tilt an orientation so the gripper TIP swings by `deg` toward the horizontal
    direction_xy (e.g. away from the robot base = the J5 direction). Rotation about the
    horizontal axis perpendicular to that direction."""
    r = np.array([direction_xy[0], direction_xy[1], 0.0], float)
    r /= np.linalg.norm(r)
    n = np.cross(r, [0.0, 0.0, 1.0])
    n /= np.linalg.norm(n)
    a = np.radians(deg)
    Kx = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    Rn = np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
    return R_to_rotvec(Rn @ rotvec_to_R(rv_deg)).tolist()


def tilt_from_down(rv_deg):
    """Degrees between the gripper axis and straight down (0 = pointing straight down)."""
    z = rotvec_to_R(rv_deg)[:, 2]
    return float(np.degrees(np.arccos(np.clip(-z[2], -1, 1))))


def rotate_about_vertical(rv_deg, deg):
    """Turn an orientation about the world vertical axis (gripper yaw)."""
    a = np.radians(deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return R_to_rotvec(Rz @ rotvec_to_R(rv_deg)).tolist()


def _error(q, p, R):
    M = fk(q)
    dp = p - M[:3, 3]
    Rc = M[:3, :3]
    dR = 0.5 * (np.cross(Rc[:, 0], R[:, 0]) + np.cross(Rc[:, 1], R[:, 1]) + np.cross(Rc[:, 2], R[:, 2]))
    return np.concatenate([dp, dR * 200.0])     # weight: 1 rad of rotation ~ 200 mm


def ik(p, R, seed, iters=200):
    """Damped least squares from `seed` -> nearest solution (same wrist branch)."""
    q = np.array(seed, float)
    lam = 0.5
    for _ in range(iters):
        e = _error(q, p, R)
        if np.linalg.norm(e[:3]) < POS_TOL and np.linalg.norm(e[3:]) / 200 < ROT_TOL:
            break
        J = np.zeros((6, 6))
        h = 1e-4
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = h
            J[:, i] = (_error(q, p, R) - _error(q + dq, p, R)) / np.radians(h)
        # damped least squares step (radians -> degrees)
        step = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), e)
        q = q + np.degrees(step)
    else:
        e = _error(q, p, R)
        if np.linalg.norm(e[:3]) > 0.5 or np.linalg.norm(e[3:]) / 200 > 0.01:
            raise KinematicsError("target out of reach for this gripper orientation")
    # J1, J2, J3, J5 never need more than half a turn: wrap them into -180..180
    for i in (0, 1, 2, 4):
        q[i] = (q[i] + 180.0) % 360.0 - 180.0
    # keep J4/J6 nearest the seed (they are the wrap-around joints)
    for i in (3, 5):
        while q[i] - seed[i] > 180:
            q[i] -= 360
        while q[i] - seed[i] < -180:
            q[i] += 360
    return q


def ik_best(p, R, q0):
    """For a long (joint) move: try several starting guesses and return the in-limits
    solution closest to the current joints q0 - so a far target cannot land on an odd
    branch (wrist folded the other way, base turned a full circle)."""
    p = np.asarray(p, float)
    q0 = np.asarray(q0, float)
    j1 = np.degrees(np.arctan2(p[1], p[0]))
    seeds = [q0, np.r_[j1, q0[1:]]]
    for j5 in (40.0, -40.0, 10.0):
        seeds.append(np.array([j1, 40.0, 20.0, 0.0, j5, q0[5]]))
    best, best_cost, last = None, None, None
    for s in seeds:
        try:
            q = ik(p, R, s)
            q = np.array(q)
            for i in (3, 5):                      # wrist turns nearest the CURRENT joints
                while q[i] - q0[i] > 180:
                    q[i] -= 360
                while q[i] - q0[i] < -180:
                    q[i] += 360
            check_limits(q)
        except KinematicsError as e:
            last = e
            continue
        cost = np.max(np.abs(q - q0)) + 0.5 * (abs(q[3] - q0[3]) + abs(q[5] - q0[5]))
        if best is None or cost < best_cost:
            best, best_cost = q, cost
    if best is None:
        raise last or KinematicsError("target not reachable")
    return best


def check_limits(q, limits=LIMITS, margin=1.0):
    for i, (lo, hi) in enumerate(limits):
        if not (lo + margin <= q[i] <= hi - margin):
            raise KinematicsError(f"J{i + 1} would be at {q[i]:.1f} deg, outside its limits "
                                  f"{lo:.0f}..{hi:.0f} - target not reachable this way")


def plan_line(q0, p_target, rv_target, sample_mm=5.0):
    """Joint waypoints for a straight-line move from the pose at q0 to p_target with
    orientation rv_target. Samples every sample_mm, solving each from the previous one
    (no branch jumps), then keeps only the waypoints needed so that moving joint-by-joint
    between them never strays more than PATH_DEV_MM from the line."""
    q0 = np.asarray(q0, float)
    p0, _ = pose_of(q0)
    R1 = rotvec_to_R(rv_target)
    R0 = fk(q0)[:3, :3]
    p1 = np.asarray(p_target, float)
    # orientation interpolated by slerp-like rotation vector of the relative rotation
    rel = R_to_rotvec(R0.T @ R1)
    # sample often enough for BOTH the distance (every sample_mm) and the rotation
    # (every SAMPLE_DEG), so a pure tilt is not solved as one big jump
    n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / sample_mm)),
            int(np.ceil(np.linalg.norm(rel) / SAMPLE_DEG)))
    qs, ps = [q0], [p0]
    q = q0
    for k in range(1, n + 1):
        s = k / n
        p = p0 + s * (p1 - p0)
        R = R0 @ rotvec_to_R(rel * s)
        qn = ik(p, R, q)
        jump = np.max(np.abs(qn - q))
        if jump > MAX_STEP_DEG:
            raise KinematicsError(
                f"too close to the wrist singularity (J5 near 0): a joint would swing {jump:.0f} deg "
                "in one small step - tilt the gripper a little first, or approach from another direction")
        check_limits(qn)
        qs.append(qn)
        ps.append(p)
        q = qn
    # thin out: greedily keep the farthest waypoint whose joint-linear segment stays on the line
    keep = [0]
    i = 0
    while i < n:
        j = n
        while j > i + 1 and _segment_dev(qs[i], qs[j], ps[i], ps[j]) > PATH_DEV_MM:
            j -= 1
        keep.append(j)
        i = j
    return [qs[k] for k in keep[1:]]


def _segment_dev(qa, qb, pa, pb, samples=8):
    d = pb - pa
    L = np.linalg.norm(d)
    worst = 0.0
    for s in np.linspace(0, 1, samples + 1)[1:-1]:
        p = fk(qa + s * (qb - qa))[:3, 3]
        if L < 1e-9:
            dev = np.linalg.norm(p - pa)
        else:
            t = np.clip(np.dot(p - pa, d) / L ** 2, 0, 1)
            dev = np.linalg.norm(p - (pa + t * d))
        worst = max(worst, dev)
    return worst


def model_matches(q, xyz, rv, tol_mm=1.0, tol_deg=0.5):
    """True if this laptop model reproduces what the controller reports (no tool frame
    or DH change in the HMI). Used as a safety check before using local IK."""
    p, r = pose_of(q)
    Rd = rotvec_to_R(r).T @ rotvec_to_R(rv)
    ang = np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1)))
    return np.linalg.norm(p - np.asarray(xyz)) <= tol_mm and ang <= tol_deg
