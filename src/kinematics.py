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

# Default joint limits (deg). The firmware also enforces its own configured limits.
LIMITS = np.array([(-170, 170), (-42, 90), (-89, 52), (-180, 180), (-105, 105), (-180, 180)],
                  dtype=float)

POS_TOL = 0.01      # mm
ROT_TOL = 1e-4      # rad
MAX_STEP_DEG = 12.0   # largest joint change allowed between 5 mm path samples
PATH_DEV_MM = 2.0     # max deviation from the straight line between sent waypoints


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
    # keep J4/J6 nearest the seed (they are the wrap-around joints)
    for i in (3, 5):
        while q[i] - seed[i] > 180:
            q[i] -= 360
        while q[i] - seed[i] < -180:
            q[i] += 360
    return q


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
    n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / sample_mm)))
    # orientation interpolated by slerp-like rotation vector of the relative rotation
    rel = R_to_rotvec(R0.T @ R1)
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
                f"path passes through the wrist singularity (a joint would swing {jump:.0f} deg "
                f"in {sample_mm:.0f} mm) - approach from another direction or tilt the gripper")
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
