"""Solver 1 - per-frame weighted least squares ("WLS").

Each frame is an independent bounded nonlinear least-squares problem in the five joint angles (scipy TRF):
    position error / sigma_p                       (3 residuals)
    orientation error in the demonstrated triad, the component along the arm's LOST axis removed first,
        roll about the approach / sigma_roll, tilts / sigma_tilt     (3 residuals)
    (q - q_prev) / sigma_continuity                (5 residuals, branch keeping)
warm-started from the previous frame.  The first frame tries several seeds.  No knowledge of velocity limits, collisions or
gravity: it is the honest "what the task asks for" baseline that the other solvers are judged against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .kin import ArmKinematics, rot_err, rotation_mobility
from .targets import Targets

SEEDS_DEG = np.array([[0.0, -20.0, 40.0, 20.0, 0.0], [0.0, 30.0, -60.0, 30.0, 0.0], [0.0, 60.0, -90.0, 60.0, 90.0],
                      [0.0, -60.0, 90.0, -30.0, -90.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 45.0, -45.0, 0.0, 150.0],
                      [0.0, -90.0, 90.0, 60.0, 0.0], [0.0, -90.0, 90.0, 60.0, 180.0 - 20.0]])


@dataclass
class FrameEval:
    q: np.ndarray
    pos_err_m: float
    tilt_err_deg: float
    roll_err_deg: float
    lost_deg: float
    limit_margin_deg: float


def evaluate(kin: ArmKinematics, R_t, p_t, jaw: float, q) -> FrameEval:
    q = np.asarray(q, float)
    R, p, J = kin.grip_jac(q, jaw)
    e_r = rot_err(R_t, R)
    _, _, lost = rotation_mobility(J)
    tilt = np.degrees(np.arccos(np.clip(R_t[:, 0] @ R[:, 0], -1, 1)))
    o_t = R_t[:, 1] - R[:, 0] * (R_t[:, 1] @ R[:, 0])
    roll = np.degrees(np.arccos(np.clip((o_t @ R[:, 1]) / max(np.linalg.norm(o_t), 1e-9), -1, 1)))
    return FrameEval(q, float(np.linalg.norm(p_t - p)), float(tilt), float(roll), float(np.degrees(abs(lost @ e_r))),
                     float(np.degrees(kin.limit_margin(q))))


def solve_frame(kin: ArmKinematics, R_t, p_t, jaw: float, q0, q_ref, sig_p: float, sig_tilt_deg: float, sig_roll_deg: float,
                sig_cont_deg: float, project_lost: bool = True, max_nfev: int = 200) -> FrameEval:
    lo, hi = kin.limits[:, 0], kin.limits[:, 1]
    st, sr, sc = np.radians(sig_tilt_deg), np.radians(sig_roll_deg), np.radians(sig_cont_deg)
    a_t, o_t, n_t = R_t[:, 0], R_t[:, 1], R_t[:, 2]

    def residual(q):
        R, p, J = kin.grip_jac(q, jaw)
        e_r = rot_err(R_t, R)
        if project_lost:
            _, _, lost = rotation_mobility(J)
            e_r = e_r - lost * (lost @ e_r)
        r = [(p_t - p) / sig_p, [(a_t @ e_r) / sr, (o_t @ e_r) / st, (n_t @ e_r) / st]]
        if q_ref is not None:
            r.append((q - q_ref) / sc)
        return np.concatenate([np.ravel(x) for x in r])

    sol = least_squares(residual, np.clip(np.asarray(q0, float), lo + 1e-6, hi - 1e-6), bounds=(lo, hi), xtol=1e-9, ftol=1e-9,
                        max_nfev=max_nfev)
    return evaluate(kin, R_t, p_t, jaw, sol.x)


def _cost(r: FrameEval, sig_p, st, sr):
    return (r.pos_err_m / sig_p) ** 2 + (r.tilt_err_deg / st) ** 2 + (r.roll_err_deg / sr) ** 2


@dataclass
class Solution:
    solver: str
    q: np.ndarray               # (n,5) rad
    jaw: np.ndarray             # (n,) rad
    pos_err_mm: np.ndarray
    tilt_err_deg: np.ndarray
    roll_err_deg: np.ndarray
    lost_deg: np.ndarray
    limit_margin_deg: np.ndarray
    t: np.ndarray
    phase: np.ndarray
    hold_id: np.ndarray
    info: dict

    @property
    def n(self):
        return len(self.t)


def solve_first(kin: ArmKinematics, tg: Targets, i: int, project_lost: bool = True) -> FrameEval:
    best = None
    for s in np.radians(SEEDS_DEG):
        r = solve_frame(kin, tg.R_target(i), tg.p_b[i], tg.jaw_rad[i], s, None, tg.sigma_p_m, tg.sig_tilt_deg[i], tg.sig_roll_deg[i],
                        tg.continuity_deg, project_lost)
        c = _cost(r, tg.sigma_p_m, tg.sig_tilt_deg[i], tg.sig_roll_deg[i]) - 0.02 * r.limit_margin_deg
        if best is None or c < best[0]:
            best = (c, r)
    return best[1]


def solve_wls(kin: ArmKinematics, tg: Targets, q_init=None, project_lost: bool = True, progress=None) -> Solution:
    n = tg.n
    rs = []
    q_prev = None if q_init is None else np.asarray(q_init, float)
    for i in range(n):
        if q_prev is None:
            r = solve_first(kin, tg, i, project_lost)
        else:
            r = solve_frame(kin, tg.R_target(i), tg.p_b[i], tg.jaw_rad[i], q_prev, q_prev, tg.sigma_p_m, tg.sig_tilt_deg[i],
                            tg.sig_roll_deg[i], tg.continuity_deg, project_lost)
        rs.append(r)
        q_prev = r.q
        if progress and i % 200 == 0:
            progress(i, n)
    return Solution("wls", np.array([r.q for r in rs]), tg.jaw_rad.copy(), np.array([r.pos_err_m * 1000 for r in rs]),
                    np.array([r.tilt_err_deg for r in rs]), np.array([r.roll_err_deg for r in rs]), np.array([r.lost_deg for r in rs]),
                    np.array([r.limit_margin_deg for r in rs]), tg.t.copy(), tg.phase.copy(), tg.hold_id.copy(), {})


def free_roll_shift(kin: ArmKinematics, tg: Targets, sol: Solution) -> np.ndarray:
    """For a round object the roll at the moment of the grasp is free: read the roll the arm settled on at each contact (from a
    solve made with the roll tolerance opened) and turn the target by that constant over the hold, so that only roll CHANGES
    after the contact are followed.  Faded in over the approach, out over the release.  Returns the per-frame shift [rad]."""
    from .demo import PHASE
    shift = np.zeros(tg.n)
    for k in sorted(int(h) for h in np.unique(tg.hold_id) if h >= 0):
        frames = np.where(tg.hold_id == k)[0]
        holds = frames[tg.phase[frames] == PHASE["hold"]]
        if not len(holds):
            continue
        c = int(holds[0])
        R_t = tg.R_target(c)
        R_a, _ = kin.grip(sol.q[c], tg.jaw_rad[c])
        o_t, o_a = R_t[:, 1], R_a[:, 1] - R_t[:, 0] * (R_a[:, 1] @ R_t[:, 0])
        delta = float(np.arctan2(np.cross(o_t, o_a) @ R_t[:, 0], o_t @ o_a))
        delta = (delta + np.pi / 2) % np.pi - np.pi / 2
        pre = frames[frames < c]
        post = frames[tg.phase[frames] == PHASE["release"]]
        shift[holds] = delta
        if len(pre):
            shift[pre] = delta * np.linspace(0.0, 1.0, len(pre) + 1)[1:]
        if len(post):
            shift[post] = delta * np.linspace(1.0, 0.0, len(post) + 1)[1:]
    return shift
