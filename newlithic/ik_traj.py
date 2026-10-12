"""Solver 3 - whole-trajectory optimisation ("traj").

All frames at once: the per-frame task residuals of the WLS solver (position / sigma_p, projected orientation / sigma per
axis) PLUS residuals that couple neighbouring frames:
    (q[i+1] - q[i]) / (v_max dt)            joint speed, so the servos can follow
    (q[i+1] - 2 q[i] + q[i-1]) / (a_max dt^2)  joint acceleration, so the path is smooth
and optionally the gravity torque / tau_sigma per joint (prefer postures that load the servos less).
Solved with scipy's trust-region reflective least squares on a block-banded sparse Jacobian (finite differences grouped by
sparsity), warm-started from a per-frame solution.  This is the "offline" family: slow, global in time, and the only one
that can trade a little accuracy at one frame for feasibility of the whole motion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .kin import ArmKinematics, rot_err, rotation_mobility
from .targets import Targets


@dataclass
class TrajConfig:
    v_max_deg_s: float = 90.0          # the speed the trajectory should not need (the servo's no-load is 270 deg/s; 45 was the replay cap)
    a_max_deg_s2: float = 600.0
    torque_sigma_Nm: float = 0.0       # 0: off.  e.g. 1.5: gravity torque is a residual tau / 1.5 N m
    max_nfev: int = 40
    subsample: int = 1                 # optimise every k-th frame (then interpolate) to go faster
    project_lost: bool = True


def solve_traj(kin: ArmKinematics, tg: Targets, cfg: TrajConfig | None = None, q_init: np.ndarray | None = None, progress=None):
    from .ik_wls import Solution, solve_wls
    cfg = cfg or TrajConfig()
    if q_init is None:
        q_init = solve_wls(kin, tg, progress=progress).q
    idx = np.arange(0, tg.n, cfg.subsample)
    if idx[-1] != tg.n - 1:
        idx = np.append(idx, tg.n - 1)
    m = len(idx)
    dt = np.diff(tg.t[idx])
    sig_p = tg.sigma_p_m
    st, sr = np.radians(tg.sig_tilt_deg[idx]), np.radians(tg.sig_roll_deg[idx])
    R_t = np.stack([tg.R_target(int(i)) for i in idx])
    p_t, jaw = tg.p_b[idx], tg.jaw_rad[idx]
    sv = np.radians(cfg.v_max_deg_s) * dt                 # (m-1,)
    sa = np.radians(cfg.a_max_deg_s2) * (0.5 * (dt[1:] + dt[:-1])) ** 2
    lo, hi = np.tile(kin.limits[:, 0], m), np.tile(kin.limits[:, 1], m)
    use_tau = cfg.torque_sigma_Nm > 0
    n_task = 6 + (5 if use_tau else 0)
    import mujoco

    def residual(x):
        Q = x.reshape(m, 5)
        r = np.zeros(m * n_task + (m - 1) * 5 + (m - 2) * 5)
        for k in range(m):
            R, p, J = kin.grip_jac(Q[k], jaw[k])
            e_r = rot_err(R_t[k], R)
            if cfg.project_lost:
                _, _, lost = rotation_mobility(J)
                e_r = e_r - lost * (lost @ e_r)
            base = k * n_task
            r[base:base + 3] = (p_t[k] - p) / sig_p
            r[base + 3] = (R_t[k][:, 0] @ e_r) / sr[k]
            r[base + 4] = (R_t[k][:, 1] @ e_r) / st[k]
            r[base + 5] = (R_t[k][:, 2] @ e_r) / st[k]
            if use_tau:
                d = kin.data
                d.qvel[:] = 0.0
                mujoco.mj_forward(kin.model, d)
                r[base + 6:base + 11] = d.qfrc_bias[kin.dofadr] / cfg.torque_sigma_Nm
        o = m * n_task
        r[o:o + (m - 1) * 5] = ((Q[1:] - Q[:-1]) / sv[:, None]).ravel()
        o += (m - 1) * 5
        r[o:] = ((Q[2:] - 2 * Q[1:-1] + Q[:-2]) / sa[:, None]).ravel()
        return r

    S = lil_matrix((m * n_task + (m - 1) * 5 + (m - 2) * 5, m * 5), dtype=np.int8)
    for k in range(m):
        S[k * n_task:(k + 1) * n_task, k * 5:(k + 1) * 5] = 1
    o = m * n_task
    for k in range(m - 1):
        S[o + k * 5:o + (k + 1) * 5, k * 5:(k + 2) * 5] = 1
    o += (m - 1) * 5
    for k in range(m - 2):
        S[o + k * 5:o + (k + 1) * 5, k * 5:(k + 3) * 5] = 1
    x0 = np.clip(q_init[idx].ravel(), lo + 1e-6, hi - 1e-6)
    sol = least_squares(residual, x0, bounds=(lo, hi), jac_sparsity=S.tocsr(), method="trf", x_scale="jac", max_nfev=cfg.max_nfev,
                        verbose=0, tr_solver="lsmr")
    Q = sol.x.reshape(m, 5)
    if cfg.subsample > 1:
        q = np.stack([np.interp(tg.t, tg.t[idx], Q[:, j]) for j in range(5)], 1)
    else:
        q = Q
    return Solution("traj", q, tg.jaw_rad.copy(), np.zeros(tg.n), np.zeros(tg.n), np.zeros(tg.n), np.zeros(tg.n), np.zeros(tg.n),
                    tg.t.copy(), tg.phase.copy(), tg.hold_id.copy(), dict(nfev=int(sol.nfev), cost=float(sol.cost), status=int(sol.status)))
