"""One entry point for every single-arm solver, so scripts can switch between them by name.

  wls   per-frame weighted least squares (ik_wls)                       - the task baseline
  dls   damped least-squares differential IK, no limits/projection      - the textbook baseline that lerobot-style IK uses
  qp    differential IK as a QP with limits (and collisions when a scene is given) (ik_qp)
  traj  whole-trajectory optimisation with smoothness (ik_traj)
"""
from __future__ import annotations

import numpy as np

from .ik_wls import Solution, free_roll_shift, solve_first, solve_wls
from .kin import ArmKinematics, rot_err
from .targets import Targets, make_targets, Tolerances


def solve_dls(kin: ArmKinematics, tg: Targets, q_init=None, damping: float = 0.05, iters: int = 10, step_cap_rad: float = 0.35,
              progress=None) -> Solution:
    """Classic damped least squares on the full 6-D twist error, no joint-limit awareness beyond clipping, isotropic weights
    (position in m, rotation in rad: 1 rad ~ 1 m)."""
    q = np.asarray(q_init, float) if q_init is not None else solve_first(kin, tg, 0).q
    out = np.zeros((tg.n, 5))
    lo, hi = kin.limits[:, 0], kin.limits[:, 1]
    for i in range(tg.n):
        R_t, p_t = tg.R_target(i), tg.p_b[i]
        for _ in range(iters):
            R, p, J = kin.grip_jac(q, tg.jaw_rad[i])
            e = np.concatenate([p_t - p, rot_err(R_t, R)])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            nrm = np.linalg.norm(dq)
            if nrm > step_cap_rad:
                dq *= step_cap_rad / nrm
            q = np.clip(q + dq, lo, hi)
        out[i] = q
        if progress and i % 200 == 0:
            progress(i, tg.n)
    return Solution("dls", out, tg.jaw_rad.copy(), np.zeros(tg.n), np.zeros(tg.n), np.zeros(tg.n), np.zeros(tg.n), np.zeros(tg.n),
                    tg.t.copy(), tg.phase.copy(), tg.hold_id.copy(), {})


def run_solver(name: str, kin: ArmKinematics, tg: Targets, demo, cad, roll_symmetric: bool = True, progress=None, model=None,
               prefix: str = "", obstacle_geoms=None, **kw) -> Solution:
    """Solve with `name`; a free-roll pass (round object) is shared by all solvers so they see the same targets."""
    if roll_symmetric and tg.roll_shift is None:
        tg_free = make_targets(demo, tg.hand, tg.placement, cad, Tolerances(), free_roll=True, flip=tg.flip)
        sol1 = solve_wls(kin, tg_free, progress=progress)
        tg.roll_shift = free_roll_shift(kin, tg_free, sol1)
    if name == "wls":
        return solve_wls(kin, tg, progress=progress)
    if name == "dls":
        return solve_dls(kin, tg, progress=progress)
    if name == "qp":
        from .ik_qp import QPConfig, solve_qp
        return solve_qp(model or kin.model, kin, tg, QPConfig(**kw), prefix=prefix, obstacle_geoms=obstacle_geoms, progress=progress)
    if name == "traj":
        from .ik_traj import TrajConfig, solve_traj
        return solve_traj(kin, tg, TrajConfig(**kw), progress=progress)
    raise ValueError(name)
