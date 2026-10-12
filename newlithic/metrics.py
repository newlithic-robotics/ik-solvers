"""How good is a joint trajectory for the task?  One function, every solver, the same numbers."""
from __future__ import annotations

import numpy as np

from .demo import PHASE, PHASE_NAME
from .ik_wls import Solution, evaluate
from .kin import ArmKinematics
from .model import JOINTS
from .targets import Targets


def evaluate_solution(kin: ArmKinematics, tg: Targets, q: np.ndarray, solver: str, info: dict | None = None) -> Solution:
    """Errors of a joint trajectory (n,5 rad) against the targets, computed the same way for every solver."""
    rs = [evaluate(kin, tg.R_target(i), tg.p_b[i], tg.jaw_rad[i], q[i]) for i in range(tg.n)]
    return Solution(solver, np.asarray(q, float), tg.jaw_rad.copy(), np.array([r.pos_err_m * 1000 for r in rs]),
                    np.array([r.tilt_err_deg for r in rs]), np.array([r.roll_err_deg for r in rs]), np.array([r.lost_deg for r in rs]),
                    np.array([r.limit_margin_deg for r in rs]), tg.t.copy(), tg.phase.copy(), tg.hold_id.copy(), dict(info or {}))


def joint_speeds_deg_s(sol: Solution) -> np.ndarray:
    dq = np.diff(np.degrees(sol.q), axis=0)
    dt = np.diff(sol.t)[:, None]
    return np.abs(dq / dt)


def summary(sol: Solution, torque: np.ndarray | None = None, stall_Nm: float | None = None, collisions: np.ndarray | None = None) -> dict:
    out = {"solver": sol.solver, "frames": int(sol.n), "phases": {}}
    for code, name in PHASE_NAME.items():
        m = sol.phase == code
        if not m.any():
            continue
        out["phases"][name] = dict(frames=int(m.sum()), pos_mm_median=float(np.median(sol.pos_err_mm[m])), pos_mm_p95=float(np.percentile(sol.pos_err_mm[m], 95)),
                                   pos_mm_max=float(sol.pos_err_mm[m].max()), tilt_deg_median=float(np.median(sol.tilt_err_deg[m])),
                                   tilt_deg_p95=float(np.percentile(sol.tilt_err_deg[m], 95)), roll_deg_median=float(np.median(sol.roll_err_deg[m])),
                                   roll_deg_p95=float(np.percentile(sol.roll_err_deg[m], 95)), lost_deg_median=float(np.median(sol.lost_deg[m])),
                                   limit_margin_min_deg=float(sol.limit_margin_deg[m].min()))
    task = sol.phase != PHASE["transport"]
    out["task_pos_mm_median"] = float(np.median(sol.pos_err_mm[task])) if task.any() else None
    out["task_tilt_deg_median"] = float(np.median(sol.tilt_err_deg[task])) if task.any() else None
    out["task_tilt_deg_p95"] = float(np.percentile(sol.tilt_err_deg[task], 95)) if task.any() else None
    sp = joint_speeds_deg_s(sol)
    out["joint_speed_deg_s"] = {n: dict(p95=float(np.percentile(sp[:, k], 95)), max=float(sp[:, k].max())) for k, n in enumerate(JOINTS)}
    acc = np.abs(np.diff(sp, axis=0)) / np.diff(sol.t)[1:, None]
    out["joint_accel_deg_s2_p95"] = float(np.percentile(acc, 95))
    out["smoothness_deg"] = float(np.mean(np.abs(np.diff(np.degrees(sol.q), n=2, axis=0))))   # mean second difference per frame
    if torque is not None:
        out["torque_Nm"] = {n: dict(p95=float(np.percentile(np.abs(torque[:, k]), 95)), max=float(np.abs(torque[:, k]).max())) for k, n in enumerate(JOINTS)}
        if stall_Nm:
            out["torque_ratio_max"] = float(np.abs(torque).max() / stall_Nm)
    if collisions is not None:
        out["collision_frames"] = int(np.sum(collisions > 0))
        out["collision_max_penetration_mm"] = float(np.max(collisions)) if len(collisions) else 0.0
    out.update({k: v for k, v in sol.info.items() if isinstance(v, (int, float, str))})
    return out


def format_summary(s: dict) -> str:
    lines = [f"{s['solver']}: {s['frames']} frames"]
    for name, p in s["phases"].items():
        lines.append(f"  {name:9s} {p['frames']:5d} | pos {p['pos_mm_median']:5.2f} med {p['pos_mm_p95']:6.2f} p95 {p['pos_mm_max']:6.1f} max mm | "
                     f"tilt {p['tilt_deg_median']:5.1f} med {p['tilt_deg_p95']:5.1f} p95 | roll {p['roll_deg_median']:5.1f} med {p['roll_deg_p95']:5.1f} p95 | "
                     f"lost {p['lost_deg_median']:4.1f} | limit margin {p['limit_margin_min_deg']:5.1f} deg")
    sp = s["joint_speed_deg_s"]
    lines.append("  joint speed p95/max deg/s: " + ", ".join(f"{n.split('_')[0]}{n.split('_')[1][0]} {v['p95']:.0f}/{v['max']:.0f}" for n, v in sp.items())
                 + f" | accel p95 {s['joint_accel_deg_s2_p95']:.0f} deg/s2 | smoothness {s['smoothness_deg']:.3f} deg")
    if "torque_Nm" in s:
        lines.append("  gravity torque p95/max N m: " + ", ".join(f"{n.split('_')[0]}{n.split('_')[1][0]} {v['p95']:.2f}/{v['max']:.2f}" for n, v in s["torque_Nm"].items())
                     + (f" | max/stall {s['torque_ratio_max']:.2f}" if "torque_ratio_max" in s else ""))
    if "collision_frames" in s:
        lines.append(f"  collision frames {s['collision_frames']} (max penetration {s['collision_max_penetration_mm']:.1f} mm)")
    extra = {k: v for k, v in s.items() if k not in ("solver", "frames", "phases", "joint_speed_deg_s", "joint_accel_deg_s2_p95", "smoothness_deg", "torque_Nm", "torque_ratio_max", "collision_frames", "collision_max_penetration_mm", "task_pos_mm_median", "task_tilt_deg_median", "task_tilt_deg_p95")}
    if extra:
        lines.append("  " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in extra.items()))
    return "\n".join(lines)
