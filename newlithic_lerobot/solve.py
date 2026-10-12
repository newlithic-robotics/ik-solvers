"""Stage 1: the demonstration -> joint trajectories for both arms (a retargeted episode).

What `docs/findings.md` recommends for a recorded dataset, wired as one call:

  1. bimanual QP   both arms in one QP per frame in the cell model; the right gripper is asked for its pose RELATIVE to the
                   left while the left holds the screw in the air; joint / velocity limits and arm-arm / arm-table collisions
                   as hard constraints; the roll at round-object grasps left free.
  2. traj          per arm, the whole-trajectory least squares with speed and acceleration residuals, initialised from (1).
                   Its targets are the grip poses the QP solution actually reaches (not the demonstration), so the arm-arm
                   coupling found in (1) survives the smoothing; the per-phase orientation tolerances still come from the
                   demonstration's phases.
  3. evaluation    residuals against the demonstrated targets (the numbers the findings report), gravity torque, collisions.

Needs the `newlithic` package of this repository (MuJoCo, mink, qpsolvers: `pip install -e .[solve]`).  Everything downstream
(export) is independent of it.
"""
from __future__ import annotations

import copy
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from .episode import HandTrajectory, RetargetedEpisode, pose7

DEFAULT_TASK = "Unscrew the M5 nut from the bolt held by the other hand"
REPO = Path(__file__).resolve().parents[1]


@dataclass
class QPConfig:
    iterations: int = 3                  # QP sub-steps per frame (the velocity cap is shared over them)
    rel_orient_deg: float = 4.0          # sigma of the right-relative-to-left orientation task


@dataclass
class SmoothConfig:
    enabled: bool = True
    v_max_deg_s: float = 90.0
    a_max_deg_s2: float = 600.0
    max_nfev: int = 40


def _git_rev(root: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _fk(kin, q: np.ndarray, jaw: np.ndarray):
    R, p = zip(*(kin.grip(q[i], jaw[i]) for i in range(len(q))))
    return np.stack(R), np.stack(p)


def slice_demo(demo, a: int, b: int):
    """The demonstration restricted to frames a:b (every hand), for smoke tests and partial solves."""
    demo = copy.deepcopy(demo)
    for h in demo.hands.values():
        n = h.n
        a0, b0 = max(0, a), min(n, b)
        if b0 - a0 < 2:
            raise ValueError(f"frames {a}:{b} leave fewer than two of {n} frames")
        for k in ("t", "R", "p", "gap", "phase", "hold_id", "valid", "height", "tips_height"):
            setattr(h, k, getattr(h, k)[a0:b0])
        # the grasp events index frames (the scene places the screws at the picks): shift them into the slice and clamp
        for ev in h.events:
            for key in ("cross", "contact", "release", "open"):
                if key in ev:
                    ev[key] = int(min(max(int(ev[key]) - a0, 0), b0 - a0 - 1))
    return demo


def solve_episode(demo_path: Path, placements_mm: dict, qp: Optional[QPConfig] = None, smooth: Optional[SmoothConfig] = None,
                  collisions: bool = True, task: str = DEFAULT_TASK, frames: Optional[tuple] = None, log=print) -> RetargetedEpisode:
    """placements_mm: hand -> (x_mm, y_mm, yaw_deg) of the arm base in the demonstration's world (as `scripts/place.py` prints)."""
    qp = qp or QPConfig()
    smooth = smooth or SmoothConfig()
    from newlithic.cad import load_rig
    from newlithic.collide import Collider
    from newlithic.demo import Placement, load_demo
    from newlithic.gravity import gravity_torques
    from newlithic.ik_bimanual import BimanualConfig, solve_bimanual
    from newlithic.ik_traj import TrajConfig, solve_traj
    from newlithic.kin import ArmKinematics
    from newlithic.metrics import evaluate_solution, summary
    from newlithic.model import ArmConfig, PLATE_THICKNESS, STS3215, TABLE_Z_BASE, compile_arm
    from newlithic.scene import SceneConfig, build_scene
    from newlithic.targets import Tolerances, make_targets

    t0 = time.time()
    demo = load_demo(Path(demo_path))
    if frames is not None:
        demo = slice_demo(demo, *frames)
    cad = load_rig()
    surface = TABLE_Z_BASE + PLATE_THICKNESS
    arms = {}
    for name, (x, y, yaw) in placements_mm.items():
        if name not in demo.hands:
            raise ValueError(f"the demonstration has no hand {name!r} (hands: {list(demo.hands)})")
        arms[name] = Placement(float(x) / 1000.0, float(y) / 1000.0, float(yaw), surface)
    log(f"{demo.episode}: {demo.hand(next(iter(arms))).n} frames at {demo.fps:.2f} fps; arms {list(arms)}")
    scene = build_scene(demo, SceneConfig(arms=arms))
    kins = {n: ArmKinematics(scene.model, prefix=f"{n}_", cad=cad) for n in arms}
    single_model, _, _, _ = compile_arm(ArmConfig(), save=False)
    single_kins = {n: ArmKinematics(single_model, cad=cad) for n in arms}

    # ---- 1. bimanual QP
    cfg = BimanualConfig(iterations=qp.iterations, relative_orientation_cost=1.0 / np.radians(qp.rel_orient_deg))
    prog = lambda i, n: log(f"  QP frame {i}/{n}")
    sols, tgs, rel = solve_bimanual(scene, demo, cad, kins, cfg, single_kins=single_kins, progress=prog)
    t_qp = time.time() - t0
    log(f"bimanual QP done in {t_qp:.0f} s, {int(rel.sum())} of {len(rel)} frames solved relative")

    hands, summaries, q_final, jaw_final = {}, {}, {}, {}
    for n in arms:
        tg = make_targets(demo, n, arms[n], cad, Tolerances())
        tg.roll_shift = tgs[n].roll_shift
        q, jaw = sols[n].q, tg.jaw_rad
        stages = ["bimanual_qp"]
        # ---- 2. smoothing toward the QP's own poses (keeps the coupling), per-phase tolerances from the demonstration
        if smooth.enabled:
            t1 = time.time()
            R_fk, p_fk = _fk(single_kins[n], q, jaw)
            tg_s = replace(tg, R_b=R_fk, p_b=p_fk, roll_shift=None, flip=False)
            tcfg = TrajConfig(v_max_deg_s=smooth.v_max_deg_s, a_max_deg_s2=smooth.a_max_deg_s2, max_nfev=smooth.max_nfev)
            res = solve_traj(single_kins[n], tg_s, tcfg, q_init=q)
            q = res.q
            stages.append("traj")
            log(f"{n}: trajectory optimiser done in {time.time() - t1:.0f} s ({res.info})")
        q_final[n], jaw_final[n] = q, jaw
        # ---- 3. evaluation against the demonstration
        sol = evaluate_solution(single_kins[n], tg, q, "+".join(stages), dict(relative_frames=int(rel.sum())))
        tau = gravity_torques(single_model, sol.q, sol.jaw)
        summaries[n] = (sol, tau, tg)

    pen = None
    if collisions:
        col = Collider(scene.model, {f"{n}_": None for n in arms})
        rep = col.check({f"{n}_": q_final[n] for n in arms}, {f"{n}_": jaw_final[n] for n in arms})
        pen = rep.penetration_mm
        log(rep.summary())

    solver_meta = dict(stages=["bimanual_qp"] + (["traj"] if smooth.enabled else []), qp=asdict(qp), smooth=asdict(smooth),
                       relative_frames=int(rel.sum()), qp_elapsed_s=t_qp, frames=list(frames) if frames else None, summaries={})
    for n in arms:
        sol, tau, tg = summaries[n]
        s = summary(sol, torque=tau, stall_Nm=STS3215["12V"]["stall_Nm"], collisions=pen)
        solver_meta["summaries"][n] = s
        R_fk, p_fk = _fk(single_kins[n], sol.q, sol.jaw)
        hands[n] = HandTrajectory(q_deg=np.degrees(sol.q), jaw_deg=np.degrees(sol.jaw), gap_mm=tg.gap_m * 1000.0,
                                  phase=tg.phase.astype(np.int64), hold_id=tg.hold_id.astype(np.int64),
                                  pose_demo=pose7(tg.R_b, tg.p_b), pose_fk=pose7(R_fk, p_fk),
                                  pos_err_mm=sol.pos_err_mm, tilt_err_deg=sol.tilt_err_deg, roll_err_deg=sol.roll_err_deg,
                                  lost_deg=sol.lost_deg)
        task_m = sol.phase != 0
        if task_m.any():
            log(f"{n}: task position error median {np.median(sol.pos_err_mm[task_m]):.2f} mm, tilt {np.median(sol.tilt_err_deg[task_m]):.1f} deg, "
                f"speed p95 {max(v['p95'] for v in s['joint_speed_deg_s'].values()):.0f} deg/s, accel p95 {s['joint_accel_deg_s2_p95']:.0f} deg/s2, "
                f"smoothness {s['smoothness_deg']:.3f} deg/frame")
    solver_meta["elapsed_s"] = time.time() - t0
    gripper = dict(open_stop_deg=float(cad.open_stop_deg), gap_mm_at_0=float(cad.gap_mm(0.0)),
                   gap_mm_at_stop=float(cad.gap_mm(cad.open_stop_deg)),
                   law="pad gap between the contact-strip centres from the CAD pivots and the strip polar law (newlithic.cad)")
    t_ref = next(iter(summaries.values()))[2].t
    ep = RetargetedEpisode(episode=demo.episode, fps=float(demo.fps), t=np.asarray(t_ref, float), hands=hands,
                           placements={n: arms[n].as_dict() for n in arms}, gripper=gripper, solver=solver_meta,
                           provenance=dict(demo=str(Path(demo_path).resolve()), demo_notes=demo.notes, repo_git=_git_rev(REPO),
                                           newlithic_lerobot=__version__, created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                           surface_z_base_m=surface),
                           task=task)
    return ep
