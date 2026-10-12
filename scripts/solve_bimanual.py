"""Solve both arms together in the cell (coupled QP) and write both trajectories.

    .venv/Scripts/python.exe scripts/solve_bimanual.py --left=x,y,yaw --right=x,y,yaw --out out/bimanual
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newlithic.cad import load_rig                                            # noqa: E402
from newlithic.demo import Placement, load_demo                               # noqa: E402
from newlithic.gravity import gravity_torques                                 # noqa: E402
from newlithic.ik_bimanual import BimanualConfig, solve_bimanual              # noqa: E402
from newlithic.io import write_solution                                       # noqa: E402
from newlithic.kin import ArmKinematics                                       # noqa: E402
from newlithic.metrics import evaluate_solution, format_summary, summary      # noqa: E402
from newlithic.model import ArmConfig, PLATE_THICKNESS, STS3215, TABLE_Z_BASE, compile_arm   # noqa: E402
from newlithic.scene import SceneConfig, build_scene, save_scene              # noqa: E402
from newlithic.collide import Collider                                        # noqa: E402
from newlithic.targets import make_targets, Tolerances                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--left", required=True)
ap.add_argument("--right", required=True)
ap.add_argument("--out", default="out/bimanual")
ap.add_argument("--iterations", type=int, default=3)
ap.add_argument("--rel-orient-deg", type=float, default=4.0)
args = ap.parse_args()

import mujoco  # noqa: E402

t0 = time.time()
demo = load_demo()
cad = load_rig()
surface = TABLE_Z_BASE + PLATE_THICKNESS
arms = {}
for name, s in (("left", args.left), ("right", args.right)):
    x, y, yaw = [float(v) for v in s.split(",")]
    arms[name] = Placement(x / 1000, y / 1000, yaw, surface)
scene = build_scene(demo, SceneConfig(arms=arms))
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
save_scene(scene, out / "cell.xml")
kins = {n: ArmKinematics(scene.model, prefix=f"{n}_", cad=cad) for n in arms}
single_model, _, _, _ = compile_arm(ArmConfig(), save=False)
single_kins = {n: ArmKinematics(single_model, cad=cad) for n in arms}
cfg = BimanualConfig(iterations=args.iterations, relative_orientation_cost=1.0 / np.radians(args.rel_orient_deg))
prog = lambda i, n: print(f"  frame {i}/{n}", end="\r")
sols, tgs, rel = solve_bimanual(scene, demo, cad, kins, cfg, single_kins=single_kins, progress=prog)
print(f"\nrelative frames: {int(rel.sum())} of {len(rel)}")
col = Collider(scene.model, {f"{n}_": None for n in arms})
rep = col.check({f"{n}_": sols[n].q for n in arms}, {f"{n}_": sols[n].jaw for n in arms})
print(rep.summary())
summaries = {}
for n in arms:
    # evaluate against the targets in the arm's own base frame (the same numbers as the single-arm solvers report)
    tg = make_targets(demo, n, arms[n], cad, Tolerances())
    tg.roll_shift = tgs[n].roll_shift
    sol = evaluate_solution(single_kins[n], tg, sols[n].q, "bimanual_qp", sols[n].info)
    tau = gravity_torques(single_model, sol.q, sol.jaw)
    s = summary(sol, torque=tau, stall_Nm=STS3215["12V"]["stall_Nm"], collisions=rep.penetration_mm)
    s["elapsed_s"] = time.time() - t0
    print(f"--- {n}")
    print(format_summary(s))
    write_solution(out, f"{demo.episode}_{n}_bimanual", sol, tg, cad, s, extra=dict(surface_z_base_m=surface, other_arm=arms["right" if n == "left" else "left"].as_dict()))
    summaries[n] = s
# the relation between the two grippers during the relative frames: how well the pair geometry is kept
R_l, p_l = [], []
for i in np.where(rel)[0]:
    RL, pL = kins["left"].grip(sols["left"].q[i], sols["left"].jaw[i])
    RR, pR = kins["right"].grip(sols["right"].q[i], sols["right"].jaw[i])
    RLt, pLt = tgs["left"].R_target(i), tgs["left"].p_b[i]
    RRt, pRt = tgs["right"].R_target(i), tgs["right"].p_b[i]
    d_a = np.linalg.norm(RL.T @ (pR - pL) - RLt.T @ (pRt - pLt)) * 1000
    ang = np.degrees(np.arccos(np.clip((np.trace((RL.T @ RR) @ (RLt.T @ RRt).T) - 1) / 2, -1, 1)))
    R_l.append(ang); p_l.append(d_a)
print(f"pair geometry during the {int(rel.sum())} relative frames: relative position error {np.median(p_l):.2f} mm median / {np.percentile(p_l, 95):.2f} p95, "
      f"relative orientation error {np.median(R_l):.1f} deg median / {np.percentile(R_l, 95):.1f} p95")
(out / "bimanual_summary.json").write_text(json.dumps(dict(left=summaries["left"], right=summaries["right"], relative_frames=int(rel.sum()),
                                                            rel_pos_mm_median=float(np.median(p_l)), rel_rot_deg_median=float(np.median(R_l)),
                                                            placements={n: a.as_dict() for n, a in arms.items()}), indent=1, default=float))
print("done", f"{time.time()-t0:.0f} s")
