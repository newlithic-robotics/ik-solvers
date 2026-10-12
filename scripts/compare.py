"""Run every single-arm solver on one hand at one placement and tabulate.

    .venv/Scripts/python.exe scripts/compare.py --hand left --placement=x,y,yaw --out out/compare
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
from newlithic.collide import Collider                                        # noqa: E402
from newlithic.demo import Placement, load_demo                               # noqa: E402
from newlithic.gravity import gravity_torques                                 # noqa: E402
from newlithic.io import write_solution                                       # noqa: E402
from newlithic.kin import ArmKinematics                                       # noqa: E402
from newlithic.metrics import evaluate_solution, format_summary, summary      # noqa: E402
from newlithic.model import ArmConfig, PLATE_THICKNESS, STS3215, TABLE_Z_BASE, compile_arm   # noqa: E402
from newlithic.scene import SceneConfig, build_scene                          # noqa: E402
from newlithic.solvers import run_solver                                      # noqa: E402
from newlithic.targets import Tolerances, make_targets                        # noqa: E402
from newlithic.ik_bimanual import world_targets                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hand", default="left")
ap.add_argument("--placement", required=True)
ap.add_argument("--solvers", default="dls,wls,qp,traj")
ap.add_argument("--out", default="out/compare")
args = ap.parse_args()

import mujoco  # noqa: E402

demo = load_demo()
cad = load_rig()
x, y, yaw = [float(v) for v in args.placement.split(",")]
pl = Placement(x / 1000, y / 1000, yaw, TABLE_Z_BASE + PLATE_THICKNESS)
single_model, _, _, _ = compile_arm(ArmConfig(), save=False)
kin = ArmKinematics(single_model, cad=cad)
# the scene with this one arm, for the QP's collision constraints and for collision checks of every result
scene = build_scene(demo, SceneConfig(arms={args.hand: pl}), lead_hand="left")
skin = ArmKinematics(scene.model, prefix=f"{args.hand}_", cad=cad)
col = Collider(scene.model, {f"{args.hand}_": None})
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rows = {}
tg = make_targets(demo, args.hand, pl, cad, Tolerances())
prog = lambda i, n: print(f"  frame {i}/{n}", end="\r")
for name in args.solvers.split(","):
    t0 = time.time()
    if name == "qp":
        # the QP runs in the cell model, whose frame is the demonstration world: give it world-frame targets (same roll shifts)
        tgw = world_targets(demo, args.hand, cad, Tolerances())
        tgw.roll_shift = tg.roll_shift
        sol = run_solver(name, skin, tgw, demo, cad, progress=prog, model=scene.model, prefix=f"{args.hand}_", obstacle_geoms=scene.obstacle_geoms)
    else:
        sol = run_solver(name, kin, tg, demo, cad, progress=prog)
    el = time.time() - t0
    sol = evaluate_solution(kin, tg, sol.q, sol.solver, sol.info)
    rep = col.check({f"{args.hand}_": sol.q}, {f"{args.hand}_": sol.jaw})
    tau = gravity_torques(single_model, sol.q, sol.jaw)
    s = summary(sol, torque=tau, stall_Nm=STS3215["12V"]["stall_Nm"], collisions=rep.penetration_mm)
    s["object_contact_frames"] = int((rep.object_mm > 0).sum())
    s["elapsed_s"] = el
    print(f"\n=== {name} ({el:.0f} s)")
    print(format_summary(s))
    print(rep.summary())
    write_solution(out, f"{demo.episode}_{args.hand}_{name}", sol, tg, cad, s)
    rows[name] = s
(out / f"{args.hand}_comparison.json").write_text(json.dumps(rows, indent=1, default=float))
print("\n%-8s %8s %8s %8s %8s %8s %8s %8s %8s %8s" % ("solver", "pos_med", "pos_p95", "tilt_med", "tilt_p95", "roll_med", "speed95", "smooth", "collide", "time_s"))
for name, s in rows.items():
    task = [p for k, p in s["phases"].items() if k != "transport"]
    print("%-8s %8.2f %8.2f %8.1f %8.1f %8.1f %8.0f %8.3f %8d %8.0f" % (
        name, s["task_pos_mm_median"], max(p["pos_mm_p95"] for p in task), s["task_tilt_deg_median"], s["task_tilt_deg_p95"],
        np.median([p["roll_deg_median"] for p in task]), max(v["p95"] for v in s["joint_speed_deg_s"].values()), s["smoothness_deg"],
        s.get("collision_frames", -1), s["elapsed_s"]))
