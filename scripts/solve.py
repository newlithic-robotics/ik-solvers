"""Retarget one hand of the demonstration onto the arm with one solver.

    .venv/Scripts/python.exe scripts/solve.py --hand left --placement=-34,-107,101 --solver wls --out out/solve

--placement x_mm,y_mm,yaw_deg of the arm's base in the demonstration's world.  The demonstration's reference surface (the
plate top) is put at the arm's base bottom + plate thickness unless --surface-z-mm says otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newlithic.cad import load_rig                                   # noqa: E402
from newlithic.demo import Placement, describe, load_demo            # noqa: E402
from newlithic.ik_wls import free_roll_shift, solve_wls              # noqa: E402
from newlithic.io import write_solution                              # noqa: E402
from newlithic.kin import ArmKinematics                              # noqa: E402
from newlithic.metrics import evaluate_solution, format_summary, summary   # noqa: E402
from newlithic.model import ArmConfig, compile_arm, STS3215, TABLE_Z_BASE, PLATE_THICKNESS   # noqa: E402
from newlithic.targets import Tolerances, make_targets               # noqa: E402
from newlithic.gravity import gravity_torques                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hand", default="left")
ap.add_argument("--placement", default="-34,-107,101")
ap.add_argument("--surface-z-mm", type=float, default=None, help="reference surface (plate top) height in the base frame (default: measured table -17.55 + 10 mm plate)")
ap.add_argument("--solver", default="wls", choices=["wls", "qp", "traj", "dls"])
ap.add_argument("--roll-symmetric", action="store_true", default=True, help="the held object is round: the roll at the grasp is free")
ap.add_argument("--no-roll-symmetric", dest="roll_symmetric", action="store_false")
ap.add_argument("--flip", action="store_true", help="jaws swapped (half turn about the fingers)")
ap.add_argument("--out", default="out/solve")
ap.add_argument("--name", default=None)
args = ap.parse_args()

import mujoco  # noqa: E402

t0 = time.time()
model, spec, cad, parts = compile_arm(ArmConfig(), save=False)
kin = ArmKinematics(model, cad=cad)
demo = load_demo()
print(describe(demo))
x, y, yaw = [float(v) for v in args.placement.split(",")]
surface = args.surface_z_mm / 1000 if args.surface_z_mm is not None else TABLE_Z_BASE + PLATE_THICKNESS
pl = Placement(x / 1000, y / 1000, yaw, surface)
print(f"placement: base at world ({x:.0f}, {y:.0f}) mm heading {yaw:.1f} deg; plate top at z = {surface*1000:.2f} mm in the base frame")
tol = Tolerances()
tg = make_targets(demo, args.hand, pl, cad, tol, free_roll=False, flip=args.flip)
print(f"targets: {tg.n} frames, reach from the pan axis {np.round(np.percentile(np.hypot(*(tg.p_b[:, :2] - kin.pan_axis_xy).T) * 1000, [5, 50, 95]), 0)} mm, "
      f"height {np.round(np.percentile(tg.p_b[:, 2] * 1000, [5, 50, 95]), 0)} mm")

prog = lambda i, n: print(f"  frame {i}/{n}", end="\r")
if args.solver == "wls":
    if args.roll_symmetric:
        tg_free = make_targets(demo, args.hand, pl, cad, tol, free_roll=True, flip=args.flip)
        sol1 = solve_wls(kin, tg_free, progress=prog)
        tg.roll_shift = free_roll_shift(kin, tg_free, sol1)
        print(f"free-roll pass: roll shifts per hold {np.round(np.degrees([tg.roll_shift[tg.hold_id == k].max() for k in sorted(set(tg.hold_id[tg.hold_id >= 0]))]), 1)} deg")
    sol = solve_wls(kin, tg, progress=prog)
else:
    from newlithic.solvers import run_solver
    sol = run_solver(args.solver, kin, tg, demo, cad, roll_symmetric=args.roll_symmetric, progress=prog)
sol = evaluate_solution(kin, tg, sol.q, sol.solver, sol.info)
tau = gravity_torques(model, sol.q, sol.jaw)
s = summary(sol, torque=tau, stall_Nm=STS3215["12V"]["stall_Nm"])
s["elapsed_s"] = time.time() - t0
print(format_summary(s))
name = args.name or f"{demo.episode}_{args.hand}_{args.solver}"
write_solution(Path(args.out), name, sol, tg, cad, s, extra=dict(surface_z_base_m=surface))
print("wrote", Path(args.out) / name, f"({s['elapsed_s']:.0f} s)")
