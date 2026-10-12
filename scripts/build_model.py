"""Build the SO-ARM101 + Newlithic gripper model, print what it weighs, render it.

    .venv/Scripts/python.exe scripts/build_model.py [--jaw-gap-mm 22.4] [--pose 0,-90,90,60,0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newlithic.cad import describe_parts                                   # noqa: E402
from newlithic.model import ArmConfig, compile_arm, set_arm, base_bottom_z, mass_summary, gravity_torque, contact_pairs, JOINTS   # noqa: E402
from newlithic.render import render_views                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--jaw-gap-mm", type=float, default=22.4)
ap.add_argument("--pose", default="0,-60,60,60,0", help="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll [deg]")
ap.add_argument("--out", default="out/model")
ap.add_argument("--voltage", default="12V")
args = ap.parse_args()

import mujoco  # noqa: E402

model, spec, cad, parts = compile_arm(ArmConfig(voltage=args.voltage))
data = mujoco.MjData(model)
print("parts:")
print(describe_parts(parts))
print("bodies:")
print(mass_summary(model))
print("base bottom (table contact) at z =", round(base_bottom_z(model, data) * 1000, 2), "mm in base_link")
print("joints:", [(model.joint(i).name, np.round(np.degrees(model.jnt_range[i]), 1).tolist()) for i in range(model.njnt)])
print("actuators:", [(model.actuator(i).name, np.round(model.actuator_forcerange[i], 2).tolist()) for i in range(model.nu)])
print("grip triad in g (cols approach/opening/normal):\n", np.round(cad.grip_triad_g(), 4))
print("grip origin closed [mm]", np.round(cad.grip_point_g_m(0) * 1000, 2), " tips", np.round(cad.tips_g_m() * 1000, 2),
      " camera", np.round(cad.camera_g_m()[0] * 1000, 2))
q = [float(x) for x in args.pose.split(",")]
jaw = float(cad.q_from_gap(args.jaw_gap_mm))
set_arm(model, data, q, jaw)
mujoco.mj_forward(model, data)
sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
pL, pR = data.site_xpos[sid("pad_L")], data.site_xpos[sid("pad_R")]
print(f"jaw {jaw:.2f} deg: pad gap in the model {np.linalg.norm(pL-pR)*1000:.2f} mm (asked {args.jaw_gap_mm}); grip point "
      f"{np.round(0.5*(pL+pR)*1000, 1)} mm, flange {np.round(data.site_xpos[sid('flange')]*1000, 1)}, camera {np.round(data.site_xpos[sid('camera')]*1000, 1)}")
print("bodies:", [model.body(i).name for i in range(model.nbody)])
print("gravity torque per joint [N m]:", {k: round(v, 3) for k, v in gravity_torque(model, data).items()})
print("contacts at this pose:", contact_pairs(model, data))
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
render_views(model, data, out, "arm", lookat=data.site_xpos[sid("flange")], distance=0.45)
print("renders in", out)
