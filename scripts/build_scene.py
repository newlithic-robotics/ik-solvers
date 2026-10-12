"""Build the two-arm cell at given placements and render it (arms at a rest pose, or at the joints of solved CSVs).

    .venv/Scripts/python.exe scripts/build_scene.py --left=-34,-107,101 --right=291.5,411.9,236.2 --out out/scene
    .venv/Scripts/python.exe scripts/build_scene.py --left ... --right ... --left-csv out/solve/Episode-1_left_wls.npz --at 13.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newlithic.demo import Placement, load_demo                         # noqa: E402
from newlithic.model import PLATE_THICKNESS, TABLE_Z_BASE, set_arm        # noqa: E402
from newlithic.render import render_views                                # noqa: E402
from newlithic.scene import SceneConfig, build_scene, save_scene          # noqa: E402
from newlithic.collide import Collider                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--left", default="-34,-107,101")
ap.add_argument("--right", default=None)
ap.add_argument("--left-csv", default=None, help=".npz of a solved left trajectory")
ap.add_argument("--right-csv", default=None)
ap.add_argument("--at", type=float, default=None, help="time [s] to pose the arms at")
ap.add_argument("--physics", action="store_true")
ap.add_argument("--out", default="out/scene")
ap.add_argument("--close", default=None, help="arm name: close-up views around its grip point")
args = ap.parse_args()

import mujoco  # noqa: E402

demo = load_demo()
surface = TABLE_Z_BASE + PLATE_THICKNESS
arms = {}
for name, s in (("left", args.left), ("right", args.right)):
    if s:
        x, y, yaw = [float(v) for v in s.split(",")]
        arms[name] = Placement(x / 1000, y / 1000, yaw, surface)
scene = build_scene(demo, SceneConfig(arms=arms, physics_objects=args.physics, tray=True))
model, data = scene.model, mujoco.MjData(scene.model)
print("bodies", model.nbody, "geoms", model.ngeom, "nq", model.nq, "nu", model.nu, "| plate mismatch", scene.names.get("plate_mismatch_mm"), "mm")
save_scene(scene, Path(args.out) / "cell.xml")
rest = np.radians([0.0, -100.0, 90.0, 60.0, 0.0])
for name in arms:
    q, jaw = rest, np.radians(6.7)
    f = getattr(args, f"{name}_csv")
    if f and args.at is not None:
        z = np.load(f)
        i = int(np.argmin(np.abs(z["t"] - args.at)))
        q, jaw = z["q"][i], float(z["jaw"][i])
    set_arm(model, data, np.degrees(q), np.degrees(jaw), prefix=f"{name}_")
mujoco.mj_forward(model, data)
col = Collider(model, {f"{n}_": None for n in arms})
print("permanent overlaps ignored:", sorted(col.ignore))
rep = col.check({f"{n}_": data.qpos[[model.jnt_qposadr[col.ids[f'{n}_'][j]] for j in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')]][None] for n in arms},
                {f"{n}_": np.array([data.qpos[model.jnt_qposadr[col.ids[f'{n}_']['jaw_L']]]]) for n in arms})
print(rep.summary())
centre = np.mean([scene.names[f"{n}_T_world_base"][:3, 3] for n in arms], axis=0) if arms else np.zeros(3)
look = 0.5 * (centre + (scene.plate_T[:3, 3] if scene.plate_T is not None else centre))
render_views(model, data, Path(args.out), f"cell{'_t%.1f' % args.at if args.at is not None else ''}", lookat=look + [0, 0, 0.05], distance=0.9,
             views={"iso": (135, -30), "side": (90, -10), "top": (90, -89), "front": (0, -15)})
if args.close:
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
    g = 0.5 * (data.site_xpos[sid(f"{args.close}_pad_L")] + data.site_xpos[sid(f"{args.close}_pad_R")])
    render_views(model, data, Path(args.out), f"close_{args.close}{'_t%.1f' % args.at if args.at is not None else ''}", lookat=g, distance=0.14,
                 views={"a": (135, -20), "b": (45, -20), "c": (90, -5), "low": (0, 5)})
print("renders in", args.out)
