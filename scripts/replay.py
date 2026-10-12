"""Replay solved trajectories in the simulated cell and record a video.

    .venv/Scripts/python.exe scripts/replay.py --left-traj out/bimanual/Episode-1_left_bimanual.npz --right-traj out/bimanual/Episode-1_right_bimanual.npz \
        --mode physics --out out/replay/bimanual.mp4

Placements are read from the trajectories' JSON side files.  --mode kinematic poses the arms at the commands (what the
solver produced); --mode physics drives the position servos (torque-limited) with the screws as free bodies.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newlithic.demo import PHASE_NAME, Placement, load_demo                   # noqa: E402
from newlithic.model import JOINTS                                            # noqa: E402
from newlithic.replay import GraspEmulator, Replay, Trajectory               # noqa: E402
from newlithic.scene import SceneConfig, build_scene, save_scene              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--left-traj", default=None)
ap.add_argument("--right-traj", default=None)
ap.add_argument("--mode", default="kinematic", choices=["kinematic", "physics"])
ap.add_argument("--out", default="out/replay/replay.mp4")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--every", type=int, default=1, help="kinematic: render every k-th frame")
ap.add_argument("--x3", action="store_true", help="also render each gripper's X3 finger-side camera")
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--squeeze-mm", type=float, default=4.0, help="physics: close the jaws this much past the demonstrated gap (a position servo only grips past contact)")
ap.add_argument("--t0", type=float, default=None)
ap.add_argument("--t1", type=float, default=None)
ap.add_argument("--close", default=None, help="arm name: add a close-up camera following its grip point")
ap.add_argument("--no-gravity-comp", action="store_true", help="physics: no gravity feed-forward on the servo commands")
ap.add_argument("--kp", type=float, default=None, help="physics: servo stiffness N m/rad (default ArmConfig.kp)")
ap.add_argument("--grasp", default="weld", choices=["weld", "contact"], help="physics: emulate grasps with a weld while the demonstration holds (default) or rely on pad contact only")
args = ap.parse_args()

import cv2  # noqa: E402
import mujoco  # noqa: E402

demo = load_demo()
trajs, arms = {}, {}
for name, f in (("left", args.left_traj), ("right", args.right_traj)):
    if not f:
        continue
    meta = json.loads(Path(f).with_suffix(".json").read_text())
    pl = meta["placement"]
    arms[name] = Placement(pl["x_m"], pl["y_m"], pl["yaw_deg"], pl["surface_z_base"])
    trajs[name] = Trajectory.load(f)
from newlithic.model import ArmConfig  # noqa: E402
arm_cfg = ArmConfig(kp=args.kp) if args.kp else ArmConfig()
scene = build_scene(demo, SceneConfig(arms=arms, physics_objects=(args.mode == "physics"), tray=True, arm_cfg=arm_cfg))
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
save_scene(scene, out.with_suffix(".xml"))
rp = Replay(scene, trajs, fps=args.fps, width=args.width, height=args.height)
if args.t0 is not None:
    rp.t0 = max(rp.t0, args.t0)
if args.t1 is not None:
    rp.t1 = min(rp.t1, args.t1)
centre = np.mean([scene.names[f"{n}_T_world_base"][:3, 3] for n in arms], axis=0)
look = 0.5 * (centre + (scene.plate_T[:3, 3] if scene.plate_T is not None else centre)) + [0, 0, 0.08]
cams = [rp.camera(look, 0.85, 150, -28), rp.camera(look, 0.55, 60, -20)]
if args.close:
    sid = lambda n: mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, n)
    pads = (sid(f"{args.close}_pad_L"), sid(f"{args.close}_pad_R"))

    class FollowCam(mujoco.MjvCamera):
        pass
    close_cam = rp.camera(look, 0.16, 120, -15)
    cams = [cams[0], close_cam]
    _orig_run_k, _orig_run_p = rp.run_kinematic, rp.run_physics

    def _follow(t, d):
        close_cam.lookat[:] = 0.5 * (d.site_xpos[pads[0]] + d.site_xpos[pads[1]])
    rp_on_frame = _follow
else:
    rp_on_frame = None
phase_of = {n: demo.hand(n).phase for n in arms}
t_of = {n: demo.hand(n).t for n in arms}


def hud(img, t):
    img = np.ascontiguousarray(img)
    txt = f"t = {t:5.2f} s  ({args.mode})"
    for n in arms:
        i = int(np.clip(np.searchsorted(t_of[n], t), 0, len(t_of[n]) - 1))
        txt += f"   {n}: {PHASE_NAME[int(phase_of[n][i])]}"
    cv2.putText(img, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


t0 = time.time()
if args.mode == "kinematic":
    rp.run_kinematic(out, cams, every=args.every, x3=args.x3, hud=hud, on_frame=rp_on_frame)
else:
    jaw_sq = float(np.radians(scene_cad_q := 0.0))
    from newlithic.cad import load_rig
    cad = load_rig()
    jaw_sq = float(np.radians(cad.q_from_gap(9.5) - cad.q_from_gap(9.5 - args.squeeze_mm)))
    ge = GraspEmulator(scene, demo) if args.grasp == "weld" else None
    from newlithic.demo import PHASE

    def squeeze_fn(hand, t):            # squeeze only while the demonstration holds (hold / release phases)
        h = demo.hand(hand)
        i = int(np.clip(np.searchsorted(h.t, t), 0, h.n - 1))
        return int(h.phase[i]) in (PHASE["hold"], PHASE["release"])
    log = rp.run_physics(out, cams, x3=args.x3, hud=hud, jaw_squeeze_rad=jaw_sq, on_frame=rp_on_frame, gravity_comp=not args.no_gravity_comp, grasp=ge,
                         squeeze_fn=squeeze_fn)
    if ge:
        print("grasp events:", ge.log)
    np.savez_compressed(out.with_suffix(".npz"), **log)
    for n in arms:
        err = np.degrees(log[f"{n}_cmd"][:, :5] - log[f"{n}_meas"][:, :5])
        print(f"{n}: tracking error deg (mean |e| per joint) {np.round(np.abs(err).mean(0), 2)}, max {np.round(np.abs(err).max(0), 1)}; "
              f"actuator force p95 N m {np.round(np.percentile(np.abs(log[f'{n}_force']), 95, axis=0), 2)}")
    if len(log["bolts"]) and log["bolts"].shape[1]:
        z0, z1 = log["bolts"][0, :, 2], log["bolts"][-1, :, 2]
        print("bolt heights start -> end [mm]:", np.round(z0 * 1000, 1), "->", np.round(z1 * 1000, 1))
print("wrote", out, f"({time.time()-t0:.0f} s)")
