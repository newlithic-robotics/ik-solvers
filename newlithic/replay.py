"""Replay solved joint trajectories in the cell: kinematically (pose the arms, check collisions, render) or with physics
(position-servo actuators with the STS3215 torque limits drive the arms through the commands; the screws are free bodies).

Videos: a free camera view and, optionally, the X3 "finger-side" camera of each gripper.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .collide import Collider
from .model import JAW_JOINTS, JOINTS, joint_ids
from .render import VideoWriter
from .scene import Scene


@dataclass
class Trajectory:
    t: np.ndarray
    q: np.ndarray          # (n,5) rad
    jaw: np.ndarray        # (n,) rad

    @staticmethod
    def load(path: Path) -> "Trajectory":
        z = np.load(Path(path))
        return Trajectory(z["t"], z["q"], z["jaw"])

    def at(self, t: float):
        i = np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1)
        u = (t - self.t[i - 1]) / max(self.t[i] - self.t[i - 1], 1e-9)
        u = float(np.clip(u, 0.0, 1.0))
        return (1 - u) * self.q[i - 1] + u * self.q[i], (1 - u) * self.jaw[i - 1] + u * self.jaw[i]


def _set(model, data, prefix, q, jaw):
    ids = joint_ids(model, prefix)
    for n, v in zip(JOINTS, q):
        data.qpos[model.jnt_qposadr[ids[n]]] = v
    data.qpos[model.jnt_qposadr[ids["jaw_L"]]] = jaw
    data.qpos[model.jnt_qposadr[ids["jaw_R"]]] = -jaw


def _ctrl(model, data, prefix, q, jaw, comp=None):
    """comp: (5,) rad added to the joint commands (gravity feed-forward)."""
    for k, (n, v) in enumerate(zip(JOINTS, q)):
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}act_{n}")] = v + (comp[k] if comp is not None else 0.0)
    data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}act_jaw")] = -jaw


class GravityComp:
    """Joint offsets that cancel the static droop of position servos: tau_gravity(q_cmd) / kp per joint, from the model's own
    masses (what a real controller does with an identified gravity model)."""

    def __init__(self, model, prefixes):
        self.model = model
        self.data = mujoco.MjData(model)
        self.ids = {p: joint_ids(model, p) for p in prefixes}
        self.kp = {p: np.array([model.actuator_gainprm[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{p}act_{n}"), 0] for n in JOINTS]) for p in prefixes}

    def __call__(self, cmds: dict) -> dict:
        d = self.data
        for p, (q, jaw) in cmds.items():
            _set(self.model, d, p, q, jaw)
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)
        out = {}
        for p in cmds:
            dof = [self.model.jnt_dofadr[self.ids[p][n]] for n in JOINTS]
            out[p] = d.qfrc_bias[dof] / self.kp[p]
        return out


class GraspEmulator:
    """Welds an object to a gripper while the demonstration holds it: engages when the object's grasp point lies within
    `capture_m` of the pad midpoint during a grasp/hold phase of that hand, releases when the hand's phase leaves the hold.
    The left hand holds bolts (by the head), the right hand nuts.  A documented stand-in for the compliant TPU pads: rigid
    convex pads in MuJoCo let a 5 mm screw head slide out during the fast whip-up, which the real pads do not."""

    def __init__(self, scene: Scene, demo, capture_m: float = 0.010):
        from .demo import PHASE
        from .scene import BOLT
        self.scene, self.model, self.demo = scene, scene.model, demo
        self.capture = capture_m
        self.PHASE = PHASE
        m = self.model
        self.welds = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, n) for k, n in scene.names.get("welds", {}).items()}
        self.objects = {"left": [f"bolt{k}" for k in range(len(scene.screw_xy))], "right": [f"nut{k}" for k in range(len(scene.screw_xy))]}
        self.bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in sum(self.objects.values(), [])}
        self.grasp_point = {n: (np.array([0, 0, BOLT["length"] + BOLT["head_h"] / 2]) if n.startswith("bolt") else np.array([0, 0, BOLT["nut_h"] / 2]))
                            for n in self.bid}
        self.width = {n: (BOLT["head_r"] * 2 if n.startswith("bolt") else BOLT["nut_af"]) for n in self.bid}   # what the pads close on
        self.held = {}
        self.log = []

    def _hand_phase(self, hand, t):
        h = self.demo.hand(hand)
        i = int(np.clip(np.searchsorted(h.t, t), 0, h.n - 1))
        return int(h.phase[i])

    def step(self, t, d):
        m = self.model
        for hand in self.objects:
            if hand not in self.scene.cfg.arms:
                continue
            phase = self._hand_phase(hand, t)
            sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
            mid = 0.5 * (d.site_xpos[sid(f"{hand}_pad_L")] + d.site_xpos[sid(f"{hand}_pad_R")])
            gap = float(np.linalg.norm(d.site_xpos[sid(f"{hand}_pad_L")] - d.site_xpos[sid(f"{hand}_pad_R")]))
            if hand in self.held:
                obj = self.held[hand]
                # released when the demonstration leaves the hold (not on the measured gap: with both grippers welded to the
                # bolt-nut pair the constraint forces can push a pad pair apart for a moment)
                if phase not in (self.PHASE["grasp"], self.PHASE["hold"], self.PHASE["release"]):
                    self.held.pop(hand)
                    d.eq_active[self.welds[(hand, obj)]] = 0
                    self.log.append((round(t, 2), hand, "release", obj))
                continue
            if phase in (self.PHASE["grasp"], self.PHASE["hold"]):
                for obj in self.objects[hand]:
                    b = self.bid[obj]
                    gp = d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.grasp_point[obj]
                    # engaged only once the jaws have actually closed on it (measured pad gap near the object's width: the
                    # head's diameter, or the nut's across-flats + the corner allowance)
                    allow = 0.0015 if obj.startswith("bolt") else 0.0025
                    if np.linalg.norm(gp - mid) < self.capture and gap < self.width[obj] + allow:
                        eq = self.welds[(hand, obj)]
                        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{hand}_gripper_link")
                        Rg = d.xmat[g].reshape(3, 3)
                        rel_p = Rg.T @ (d.xpos[b] - d.xpos[g])
                        q_gi, rel_q = np.zeros(4), np.zeros(4)
                        mujoco.mju_negQuat(q_gi, d.xquat[g])
                        mujoco.mju_mulQuat(rel_q, q_gi, d.xquat[b])
                        m.eq_data[eq, :3] = 0.0                 # anchor in body1
                        m.eq_data[eq, 3:6] = rel_p              # body2 pose relative to body1 ...
                        m.eq_data[eq, 6:10] = rel_q             # ... (pos, quat)
                        m.eq_data[eq, 10] = 1.0                 # torque scale
                        d.eq_active[eq] = 1
                        self.held[hand] = obj
                        self.log.append((round(t, 2), hand, "grasp", obj))
                        break


class Replay:
    def __init__(self, scene: Scene, trajectories: dict, fps: float = 30.0, width: int = 1280, height: int = 960):
        self.scene, self.model = scene, scene.model
        self.data = mujoco.MjData(self.model)
        self.traj = trajectories                       # "left"/"right" -> Trajectory
        self.fps, self.width, self.height = fps, width, height
        self.t0 = max(tr.t[0] for tr in trajectories.values())
        self.t1 = min(tr.t[-1] for tr in trajectories.values())

    def camera(self, lookat, distance, azimuth, elevation):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        return cam

    def pose_at(self, t: float):
        for name, tr in self.traj.items():
            q, jaw = tr.at(t)
            _set(self.model, self.data, f"{name}_", q, jaw)
        mujoco.mj_forward(self.model, self.data)

    def run_kinematic(self, out_mp4: Path, cams: list, every: int = 1, x3: bool = False, hud=None, on_frame=None):
        """Pose the arms at every frame and record.  cams: list of MjvCamera (free) to render side by side."""
        r = mujoco.Renderer(self.model, self.height, self.width)
        opt = mujoco.MjvOption()
        ts = np.arange(self.t0, self.t1, every / self.fps)
        vw = VideoWriter(out_mp4, self.fps / every, self.width * (len(cams) + (2 if x3 else 0)), self.height)
        x3_cams = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, f"{n}_x3_fingers") for n in self.traj] if x3 else []
        for t in ts:
            self.pose_at(t)
            if on_frame:
                on_frame(t, self.data)
            tiles = []
            for cam in cams:
                r.update_scene(self.data, cam, opt)
                tiles.append(r.render().copy())
            for cid in x3_cams:
                r.update_scene(self.data, cid, opt)
                tiles.append(r.render().copy())
            img = np.hstack(tiles)
            if hud:
                img = hud(img, t)
            vw.add(img)
        vw.close()
        r.close()
        return out_mp4

    def run_physics(self, out_mp4: Path | None, cams: list, log_every: int = 1, x3: bool = False, hud=None, settle_s: float = 0.5,
                    jaw_squeeze_rad: float = 0.0, on_frame=None, gravity_comp: bool = True, grasp=None, squeeze_fn=None):
        """squeeze_fn(hand, t) -> bool: when the jaws should be commanded `jaw_squeeze_rad` past the demonstrated angle (a position
        servo only grips past contact); default: always."""
        """Drive the actuators with the commanded joints; the simulation runs at the model timestep.  Returns a log dict:
        t, commanded/measured joints per arm, actuator forces, object poses."""
        m, d = self.model, self.data
        dt = m.opt.timestep
        # start: put the arms at the first command, let the objects settle
        self.pose_at(self.t0)
        gc = GravityComp(m, [f"{n}_" for n in self.traj]) if gravity_comp else None
        comp = gc({f"{n}_": tr.at(self.t0) for n, tr in self.traj.items()}) if gc else {}
        for name, tr in self.traj.items():
            q, jaw = tr.at(self.t0)
            _ctrl(m, d, f"{name}_", q, jaw, comp.get(f"{name}_"))
        mujoco.mj_forward(m, d)
        for _ in range(int(settle_s / dt)):
            mujoco.mj_step(m, d)
        r = mujoco.Renderer(m, self.height, self.width) if out_mp4 else None
        opt = mujoco.MjvOption()
        vw = VideoWriter(out_mp4, self.fps, self.width * (len(cams) + (2 if x3 else 0)), self.height) if out_mp4 else None
        x3_cams = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"{n}_x3_fingers") for n in self.traj] if x3 else []
        ids = {n: joint_ids(m, f"{n}_") for n in self.traj}
        log = {"t": [], **{f"{n}_cmd": [] for n in self.traj}, **{f"{n}_meas": [] for n in self.traj}, **{f"{n}_force": [] for n in self.traj}}
        bolt_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.scene.names.get("screws", [])]
        log["bolts"] = []
        next_frame = self.t0
        t = self.t0
        k_step = 0
        while t < self.t1:
            cmds = {f"{n}_": tr.at(t) for n, tr in self.traj.items()}
            if gc and k_step % 10 == 0:
                comp = gc(cmds)
            for name, tr in self.traj.items():
                q, jaw = cmds[f"{name}_"]
                sq = jaw_squeeze_rad if (squeeze_fn is None or squeeze_fn(name, t)) else 0.0
                _ctrl(m, d, f"{name}_", q, jaw - sq, comp.get(f"{name}_") if gc else None)
            if grasp is not None and k_step % 5 == 0:
                grasp.step(t, d)
            mujoco.mj_step(m, d)
            k_step += 1
            t += dt
            if t >= next_frame:
                next_frame += 1.0 / self.fps
                log["t"].append(t)
                for name, tr in self.traj.items():
                    q, jaw = tr.at(t)
                    log[f"{name}_cmd"].append(np.append(q, jaw))
                    meas = [d.qpos[m.jnt_qposadr[ids[name][n]]] for n in JOINTS] + [d.qpos[m.jnt_qposadr[ids[name]["jaw_L"]]]]
                    log[f"{name}_meas"].append(np.array(meas))
                    log[f"{name}_force"].append(np.array([d.actuator_force[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_act_{n}")] for n in JOINTS]))
                log["bolts"].append(np.array([np.concatenate([d.xpos[b], d.xquat[b]]) for b in bolt_ids]))
                if on_frame:
                    on_frame(t, d)
                if vw:
                    tiles = []
                    for cam in cams:
                        r.update_scene(d, cam, opt)
                        tiles.append(r.render().copy())
                    for cid in x3_cams:
                        r.update_scene(d, cid, opt)
                        tiles.append(r.render().copy())
                    img = np.hstack(tiles)
                    if hud:
                        img = hud(img, t)
                    vw.add(img)
        if vw:
            vw.close()
            r.close()
        return {k: (np.array(v) if isinstance(v, list) else v) for k, v in log.items()}
