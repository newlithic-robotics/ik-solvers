"""Forward kinematics and Jacobians of one arm, straight from the MuJoCo model (so every solver and the simulator agree).

The grip point is the midpoint of the two pad contact sites (it moves with the jaw angle); the grip orientation is the
gripper_link orientation times the constant CAD triad (columns approach, opening, normal).  Joint angles are radians here;
the lerobot/URDF zero convention throughout.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .cad import GripperCAD, load_rig
from .model import JAW_JOINTS, JOINTS, compile_arm, joint_ids


class ArmKinematics:
    def __init__(self, model: mujoco.MjModel | None = None, prefix: str = "", cad: GripperCAD | None = None):
        if model is None:
            model, _, cad, _ = compile_arm(save=False)
        self.model = model
        self.data = mujoco.MjData(model)
        self.prefix = prefix
        self.cad = cad or load_rig()
        ids = joint_ids(model, prefix)
        self.qadr = np.array([model.jnt_qposadr[ids[n]] for n in JOINTS])
        self.dofadr = np.array([model.jnt_dofadr[ids[n]] for n in JOINTS])
        self.jaw_qadr = np.array([model.jnt_qposadr[ids[n]] for n in JAW_JOINTS])
        self.limits = np.array([model.jnt_range[ids[n]] for n in JOINTS])          # (5,2) rad
        sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + n)
        self.s_padL, self.s_padR, self.s_flange, self.s_cam = sid("pad_L"), sid("pad_R"), sid("flange"), sid("camera")
        self.b_grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "gripper_link")
        self.b_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link")
        self.T_grip = self.cad.grip_triad_g()
        self._jp = np.zeros((3, model.nv)); self._jp2 = np.zeros((3, model.nv)); self._jr = np.zeros((3, model.nv))

    # ------------------------------------------------------------------ state
    def set(self, q: np.ndarray, jaw: float):
        self.data.qpos[self.qadr] = q
        self.data.qpos[self.jaw_qadr[0]] = jaw
        self.data.qpos[self.jaw_qadr[1]] = -jaw
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def grip(self, q, jaw: float):
        """(R, p): the grip frame in the model's world (= the base frame for a single-arm model)."""
        self.set(q, jaw)
        d = self.data
        p = 0.5 * (d.site_xpos[self.s_padL] + d.site_xpos[self.s_padR])
        R = d.xmat[self.b_grip].reshape(3, 3) @ self.T_grip
        return R, p

    def grip_jac(self, q, jaw: float):
        """(R, p, J) with J (6,5): rows [v; w] of the grip point per unit joint rate, in the model's world frame."""
        R, p = self.grip(q, jaw)
        m, d = self.model, self.data
        mujoco.mj_jacSite(m, d, self._jp, self._jr, self.s_padL)
        mujoco.mj_jacSite(m, d, self._jp2, None, self.s_padR)
        J = np.zeros((6, 5))
        J[:3] = 0.5 * (self._jp[:, self.dofadr] + self._jp2[:, self.dofadr])
        J[3:] = self._jr[:, self.dofadr]
        return R, p, J

    def flange(self, q, jaw: float = 0.0):
        self.set(q, jaw)
        return self.data.xmat[self.b_grip].reshape(3, 3).copy(), self.data.site_xpos[self.s_flange].copy()

    def camera(self, q, jaw: float = 0.0):
        self.set(q, jaw)
        return self.data.site_xmat[self.s_cam].reshape(3, 3).copy(), self.data.site_xpos[self.s_cam].copy()

    def limit_margin(self, q) -> float:
        return float(np.minimum(q - self.limits[:, 0], self.limits[:, 1] - q).min())

    @property
    def pan_axis_xy(self) -> np.ndarray:
        """The shoulder-pan axis (x, y) in the base frame."""
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.prefix + "shoulder_pan")
        b = self.model.jnt_bodyid[j]
        self.set(np.zeros(5), 0.0)
        return (self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ self.model.jnt_pos[j])[:2].copy()


def rot_err(R_target: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Axis-angle vector (rad, world frame) that turns R onto R_target."""
    R_e = R_target @ R.T
    c = np.clip((np.trace(R_e) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R_e[2, 1] - R_e[1, 2], R_e[0, 2] - R_e[2, 0], R_e[1, 0] - R_e[0, 1]])
    return ang / (2 * np.sin(ang)) * v


def rotation_mobility(J: np.ndarray):
    """With the grip point fixed, which rotations can five joints make?  Returns (reachable (3,2), gains (2,), lost (3,))."""
    Jp, Jo = J[:3], J[3:]
    U, s, Vt = np.linalg.svd(Jp)
    rank = int((s > 1e-6 * max(s[0], 1e-12)).sum())
    N = Vt[rank:].T
    M = Jo @ N
    Um, sm, _ = np.linalg.svd(M, full_matrices=True)
    k = min(2, M.shape[1])
    lost = Um[:, -1] if M.shape[1] < 3 else np.zeros(3)
    return Um[:, :k], sm[:k], lost


def split_rotation_error(e_r: np.ndarray, lost: np.ndarray):
    l = lost * float(lost @ e_r)
    return e_r - l, l
