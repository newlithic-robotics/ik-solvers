"""Gravity load along a joint trajectory, from the model's masses (arm links from the URDF, gripper and camera from the CAD)."""
from __future__ import annotations

import mujoco
import numpy as np

from .model import JAW_JOINTS, JOINTS, joint_ids


def gravity_torques(model: mujoco.MjModel, q: np.ndarray, jaw: np.ndarray, prefix: str = "") -> np.ndarray:
    """(n,5) joint torque [N m] holding each pose still against gravity (sign: the torque the servo must produce)."""
    data = mujoco.MjData(model)
    ids = joint_ids(model, prefix)
    qadr = [model.jnt_qposadr[ids[n]] for n in JOINTS]
    dof = [model.jnt_dofadr[ids[n]] for n in JOINTS]
    jq = [model.jnt_qposadr[ids[n]] for n in JAW_JOINTS]
    out = np.zeros((len(q), 5))
    for i in range(len(q)):
        data.qpos[qadr] = q[i]
        data.qpos[jq[0]], data.qpos[jq[1]] = jaw[i], -jaw[i]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        out[i] = data.qfrc_bias[dof]
    return out
