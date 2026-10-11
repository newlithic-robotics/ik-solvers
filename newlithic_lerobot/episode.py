"""The retargeted episode: what the IK stage produces and the export stage consumes.

One `.npz` + one `.json` per episode.  Everything a dataset needs is in here, so the export stage never touches MuJoCo:

    t                   (n,)     seconds, the demonstration's clock (the left rig's pose instants)
    <hand>_q_deg        (n,5)    arm joints, lerobot / URDF convention (zero = middle of the calibrated travel), degrees
    <hand>_jaw_deg      (n,)     jaw opening q (REV_L; 0 = pads touching, open_stop_deg = the stop)
    <hand>_gap_mm       (n,)     the pad gap the demonstration measured (what the jaw angle reproduces)
    <hand>_phase        (n,)     0 transport, 1 grasp, 2 hold, 3 release, 4 approach
    <hand>_hold_id      (n,)     which grasp a frame belongs to, -1 outside one
    <hand>_pose_demo    (n,7)    demonstrated grip pose in that arm's base frame: x y z [m], qw qx qy qz
    <hand>_pose_fk      (n,7)    the grip pose the solved joints actually give (forward kinematics)
    <hand>_pos_err_mm, _tilt_err_deg, _roll_err_deg, _lost_deg   (n,)  per-frame residuals against the demonstration

and in the JSON: episode name, fps, hand names, joint names, the base placements, the gripper law, the solver stages,
summary numbers and provenance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
PHASE = {"transport": 0, "grasp": 1, "hold": 2, "release": 3, "approach": 4}
PHASE_NAME = {v: k for k, v in PHASE.items()}
HAND_ARRAYS = ("q_deg", "jaw_deg", "gap_mm", "phase", "hold_id", "pose_demo", "pose_fk",
               "pos_err_mm", "tilt_err_deg", "roll_err_deg", "lost_deg")


@dataclass
class HandTrajectory:
    q_deg: np.ndarray
    jaw_deg: np.ndarray
    gap_mm: np.ndarray
    phase: np.ndarray
    hold_id: np.ndarray
    pose_demo: np.ndarray
    pose_fk: np.ndarray
    pos_err_mm: np.ndarray
    tilt_err_deg: np.ndarray
    roll_err_deg: np.ndarray
    lost_deg: np.ndarray

    @property
    def n(self) -> int:
        return len(self.jaw_deg)

    def validate(self, n: int, name: str = "") -> None:
        for k in HAND_ARRAYS:
            a = getattr(self, k)
            if a.shape[0] != n:
                raise ValueError(f"{name}{k}: {a.shape[0]} frames, expected {n}")
        if self.q_deg.shape[1:] != (5,):
            raise ValueError(f"{name}q_deg must be (n,5), got {self.q_deg.shape}")
        for k in ("pose_demo", "pose_fk"):
            if getattr(self, k).shape[1:] != (7,):
                raise ValueError(f"{name}{k} must be (n,7) [x y z qw qx qy qz]")


@dataclass
class RetargetedEpisode:
    episode: str
    fps: float
    t: np.ndarray
    hands: dict                      # name -> HandTrajectory, in the order the dataset features will use
    placements: dict                 # name -> dict(x_m, y_m, yaw_deg, surface_z_base)
    gripper: dict                    # open_stop_deg, mm_per_deg, ... (the jaw law, for the record)
    solver: dict = field(default_factory=dict)       # stages, configs, summaries
    provenance: dict = field(default_factory=dict)   # where the demonstration came from, which code produced this
    task: Optional[str] = None

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0])

    def validate(self) -> None:
        if self.t.ndim != 1 or self.n < 2:
            raise ValueError("t must be a 1-D array of at least two instants")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("t must be strictly increasing")
        for name, h in self.hands.items():
            h.validate(self.n, f"{name}_")

    def save(self, path: Path) -> Path:
        """Writes <path>.npz and <path>.json (the suffix of `path` is ignored)."""
        self.validate()
        base = Path(path).with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"t": np.asarray(self.t, float)}
        for name, h in self.hands.items():
            for k in HAND_ARRAYS:
                arrays[f"{name}_{k}"] = np.asarray(getattr(h, k))
        np.savez_compressed(base.with_suffix(".npz"), **arrays)
        meta = dict(format="newlithic_retargeted_episode", version=1, episode=self.episode, fps=self.fps, frames=int(self.n),
                    duration_s=self.duration_s, hands=list(self.hands), joints=list(JOINTS), phase_codes=PHASE, task=self.task,
                    placements=self.placements, gripper=self.gripper, solver=self.solver, provenance=self.provenance)
        base.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=_jsonable), encoding="utf-8")
        return base.with_suffix(".npz")

    @classmethod
    def load(cls, path: Path) -> "RetargetedEpisode":
        base = Path(path).with_suffix("")
        meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("format") != "newlithic_retargeted_episode":
            raise ValueError(f"{base}.json is not a retargeted episode")
        z = np.load(base.with_suffix(".npz"))
        hands = {name: HandTrajectory(**{k: z[f"{name}_{k}"] for k in HAND_ARRAYS}) for name in meta["hands"]}
        ep = cls(episode=meta["episode"], fps=float(meta["fps"]), t=z["t"], hands=hands, placements=meta["placements"],
                 gripper=meta["gripper"], solver=meta.get("solver", {}), provenance=meta.get("provenance", {}),
                 task=meta.get("task"))
        ep.validate()
        return ep

    def describe(self) -> str:
        lines = [f"{self.episode}: {self.n} frames, {self.duration_s:.1f} s at {self.fps:.2f} fps, hands {list(self.hands)}"]
        for name, h in self.hands.items():
            task = h.phase != PHASE["transport"]
            sp = np.abs(np.diff(h.q_deg, axis=0) / np.diff(self.t)[:, None])
            rng = ", ".join(f"{lo:.0f}..{hi:.0f}" for lo, hi in zip(h.q_deg.min(0), h.q_deg.max(0)))
            lines.append(f"  {name}: joints [{rng}] deg, jaw {h.jaw_deg.min():.1f}..{h.jaw_deg.max():.1f} deg "
                         f"(gap {h.gap_mm.min():.1f}..{h.gap_mm.max():.1f} mm), speed p95 {np.percentile(sp, 95):.0f} max {sp.max():.0f} deg/s")
            if task.any():
                lines.append(f"     task frames {int(task.sum())}: position error median {np.median(h.pos_err_mm[task]):.2f} mm, "
                             f"tilt {np.median(h.tilt_err_deg[task]):.1f} deg (lost axis {np.median(h.lost_deg[task]):.1f}), "
                             f"roll {np.median(h.roll_err_deg[task]):.1f} deg")
            counts = {PHASE_NAME[int(c)]: int((h.phase == c).sum()) for c in np.unique(h.phase)}
            lines.append(f"     phases {counts}, holds {sorted(int(k) for k in np.unique(h.hold_id) if k >= 0)}")
        for name, p in self.placements.items():
            lines.append(f"  base {name}: ({p['x_m'] * 1000:.1f}, {p['y_m'] * 1000:.1f}) mm, yaw {p['yaw_deg']:.1f} deg, "
                         f"reference surface at {p['surface_z_base'] * 1000:.2f} mm in the base frame")
        return "\n".join(lines)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def pose7(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(n,3,3), (n,3) -> (n,7) [x y z qw qx qy qz], the quaternion sign kept continuous along the trajectory."""
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(np.asarray(R)).as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    return np.concatenate([np.asarray(p, float), q], axis=1)
