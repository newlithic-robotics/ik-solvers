"""The dataset's feature layout and units, matching lerobot's `bi_so_follower` robot so a policy trained on this data drives
the real pair of arms without renaming anything.

    action / observation.state   (12,) float32
        left_shoulder_pan.pos, left_shoulder_lift.pos, left_elbow_flex.pos, left_wrist_flex.pos, left_wrist_roll.pos, left_gripper.pos,
        right_shoulder_pan.pos, ... right_gripper.pos

Units are a choice the deployment must match (`SOFollowerConfig.use_degrees`):
    degrees          the URDF / lerobot angle, zero at the middle of the calibrated travel.  What the IK stage produces.
    range_m100_100   lerobot's default: the calibrated travel of each servo mapped to [-100, 100].  Needs that arm's
                     calibration file (range_min / range_max in ticks per motor), see `load_calibration`.
The gripper is always a percentage (lerobot's RANGE_0_100 for the jaw servo).  Here 0 % = pads touching (jaw angle 0) and
`open_percent` (default 100) = the jaw stop, i.e. a calibration whose two ends are the closed pads and the open stop.
Calibrate the real jaw servo that way, or re-map this column.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .episode import JOINTS, RetargetedEpisode

MOTORS = JOINTS + ("gripper",)
ROBOT_TYPE = "bi_so_follower"
ENCODER_MAX = 4095                        # STS3215: 4096 ticks per turn, lerobot divides by (resolution - 1)
TICKS_PER_DEG = ENCODER_MAX / 360.0


def motor_names(hands) -> list:
    return [f"{hand}_{m}.pos" for hand in hands for m in MOTORS]


@dataclass
class Units:
    joints: str = "degrees"                                   # "degrees" | "range_m100_100"
    calibration: Optional[dict] = None                         # hand -> {motor: {"range_min": int, "range_max": int}}
    gripper_open_percent: float = 100.0                        # what the jaw stop maps to (0 or 100)

    def __post_init__(self):
        if self.joints not in ("degrees", "range_m100_100"):
            raise ValueError(f"unknown joint units {self.joints!r}")
        if self.joints == "range_m100_100" and not self.calibration:
            raise ValueError("range_m100_100 needs a calibration (range_min/range_max per motor per hand)")
        if self.gripper_open_percent not in (0.0, 100.0):
            raise ValueError("gripper_open_percent must be 0 or 100")

    def arm(self, q_deg: np.ndarray, hand: str) -> np.ndarray:
        """(…,5) degrees -> dataset units."""
        q = np.asarray(q_deg, float)
        if self.joints == "degrees":
            return q
        cal = self.calibration[hand]
        half = np.array([(cal[j]["range_max"] - cal[j]["range_min"]) / 2.0 for j in JOINTS])
        return q * TICKS_PER_DEG / half * 100.0

    def arm_to_degrees(self, v: np.ndarray, hand: str) -> np.ndarray:
        v = np.asarray(v, float)
        if self.joints == "degrees":
            return v
        cal = self.calibration[hand]
        half = np.array([(cal[j]["range_max"] - cal[j]["range_min"]) / 2.0 for j in JOINTS])
        return v / 100.0 * half / TICKS_PER_DEG

    def gripper(self, jaw_deg: np.ndarray, open_stop_deg: float) -> np.ndarray:
        frac = np.clip(np.asarray(jaw_deg, float) / open_stop_deg, 0.0, 1.0)
        return 100.0 * (frac if self.gripper_open_percent == 100.0 else 1.0 - frac)

    def gripper_to_jaw_deg(self, percent: np.ndarray, open_stop_deg: float) -> np.ndarray:
        frac = np.asarray(percent, float) / 100.0
        return open_stop_deg * (frac if self.gripper_open_percent == 100.0 else 1.0 - frac)

    def describe(self) -> dict:
        return dict(joints=self.joints, gripper="percent of the jaw travel", gripper_open_percent=self.gripper_open_percent,
                    calibrated=self.calibration is not None)


def load_calibration(path: Path) -> dict:
    """A lerobot calibration file (`~/.cache/huggingface/lerobot/calibration/robots/<robot>/<id>.json`): motor -> range_min/max."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for m in JOINTS:
        if m not in d:
            raise ValueError(f"{path}: no calibration for motor {m}")
        out[m] = {"range_min": int(d[m]["range_min"]), "range_max": int(d[m]["range_max"])}
    return out


def state_matrix(ep: RetargetedEpisode, units: Units) -> np.ndarray:
    """(n, 6 * hands) in the dataset's feature order, from the episode's degrees."""
    cols = []
    for hand, h in ep.hands.items():
        cols.append(units.arm(h.q_deg, hand))
        cols.append(units.gripper(h.jaw_deg, float(ep.gripper["open_stop_deg"]))[:, None])
    return np.concatenate(cols, axis=1)


# ------------------------------------------------------------------------------ the lerobot feature dictionary
POSE_NAMES = ("x", "y", "z", "qw", "qx", "qy", "qz")


def extra_features(hands) -> dict:
    """Per-frame context beyond state/action, for analysis and for policies that want it (float32 vectors)."""
    hands = list(hands)
    return {
        "observation.grip_pose": {"dtype": "float32", "shape": (7 * len(hands),),
                                  "names": [f"{h}_{a}" for h in hands for a in POSE_NAMES]},
        "observation.phase": {"dtype": "float32", "shape": (len(hands),), "names": [f"{h}_phase" for h in hands]},
        "observation.gap_mm": {"dtype": "float32", "shape": (len(hands),), "names": [f"{h}_gap_mm" for h in hands]},
    }


def dataset_features(hands, cameras: Optional[dict] = None, extras: bool = True) -> dict:
    """cameras: name -> (height, width).  Camera keys become `observation.images.<name>`."""
    names = motor_names(hands)
    f = {"action": {"dtype": "float32", "shape": (len(names),), "names": names},
         "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names}}
    for cam, (h, w) in (cameras or {}).items():
        f[f"observation.images.{cam}"] = {"dtype": "video", "shape": (int(h), int(w), 3), "names": ["height", "width", "channels"]}
    if extras:
        f.update(extra_features(hands))
    return f
