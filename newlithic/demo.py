"""The demonstration as the arm needs it: grip frames per hand on one clock in one world, and their expression in an arm's base.

Loaded from assets/demo/episode1.npz (written by tools/export_demo.py from the state-estimation pipeline's output).  The grip
frame of a hand is: origin = midpoint of the two pad contact centres, columns of R = approach (along the fingers), opening
(across the jaws), normal.  `gap` is the pad gap the pipeline measured (calibrated on the jig), `phase` labels every frame
as transport / approach / grasp / hold / release from the gap signal, `hold_id` numbers the grasps.

Placement: where an arm stands in the demonstration's world.  T_world_base = [rotz(yaw) | (x, y, z)], with z chosen so that
the demonstration's reference surface (the plate top the hands touched, `table_z`) sits at `surface_z_base` in the base frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cad import ASSETS, rotz

PHASE = {"transport": 0, "grasp": 1, "hold": 2, "release": 3, "approach": 4}
PHASE_NAME = {v: k for k, v in PHASE.items()}

# Holds that are not task grasps, by episode and hand (verified in the footage): relabelled as transport on load.
#   Episode-1 left hold 3 (t 42.0-43.1 s, 50 mm up, fingers 59 deg off vertical): the jaws closing while the gripper is put back
#   into its docking jig, after both screws are done.  The synchronisation claps at 3.5 s were already dropped by the exporter.
NOT_A_GRASP = {"Episode-1": {"left": [3]}}


@dataclass
class Hand:
    name: str
    t: np.ndarray
    R: np.ndarray
    p: np.ndarray
    gap: np.ndarray            # m
    phase: np.ndarray
    hold_id: np.ndarray
    valid: np.ndarray
    height: np.ndarray         # grip origin above the reference surface [m]
    tips_height: np.ndarray
    pad_to_tip_m: float
    events: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def approach(self) -> np.ndarray:
        return self.R[:, :, 0]

    @property
    def opening(self) -> np.ndarray:
        return self.R[:, :, 1]

    def mask(self, *names) -> np.ndarray:
        return np.isin(self.phase, [PHASE[n] for n in names])

    def holds(self) -> list[int]:
        return sorted(int(k) for k in np.unique(self.hold_id) if k >= 0)

    def contact_frame(self, k: int) -> int:
        """First 'hold' frame of grasp k (the contact)."""
        f = np.where((self.hold_id == k) & (self.phase == PHASE["hold"]))[0]
        return int(f[0]) if len(f) else int(np.where(self.hold_id == k)[0][0])


@dataclass
class Demo:
    episode: str
    fps: float
    table_z: float             # the reference surface (plate top) in the world [m]
    hands: dict
    notes: dict

    def hand(self, name: str) -> Hand:
        return self.hands[name]


def load_demo(path: Path = ASSETS / "demo" / "episode1.npz", drop_non_grasps: bool = True) -> Demo:
    z = np.load(path)
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    hands = {}
    for name, hm in meta["hands"].items():
        h = Hand(name=name, t=z[f"{name}_t"], R=z[f"{name}_R"], p=z[f"{name}_p"], gap=z[f"{name}_gap"], phase=z[f"{name}_phase"].copy(),
                 hold_id=z[f"{name}_hold_id"].copy(), valid=z[f"{name}_valid"].astype(bool), height=z[f"{name}_height"],
                 tips_height=z[f"{name}_tips_height"], pad_to_tip_m=float(hm["pad_to_tip_m"]), events=hm["events"])
        if drop_non_grasps:
            for k in NOT_A_GRASP.get(meta["episode"], {}).get(name, []):
                m = h.hold_id == k
                h.phase[m] = PHASE["transport"]
                h.hold_id[m] = -1
        hands[name] = h
    return Demo(meta["episode"], float(meta["fps"]), float(meta["table_z"]), hands, meta["notes"])


@dataclass
class Placement:
    x_m: float
    y_m: float
    yaw_deg: float
    surface_z_base: float      # where the demonstration's reference surface is in the base frame [m]

    def T_world_base(self, table_z_world: float) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = rotz(self.yaw_deg)
        T[:3, 3] = [self.x_m, self.y_m, table_z_world - self.surface_z_base]
        return T

    def to_base(self, R_w: np.ndarray, p_w: np.ndarray, table_z_world: float):
        """World grip frames (n,3,3), (n,3) -> base frame."""
        T = self.T_world_base(table_z_world)
        Rb, tb = T[:3, :3], T[:3, 3]
        return np.einsum("ji,njk->nik", Rb, R_w), (p_w - tb) @ Rb

    def to_world(self, R_b, p_b, table_z_world: float):
        T = self.T_world_base(table_z_world)
        Rb, tb = T[:3, :3], T[:3, 3]
        return np.einsum("ij,njk->nik", Rb, R_b), p_b @ Rb.T + tb

    def as_dict(self) -> dict:
        return dict(x_m=self.x_m, y_m=self.y_m, yaw_deg=self.yaw_deg, surface_z_base=self.surface_z_base)


def describe(demo: Demo) -> str:
    lines = [f"{demo.episode}: {demo.fps:.2f} fps, reference surface (plate top) at z = {demo.table_z*1000:.1f} mm in the world"]
    for name, h in demo.hands.items():
        lines.append(f"  {name}: {h.n} frames, t {h.t[0]:.2f}..{h.t[-1]:.2f} s, gap {h.gap.min()*1000:.1f}..{h.gap.max()*1000:.1f} mm, holds {h.holds()}")
        for k in h.holds():
            c = h.contact_frame(k)
            rel = np.where(h.hold_id == k)[0][-1]
            el = lambda i: np.degrees(np.arccos(np.clip(-h.approach[i, 2], -1, 1)))
            lines.append(f"     hold {k}: contact t {h.t[c]:.2f} s at {h.height[c]*1000:.0f} mm, fingers {el(c):.0f} deg off vertical, "
                         f"gap {h.gap[c]*1000:.1f} mm; release t {h.t[rel]:.2f} s at {h.height[rel]*1000:.0f} mm")
    return "\n".join(lines)
