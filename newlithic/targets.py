"""What every solver is asked for: per-frame grip targets in an arm's base frame, the jaw angle, and how tightly each
orientation component is wanted in each task phase.

Orientation is expressed in the demonstrated triad: a turn about the APPROACH axis (the roll of the jaws about the fingers)
and tilts of the approach about the OPENING line and about the NORMAL.  Per phase the task needs them differently: during a
transport nothing cares, at a grasp from above the approach must be right, while holding something the roll is the task
itself (unscrewing).  The tolerances are tightened `anticipate_s` ahead of a stricter phase and smoothed in the log domain so
no solver sees a step.

Every solver in this package consumes the same Targets object, so comparisons between solvers compare the solvers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cad import GripperCAD
from .demo import PHASE, PHASE_NAME, Demo, Placement

FLIP = np.diag([1.0, -1.0, -1.0])        # the same grasp with the jaws swapped: a half turn about the approach axis


@dataclass
class Tolerances:
    sigma_p_m: float = 0.0005
    tilt_deg: dict = field(default_factory=lambda: {"transport": 25.0, "approach": 6.0, "grasp": 3.0, "hold": 6.0, "release": 4.0})
    roll_deg: dict = field(default_factory=lambda: {"transport": 60.0, "approach": 20.0, "grasp": 8.0, "hold": 10.0, "release": 10.0})
    anticipate_s: float = 1.0
    smooth_s: float = 1.0
    continuity_deg: float = 15.0          # per-frame pull toward the previous solution (branch keeping, not tracking)

    def arrays(self, phase: np.ndarray, t: np.ndarray, free_roll: bool = False):
        st = np.array([self.tilt_deg[PHASE_NAME[int(p)]] for p in phase], float)
        sr = np.array([60.0 if (free_roll and PHASE_NAME[int(p)] != "transport") else self.roll_deg[PHASE_NAME[int(p)]] for p in phase], float)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / 30
        ahead = max(0, int(round(self.anticipate_s / dt)))
        k = max(1, int(round(self.smooth_s / dt)) | 1)

        def shape(a):
            n = len(a)
            m = np.array([a[i:i + ahead + 1].min() for i in range(n)]) if ahead else a.copy()
            la = np.pad(np.log(m), (k // 2, k // 2), mode="edge")
            return np.exp(np.convolve(la, np.ones(k) / k, mode="valid"))
        return shape(st), shape(sr)


@dataclass
class Targets:
    hand: str
    t: np.ndarray
    R_b: np.ndarray            # (n,3,3) base <- grip, demonstrated
    p_b: np.ndarray            # (n,3) m
    jaw_rad: np.ndarray        # (n,) the jaw angle that gives the demonstrated pad gap
    gap_m: np.ndarray
    phase: np.ndarray
    hold_id: np.ndarray
    sig_tilt_deg: np.ndarray
    sig_roll_deg: np.ndarray
    sigma_p_m: float
    continuity_deg: float
    placement: Placement
    height_m: np.ndarray       # demonstrated grip height over the reference surface
    roll_shift: np.ndarray = None      # per-frame turn of the target about its approach (set by a free-roll pass)
    flip: bool = False

    @property
    def n(self) -> int:
        return len(self.t)

    def R_target(self, i: int) -> np.ndarray:
        R = self.R_b[i] @ FLIP if self.flip else self.R_b[i]
        if self.roll_shift is not None and self.roll_shift[i]:
            c, s = np.cos(self.roll_shift[i]), np.sin(self.roll_shift[i])
            R = R @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        return R

    def all_R_targets(self) -> np.ndarray:
        return np.stack([self.R_target(i) for i in range(self.n)])


def make_targets(demo: Demo, hand: str, placement: Placement, cad: GripperCAD, tol: Tolerances | None = None,
                 free_roll: bool = False, flip: bool = False) -> Targets:
    tol = tol or Tolerances()
    h = demo.hand(hand)
    R_b, p_b = placement.to_base(h.R, h.p, demo.table_z)
    st, sr = tol.arrays(h.phase, h.t, free_roll)
    jaw = np.radians(cad.q_from_gap(h.gap * 1000.0))
    return Targets(hand=hand, t=h.t.copy(), R_b=R_b, p_b=p_b, jaw_rad=jaw, gap_m=h.gap.copy(), phase=h.phase.copy(),
                   hold_id=h.hold_id.copy(), sig_tilt_deg=st, sig_roll_deg=sr, sigma_p_m=tol.sigma_p_m,
                   continuity_deg=tol.continuity_deg, placement=placement, height_m=h.height.copy(), flip=flip)
