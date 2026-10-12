"""Where to stand the arm: a base pose (x, y, heading) in the demonstration's world chosen by solving the task's key frames.

The score of a candidate base is what the per-frame solver leaves over on a set of key frames (grasps and releases on the
table, in-air holds, the docked rest, a few transports), each weighted by how much the task cares, plus penalties for
running into joint limits.  Coarse grid first, then a finer grid around the best few.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cad import GripperCAD
from .demo import PHASE, Demo, Placement
from .ik_wls import solve_first, solve_frame
from .kin import ArmKinematics
from .targets import Tolerances, make_targets


def key_frames(demo: Demo, hand: str, hold_every_s: float = 0.6, transport_every_s: float = 1.5, rest_weight: float = 0.3) -> tuple:
    """(indices, weights) of the frames a placement is judged on."""
    h = demo.hand(hand)
    idx, w = [], []
    dt = float(np.median(np.diff(h.t)))
    for k in h.holds():
        frames = np.where(h.hold_id == k)[0]
        c = h.contact_frame(k)
        table = h.height[c] < 0.06 and -h.approach[c, 2] > 0.7
        idx.append(c); w.append(3.0 if table else 1.5)
        holds = frames[h.phase[frames] == PHASE["hold"]]
        step = max(1, int(round(hold_every_s / dt)))
        for i in holds[step::step]:
            idx.append(int(i)); w.append(1.0)
        rel = frames[h.phase[frames] == PHASE["release"]]
        if len(rel):
            idx.append(int(rel[len(rel) // 2])); w.append(1.5 if table else 1.0)
    trans = np.where(h.phase == PHASE["transport"])[0]
    step = max(1, int(round(transport_every_s / dt)))
    for i in trans[::step]:
        idx.append(int(i)); w.append(rest_weight if (h.t[i] < 6.0 or h.t[i] > 44.0) else 0.15)
    order = np.argsort(idx)
    return np.array(idx)[order], np.array(w)[order]


@dataclass
class Candidate:
    placement: Placement
    score: float
    pos_mm_med: float
    tilt_deg_med: float
    limit_min_deg: float
    pos_mm_max: float


def score_placement(kin: ArmKinematics, demo: Demo, hand: str, pl: Placement, cad: GripperCAD, idx: np.ndarray, w: np.ndarray,
                    tol: Tolerances, pos_cap_mm: float = 20.0) -> Candidate:
    tg = make_targets(demo, hand, pl, cad, tol, free_roll=True)
    q_prev = None
    cost = 0.0
    pos, tilt, lim = [], [], []
    for i, wi in zip(idx, w):
        if q_prev is None:
            r = solve_first(kin, tg, int(i))
        else:
            r = solve_frame(kin, tg.R_target(int(i)), tg.p_b[i], tg.jaw_rad[i], q_prev, None, tg.sigma_p_m, tg.sig_tilt_deg[i], tg.sig_roll_deg[i],
                            tg.continuity_deg)
        q_prev = r.q
        p_mm = min(r.pos_err_m * 1000, pos_cap_mm)
        c = (p_mm / 0.5) ** 2 + (r.tilt_err_deg / tg.sig_tilt_deg[i]) ** 2 + 2.0 * max(0.0, 3.0 - r.limit_margin_deg)
        cost += wi * c
        pos.append(r.pos_err_m * 1000); tilt.append(r.tilt_err_deg); lim.append(r.limit_margin_deg)
    return Candidate(pl, float(cost / w.sum()), float(np.median(pos)), float(np.median(tilt)), float(np.min(lim)), float(np.max(pos)))


def search(kin: ArmKinematics, demo: Demo, hand: str, cad: GripperCAD, surface_z_base: float, centre_xy: tuple | None = None,
           radius_m: float = 0.45, step_m: float = 0.05, headings_deg=None, refine: int = 2, keep: int = 6, tol: Tolerances | None = None,
           progress=None) -> list[Candidate]:
    """Grid over base positions around the hand's workspace centre and headings; refined `refine` times around the best."""
    tol = tol or Tolerances()
    h = demo.hand(hand)
    idx, w = key_frames(demo, hand)
    task = h.phase != PHASE["transport"]
    c0 = np.array(centre_xy) if centre_xy is not None else h.p[task, :2].mean(0)
    headings = np.arange(0, 360, 30.0) if headings_deg is None else np.asarray(headings_deg, float)
    cands = []
    xs = np.arange(-radius_m, radius_m + 1e-9, step_m)
    grid = [(c0[0] + dx, c0[1] + dy, hd) for dx in xs for dy in xs for hd in headings
            if 0.12 <= np.hypot(dx, dy) <= radius_m]
    # keep only headings that roughly face the workspace from the base
    grid = [(x, y, hd) for x, y, hd in grid if abs(((np.degrees(np.arctan2(c0[1] - y, c0[0] - x)) - hd + 180) % 360) - 180) < 50]
    for n, (x, y, hd) in enumerate(grid):
        cands.append(score_placement(kin, demo, hand, Placement(x, y, hd, surface_z_base), cad, idx, w, tol))
        if progress and n % 20 == 0:
            progress(n, len(grid), cands[-1] if cands else None)
    cands.sort(key=lambda c: c.score)
    best = cands[:keep]
    for level in range(refine):
        s = step_m / (2 ** (level + 1))
        hs = 30.0 / (2 ** (level + 1))
        new = []
        for b in best:
            for dx in (-s, 0, s):
                for dy in (-s, 0, s):
                    for dh in (-hs, 0, hs):
                        if dx == dy == dh == 0:
                            continue
                        pl = Placement(b.placement.x_m + dx, b.placement.y_m + dy, b.placement.yaw_deg + dh, surface_z_base)
                        new.append(score_placement(kin, demo, hand, pl, cad, idx, w, tol))
        cands = sorted(best + new, key=lambda c: c.score)
        best = cands[:keep]
        if progress:
            progress(-1, level, best[0])
    return best
