"""Collision checks of joint trajectories in the cell model (MuJoCo's own collision pipeline, convex hulls of the CAD meshes).

A "collision" is any contact with penetration beyond `tol` between geoms that are not permanently overlapping in the
model's zero pose (meshing gears, servos inside their holders).  Reported per frame with the worst pair.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .model import JAW_JOINTS, JOINTS, joint_ids


@dataclass
class CollisionReport:
    penetration_mm: np.ndarray      # (n,) worst penetration per frame (0 = clear)
    min_distance_mm: np.ndarray     # (n,) smallest signed distance among the checked pairs (positive = clearance), from margin-expanded detection
    worst_pair: list                # (n,) (body1, body2) of the worst contact, or None
    pairs_seen: dict                # (body1, body2) -> frames in contact
    object_mm: np.ndarray = None    # (n,) deepest contact between a gripper and a manipulated object (bolt/nut): a grasp, not a collision

    def summary(self) -> str:
        n_bad = int((self.penetration_mm > 0).sum())
        lines = [f"collision frames {n_bad}/{len(self.penetration_mm)}, worst penetration {self.penetration_mm.max():.1f} mm, "
                 f"min clearance {self.min_distance_mm.min():.1f} mm"
                 + (f"; gripper-object contact frames {int((self.object_mm > 0).sum())} (grasps, not counted)" if self.object_mm is not None else "")]
        pm = getattr(self, "pair_max_mm", {})
        for pair, frames in sorted(self.pairs_seen.items(), key=lambda kv: -len(kv[1]))[:8]:
            lines.append(f"  {pair[0]} - {pair[1]}: {len(frames)} frames (first t-index {frames[0]}, max {pm.get(pair, 0.0):.1f} mm)")
        return "\n".join(lines)


class Collider:
    def __init__(self, model: mujoco.MjModel, arms: dict, margin: float = 0.02):
        """arms: prefix -> None; the joint trajectories are given per prefix to check()."""
        self.model = model
        self.data = mujoco.MjData(model)
        self.arms = list(arms)
        self.ids = {p: joint_ids(model, p) for p in self.arms}
        self.margin = margin
        # permanent overlaps: contacts in the zero pose
        self.model.opt.disableflags |= 0
        self._set_all({p: np.zeros(5) for p in self.arms}, {p: 0.0 for p in self.arms})
        mujoco.mj_forward(model, self.data)
        self.ignore = set(self._pairs(self.data, 0.0))

    def _set_all(self, q: dict, jaw: dict):
        m, d = self.model, self.data
        for p in self.arms:
            ids = self.ids[p]
            for n, v in zip(JOINTS, q[p]):
                d.qpos[m.jnt_qposadr[ids[n]]] = v
            d.qpos[m.jnt_qposadr[ids["jaw_L"]]] = jaw[p]
            d.qpos[m.jnt_qposadr[ids["jaw_R"]]] = -jaw[p]

    def _pairs(self, d, min_pen):
        m = self.model
        out = []
        for c in d.contact[:d.ncon]:
            if -c.dist >= min_pen:
                b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
                out.append((b1, b2))
        return out

    def check(self, q: dict, jaw: dict, tol_mm: float = 0.5) -> CollisionReport:
        """q: prefix -> (n,5) rad; jaw: prefix -> (n,) rad.  Static bodies use their model pose."""
        n = len(next(iter(q.values())))
        pen = np.zeros(n)
        obj = np.zeros(n)
        mind = np.full(n, np.inf)
        worst = [None] * n
        seen = {}
        m, d = self.model, self.data
        is_object = lambda b: b.startswith("bolt") or b.startswith("nut")
        is_jaw = lambda b: b.endswith("jaw_L") or b.endswith("jaw_R") or b.endswith("gripper_link")
        geom_margin = m.geom_margin.copy()
        m.geom_margin[:] = self.margin
        self._pair_max = {}
        for i in range(n):
            self._set_all({p: q[p][i] for p in self.arms}, {p: jaw[p][i] for p in self.arms})
            mujoco.mj_forward(m, d)
            for c in d.contact[:d.ncon]:
                b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
                if (b1, b2) in self.ignore or (b2, b1) in self.ignore:
                    continue
                if (is_object(b1) and is_jaw(b2)) or (is_object(b2) and is_jaw(b1)):
                    obj[i] = max(obj[i], -c.dist * 1000)
                    continue
                mind[i] = min(mind[i], c.dist * 1000)
                if -c.dist * 1000 > tol_mm and -c.dist * 1000 > pen[i]:
                    pen[i] = -c.dist * 1000
                    worst[i] = (b1, b2)
                if -c.dist * 1000 > tol_mm:
                    seen.setdefault((b1, b2), []).append(i)
                    self._pair_max[(b1, b2)] = max(self._pair_max.get((b1, b2), 0.0), -c.dist * 1000)
        m.geom_margin[:] = geom_margin
        mind[np.isinf(mind)] = self.margin * 1000
        rep = CollisionReport(pen, mind, worst, seen, obj)
        rep.pair_max_mm = dict(self._pair_max)
        return rep
