"""Solver 2 - differential inverse kinematics as a quadratic programme ("QP"), built on mink.

Every frame: minimise  sum_tasks || W (J dq - e/dt) ||^2 + damping ||dq||^2   subject to
    joint limits        (ConfigurationLimit: q + dq inside the range)
    velocity limits     (VelocityLimit: the servo's no-load speed, or a practical cap)
    collision avoidance (CollisionAvoidanceLimit: a linear constraint per geom pair, from signed distances)
then integrate dq.  Tasks: the grip frame (position cost 1/sigma_p, orientation costs per axis of the grip triad from the
phase tolerances, the LOST component projected out as in the WLS solver), a weak posture task toward a rest pose.

Compared to the per-frame least squares this is a local, velocity-level method: it follows the target as closely as the
constraints allow and never jumps between branches, which is what a servo-driven arm can execute - and it is the only
family here that can take collisions with the table, the other arm and itself as hard constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mink
import mujoco
import numpy as np

from .kin import ArmKinematics, rotation_mobility
from .model import JOINTS
from .targets import Targets


class GripTask(mink.FrameTask):
    """FrameTask on the grip site with (a) the target shifted so the site (fixed at the closed-pose midpoint) lands where the
    jaw-dependent grip point should be, and (b) the orientation error projected off the arm's lost axis."""

    def __init__(self, site: str, kin: ArmKinematics, position_cost, orientation_cost, lm_damping=1e-3, project_lost=True):
        super().__init__(frame_name=site, frame_type="site", position_cost=position_cost, orientation_cost=orientation_cost, lm_damping=lm_damping)
        self.kin = kin
        self.project_lost = project_lost
        self._lost_local = None

    def set_lost(self, lost_local):
        self._lost_local = lost_local

    def compute_error(self, configuration):
        e = super().compute_error(configuration)          # (6,) [position; orientation] in the site's local frame
        if self.project_lost and self._lost_local is not None:
            l = self._lost_local
            e[3:] = e[3:] - l * (l @ e[3:])
        return e

    def compute_jacobian(self, configuration):
        J = super().compute_jacobian(configuration)
        if self.project_lost and self._lost_local is not None:
            l = self._lost_local
            J[3:] = J[3:] - np.outer(l, l @ J[3:])
        return J


@dataclass
class QPConfig:
    dt_integration: float = 1.0 / 30
    vel_limit_rad_s: dict = field(default_factory=lambda: {n: np.radians(120.0) for n in JOINTS})   # a practical cap (no-load is 270 deg/s at 12 V);
                                                                                                    # it also damps the step overshoot (sweep: 270 -> 0.68 deg/frame chatter, 90 -> 0.36)
    posture_cost: float = 1e-3
    posture_target_deg: tuple = (0.0, -60.0, 60.0, 50.0, 0.0)
    damping: float = 1e-6
    lm_damping: float = 1e-3
    solver: str = "daqp"
    iterations: int = 3            # QP steps per frame (the targets are a position trajectory, not a velocity stream)
    gain: float = 1.0
    collision_min_dist: float = 0.003
    collision_detect_dist: float = 0.02
    project_lost: bool = True


def arm_geom_names(model: mujoco.MjModel, prefix: str, bodies: tuple) -> list:
    out = []
    for g in range(model.ngeom):
        b = model.body(model.geom_bodyid[g]).name
        if b in tuple(prefix + x for x in bodies) and model.geom_contype[g] | model.geom_conaffinity[g]:
            out.append(model.geom(g).name or g)
    return out


def default_collision_pairs(model: mujoco.MjModel, prefix: str, others: list) -> list:
    """Geom pairs to keep apart: this arm's gripper (incl. camera, jaws) vs its own lower links and base; the whole arm vs the
    scene's obstacle geoms (table, plate, jig, the other arm's geoms)."""
    gripper = arm_geom_names(model, prefix, ("gripper_link", "jaw_L", "jaw_R"))
    arm_lower = arm_geom_names(model, prefix, ("base_link", "shoulder_link", "upper_arm_link"))
    whole = arm_geom_names(model, prefix, ("upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "jaw_L", "jaw_R"))
    pairs = []
    if gripper and arm_lower:
        pairs.append((gripper, arm_lower))
    if whole and others:
        pairs.append((whole, others))
    return pairs


class QPSolver:
    def __init__(self, model: mujoco.MjModel, kin: ArmKinematics, cfg: QPConfig, prefix: str = "", obstacle_geoms: list | None = None,
                 site_name: str | None = None):
        self.model, self.kin, self.cfg, self.prefix = model, kin, cfg, prefix
        self.conf = mink.Configuration(model)
        self.site = site_name or (prefix + "grip")
        self.task = GripTask(self.site, kin, position_cost=1.0, orientation_cost=1.0, lm_damping=cfg.lm_damping, project_lost=cfg.project_lost)
        self.posture = mink.PostureTask(model, cost=cfg.posture_cost)
        q0 = np.zeros(model.nq)
        q0[kin.qadr] = np.radians(cfg.posture_target_deg)
        self.posture.set_target(q0)
        self.limits = [mink.ConfigurationLimit(model)]
        vl = {prefix + n: v for n, v in cfg.vel_limit_rad_s.items()}
        self.limits.append(mink.VelocityLimit(model, vl))
        pairs = default_collision_pairs(model, prefix, obstacle_geoms or [])
        self.collision = None
        if pairs:
            self.collision = mink.CollisionAvoidanceLimit(model, geom_pairs=pairs, minimum_distance_from_collisions=cfg.collision_min_dist,
                                                          collision_detection_distance=cfg.collision_detect_dist)
            self.limits.append(self.collision)
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site)
        # the site frame: x = opening, y = normal, z = approach (model.py); the grip triad is columns approach/opening/normal
        self.T = kin.T_grip
        self.site_offset_g0 = kin.cad.grip_point_g_m(0.0)

    def _target_se3(self, R_t: np.ndarray, p_t: np.ndarray, jaw: float) -> mink.SE3:
        # orientation of the site: R_world_site = R_t @ [o n a] (columns) = R_t @ P where P maps site axes to triad columns
        P = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])    # site x -> triad col 1 (opening), y -> col 2, z -> col 0
        R_site = R_t @ P
        delta_g = self.kin.cad.grip_point_g_m(np.degrees(jaw)) - self.site_offset_g0
        R_gl = R_t @ self.T.T
        p_site = p_t - R_gl @ delta_g
        from scipy.spatial.transform import Rotation as Rot
        q = Rot.from_matrix(R_site).as_quat()         # x y z w
        return mink.SE3(np.concatenate([[q[3], q[0], q[1], q[2]], p_site]))

    def set_costs(self, sig_p: float, sig_tilt_deg: float, sig_roll_deg: float):
        self.task.set_position_cost(1.0 / sig_p)
        st, sr = np.radians(sig_tilt_deg), np.radians(sig_roll_deg)
        self.task.set_orientation_cost(np.array([1.0 / st, 1.0 / st, 1.0 / sr]))   # site axes: x opening (tilt), y normal (tilt), z approach (roll)

    def step(self, q_arm: np.ndarray, jaw: float, R_t: np.ndarray, p_t: np.ndarray, sig_p: float, sig_tilt_deg: float, sig_roll_deg: float) -> np.ndarray:
        q = self.conf.q.copy()
        q[self.kin.qadr] = q_arm
        q[self.kin.jaw_qadr[0]], q[self.kin.jaw_qadr[1]] = jaw, -jaw
        self.conf.update(q)
        self.task.set_target(self._target_se3(R_t, p_t, jaw))
        self.set_costs(sig_p, sig_tilt_deg, sig_roll_deg)
        for _ in range(self.cfg.iterations):
            if self.cfg.project_lost:
                _, _, J = self.kin.grip_jac(self.conf.q[self.kin.qadr], jaw)
                _, _, lost_w = rotation_mobility(J)
                R_site = self.conf.data.site_xmat[self.sid].reshape(3, 3)
                self.task.set_lost(R_site.T @ lost_w)
            # each iteration gets a share of the frame's time step, so the velocity limit bounds the whole frame's motion
            dt = self.cfg.dt_integration / self.cfg.iterations
            dq = mink.solve_ik(self.conf, [self.task, self.posture], dt, self.cfg.solver, damping=self.cfg.damping, limits=self.limits)
            dq_full = np.zeros(self.model.nv)
            dq_full[self.kin.dofadr] = dq[self.kin.dofadr]
            self.conf.integrate_inplace(dq_full * self.cfg.gain, dt)
        return self.conf.q[self.kin.qadr].copy()


def solve_qp(model, kin: ArmKinematics, tg: Targets, cfg: QPConfig | None = None, q_init=None, prefix: str = "",
             obstacle_geoms: list | None = None, progress=None):
    from .ik_wls import Solution, solve_first
    cfg = cfg or QPConfig()
    solver = QPSolver(model, kin, cfg, prefix, obstacle_geoms)
    q = np.asarray(q_init, float) if q_init is not None else solve_first(kin, tg, 0).q
    out = np.zeros((tg.n, 5))
    for i in range(tg.n):
        q = solver.step(q, tg.jaw_rad[i], tg.R_target(i), tg.p_b[i], tg.sigma_p_m, tg.sig_tilt_deg[i], tg.sig_roll_deg[i])
        out[i] = q
        if progress and i % 200 == 0:
            progress(i, tg.n)
    return Solution("qp", out, tg.jaw_rad.copy(), np.zeros(tg.n), np.zeros(tg.n), np.zeros(tg.n), np.zeros(tg.n), np.zeros(tg.n),
                    tg.t.copy(), tg.phase.copy(), tg.hold_id.copy(), dict(iterations=cfg.iterations, collision=bool(solver.collision)))
