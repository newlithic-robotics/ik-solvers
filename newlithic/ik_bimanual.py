"""Solver 4 - the two arms as ONE quadratic programme ("bimanual QP").

Ten joints, one QP per frame, in the cell model (both arms, table, plate), so that
  * the RIGHT gripper is asked for its pose RELATIVE to the left gripper whenever the left holds the screw in the air
    (mink RelativeFrameTask): the unscrewing is a relation between the two grippers, not two absolute trajectories, and the
    estimator's cross-rig error lives in the absolute poses, not in that relation;
  * arm-arm, arm-table and self collisions are hard constraints (CollisionAvoidanceLimit over both arms' geoms);
  * joint and velocity limits of both arms hold at once.
Outside those frames both grippers are asked for their absolute demonstrated poses (the lost axis projected as in the
single-arm QP).  The roll of the right gripper at each of its in-air grasps is free (the nut is grabbed wherever the arm
finds it comfortable; only the turning after the grasp is followed), via the same free-roll pass as the single-arm solvers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .demo import PHASE, Demo, Placement
from .ik_qp import GripTask, QPConfig, arm_geom_names
from .ik_wls import Solution, free_roll_shift, solve_first, solve_wls
from .kin import ArmKinematics, rotation_mobility
from .model import JOINTS
from .scene import Scene
from .targets import Targets, Tolerances, make_targets

P_SITE = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])    # site axes (x opening, y normal, z approach) <- triad cols


def _se3(R, p) -> mink.SE3:
    q = Rot.from_matrix(R).as_quat()
    return mink.SE3(np.concatenate([[q[3], q[0], q[1], q[2]], p]))


@dataclass
class BimanualConfig(QPConfig):
    relative_position_cost: float = 1.0 / 0.0005
    relative_orientation_cost: float = 1.0 / np.radians(4.0)
    follower_abs_position_cost: float = 1.0 / 0.01        # while relative: the right's absolute pose only weakly
    follower_abs_orientation_cost: float = 1.0 / np.radians(30.0)
    air_height_m: float = 0.06
    arm_pair_min_dist: float = 0.004


class BimanualQP:
    def __init__(self, scene: Scene, kins: dict, cfg: BimanualConfig):
        self.scene, self.model, self.cfg = scene, scene.model, cfg
        self.kins = kins                                   # "left"/"right" -> ArmKinematics(scene.model, prefix)
        self.conf = mink.Configuration(self.model)
        self.tasks = {}
        for name, kin in kins.items():
            self.tasks[name] = GripTask(f"{name}_grip", kin, position_cost=1.0, orientation_cost=1.0, lm_damping=cfg.lm_damping,
                                        project_lost=cfg.project_lost)
        self.rel = mink.RelativeFrameTask("right_grip", "site", "left_grip", "site", position_cost=cfg.relative_position_cost,
                                          orientation_cost=cfg.relative_orientation_cost, lm_damping=cfg.lm_damping)
        self.posture = mink.PostureTask(self.model, cost=cfg.posture_cost)
        q0 = np.zeros(self.model.nq)
        for kin in kins.values():
            q0[kin.qadr] = np.radians(cfg.posture_target_deg)
        self.posture.set_target(q0)
        limits = [mink.ConfigurationLimit(self.model)]
        vl = {}
        for name in kins:
            vl.update({f"{name}_{n}": v for n, v in cfg.vel_limit_rad_s.items()})
        limits.append(mink.VelocityLimit(self.model, vl))
        pairs = []
        for name in kins:
            grip = arm_geom_names(self.model, f"{name}_", ("gripper_link", "jaw_L", "jaw_R"))
            lower = arm_geom_names(self.model, f"{name}_", ("base_link", "shoulder_link", "upper_arm_link"))
            whole = arm_geom_names(self.model, f"{name}_", ("upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "jaw_L", "jaw_R"))
            pairs.append((grip, lower))
            if scene.obstacle_geoms:
                pairs.append((whole, list(scene.obstacle_geoms)))
        if len(kins) == 2:
            a = arm_geom_names(self.model, "left_", ("lower_arm_link", "wrist_link", "gripper_link", "jaw_L", "jaw_R"))
            b = arm_geom_names(self.model, "right_", ("lower_arm_link", "wrist_link", "gripper_link", "jaw_L", "jaw_R"))
            pairs.append((a, b))
        self.collision = mink.CollisionAvoidanceLimit(self.model, geom_pairs=pairs, minimum_distance_from_collisions=cfg.arm_pair_min_dist,
                                                      collision_detection_distance=cfg.collision_detect_dist)
        limits.append(self.collision)
        self.limits = limits
        self.sids = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{n}_grip") for n in kins}

    def site_target(self, kin: ArmKinematics, R_t, p_t, jaw):
        delta_g = kin.cad.grip_point_g_m(np.degrees(jaw)) - kin.cad.grip_point_g_m(0.0)
        R_gl = R_t @ kin.T_grip.T
        return R_t @ P_SITE, p_t - R_gl @ delta_g

    def step(self, q: dict, jaw: dict, targets: dict, relative: bool, costs: dict) -> dict:
        """q/jaw: name -> arm joints / jaw angle; targets: name -> (R_t, p_t) in the world; costs: name -> (sig_p, sig_tilt, sig_roll)."""
        qq = self.conf.q.copy()
        for n, kin in self.kins.items():
            qq[kin.qadr] = q[n]
            qq[kin.jaw_qadr[0]], qq[kin.jaw_qadr[1]] = jaw[n], -jaw[n]
        self.conf.update(qq)
        site = {}
        for n, kin in self.kins.items():
            R_t, p_t = targets[n]
            Rs, ps = self.site_target(kin, R_t, p_t, jaw[n])
            site[n] = (Rs, ps)
            self.tasks[n].set_target(_se3(Rs, ps))
            sig_p, st, sr = costs[n]
            if relative and n == "right":
                self.tasks[n].set_position_cost(self.cfg.follower_abs_position_cost)
                self.tasks[n].set_orientation_cost(np.full(3, self.cfg.follower_abs_orientation_cost))
            else:
                self.tasks[n].set_position_cost(1.0 / sig_p)
                self.tasks[n].set_orientation_cost(np.array([1.0 / np.radians(st), 1.0 / np.radians(st), 1.0 / np.radians(sr)]))
        tasks = list(self.tasks.values()) + [self.posture]
        if relative and "right" in site and "left" in site:
            RL, pL = site["left"]
            RR, pR = site["right"]
            R_rel = RL.T @ RR
            p_rel = RL.T @ (pR - pL)
            self.rel.set_target(_se3(R_rel, p_rel))
            tasks.append(self.rel)
        for _ in range(self.cfg.iterations):
            if self.cfg.project_lost:
                for n, kin in self.kins.items():
                    _, _, J = kin.grip_jac(self.conf.q[kin.qadr], jaw[n])
                    _, _, lost_w = rotation_mobility(J)
                    R_site = self.conf.data.site_xmat[self.sids[n]].reshape(3, 3)
                    self.tasks[n].set_lost(R_site.T @ lost_w)
            dt = self.cfg.dt_integration / self.cfg.iterations          # the velocity limit then bounds the whole frame's motion
            dq = mink.solve_ik(self.conf, tasks, dt, self.cfg.solver, damping=self.cfg.damping, limits=self.limits)
            dq_full = np.zeros(self.model.nv)
            for kin in self.kins.values():
                dq_full[kin.dofadr] = dq[kin.dofadr]
            self.conf.integrate_inplace(dq_full * self.cfg.gain, dt)
        return {n: self.conf.q[kin.qadr].copy() for n, kin in self.kins.items()}


def world_targets(demo: Demo, hand: str, cad, tol: Tolerances, free_roll=False) -> Targets:
    """Targets in the demonstration world itself (identity placement)."""
    return make_targets(demo, hand, Placement(0.0, 0.0, 0.0, demo.table_z), cad, tol, free_roll=free_roll)


def solve_bimanual(scene: Scene, demo: Demo, cad, kins: dict, cfg: BimanualConfig | None = None, tol: Tolerances | None = None,
                   single_kins: dict | None = None, progress=None) -> dict:
    """Returns name -> Solution (joint trajectories in each arm's own joint space; the targets used are in the world)."""
    cfg = cfg or BimanualConfig()
    tol = tol or Tolerances()
    tg = {n: world_targets(demo, n, cad, tol) for n in kins}
    # free-roll pass per hand with the single-arm per-frame solver in each arm's base frame (fast), then the shifts carried over
    if single_kins:
        for n, skin in single_kins.items():
            pl = scene.cfg.arms[n]
            tgf = make_targets(demo, n, pl, cad, tol, free_roll=True)
            s1 = solve_wls(skin, tgf, progress=progress)
            tg[n].roll_shift = free_roll_shift(skin, tgf, s1)
    left, right = demo.hand("left"), demo.hand("right")
    n = tg["left"].n
    # relative frames: the left holds something in the air
    rel = (left.phase == PHASE["hold"]) & (left.height > cfg.air_height_m)
    # interpolate the right hand's targets onto the left clock (both are on the same clock, same frame count here)
    solver = BimanualQP(scene, kins, cfg)
    # initial joints: per-frame multi-seed solves in the world model
    q = {}
    for name, kin in kins.items():
        q[name] = solve_first(kin, tg[name], 0).q
    out = {name: np.zeros((n, 5)) for name in kins}
    for i in range(n):
        targets = {name: (tg[name].R_target(i), tg[name].p_b[i]) for name in kins}
        costs = {name: (tg[name].sigma_p_m, tg[name].sig_tilt_deg[i], tg[name].sig_roll_deg[i]) for name in kins}
        jaw = {name: tg[name].jaw_rad[i] for name in kins}
        q = solver.step(q, jaw, targets, bool(rel[i]) and len(kins) == 2, costs)
        for name in kins:
            out[name][i] = q[name]
        if progress and i % 200 == 0:
            progress(i, n)
    sols = {}
    for name in kins:
        t = tg[name]
        sols[name] = Solution("bimanual_qp", out[name], t.jaw_rad.copy(), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
                              t.t.copy(), t.phase.copy(), t.hold_id.copy(), dict(relative_frames=int(rel.sum())))
    return sols, tg, rel
