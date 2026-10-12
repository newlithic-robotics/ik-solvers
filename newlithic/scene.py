"""The cell as one MuJoCo model, in the DEMONSTRATION'S WORLD frame: the table, the screw plate with its screws and nuts,
and one or two SO-ARM101s with Newlithic grippers standing where their placements say.

Frame: the demonstration world (gravity aligned, z up, metres); the demonstration's reference surface (the plate top the
hands touched) is at z = demo.table_z, the bare table 10 mm lower, and each arm base is placed by Placement.T_world_base.

Objects (CAD: assets/cad/plate_tray, the 2026-09-27 Onshape export):
  plate   70 x 70 x 10 mm, nine d12 through holes; posed so that two holes sit on the first two picks of the lead hand
  bolt    M5 x 12 socket head (head d 8.5 x 5 mm), with an M5 nut (8 mm across flats x 4 mm) a couple of threads on;
          standing nut-down in a hole (the nut rests on the table through the d12 hole).  For physics the bolt is a free body
          and the nut rides on it with a hinge + slide coupled at the thread pitch (0.8 mm per turn).
Arms are attached with prefixes "left_" / "right_".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .cad import ASSETS, read_stl, rotz, write_stl, MESH_DIR
from .demo import Demo, Placement
from .model import ArmConfig, CONTACT_OBJECT, CONTACT_STRUCTURE, PLATE_THICKNESS, load_arm_spec

PLATE_HOLES = [(x, y) for x in (-0.023, 0.0, 0.023) for y in (-0.02588, -0.00048, 0.02492)]
BOLT = dict(head_r=0.00425, head_h=0.005, shank_r=0.0025, length=0.012, nut_af=0.008, nut_h=0.004, pitch=0.0008,
            nut_engaged=0.0016)      # nut top is `nut_engaged` up the thread when standing in the hole


def _quat(R):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def plate_pose_from_picks(p1: np.ndarray, p2: np.ndarray, z_top: float, holes=((-0.023, -0.02588), (0.0, -0.00048))):
    """T_world_plate (plate frame: bottom face at z 0, centre at the origin) with two holes on two pick points (xy)."""
    hA, hB = np.array(holes[0]), np.array(holes[1])
    d_plate, d_w = hB - hA, (p2 - p1)[:2]
    ang = np.degrees(np.arctan2(d_w[1], d_w[0]) - np.arctan2(d_plate[1], d_plate[0]))
    R = rotz(ang)
    mid_plate = R[:2, :2] @ (0.5 * (hA + hB))
    mid_w = 0.5 * (p1[:2] + p2[:2])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [mid_w[0] - mid_plate[0], mid_w[1] - mid_plate[1], z_top - PLATE_THICKNESS]
    return T, float(np.linalg.norm(d_w) - np.linalg.norm(d_plate)) * 1000


def _hex_prism(af, h):
    r = af / 2 / np.cos(np.pi / 6)
    a = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    bottom = np.column_stack([r * np.cos(a), r * np.sin(a), np.zeros(6)])
    top = bottom + [0, 0, h]
    V = np.vstack([bottom, top, [[0, 0, 0]], [[0, 0, h]]])
    F = []
    for k in range(6):
        a0, a1, b0, b1 = k, (k + 1) % 6, 6 + k, 6 + (k + 1) % 6
        F += [(a0, a1, b1), (a0, b1, b0), (12, a1, a0), (13, b0, b1)]
    return V, np.array(F)


@dataclass
class SceneConfig:
    arms: dict = field(default_factory=dict)     # name -> Placement
    plate: bool = True
    screws: bool = True
    physics_objects: bool = False                # screws as free bodies with a threaded nut (else static props)
    tray: bool = False
    arm_cfg: ArmConfig = field(default_factory=ArmConfig)
    table_size: tuple = (0.9, 0.9)
    table_centre: tuple = (0.05, 0.1)


@dataclass
class Scene:
    model: mujoco.MjModel
    spec: mujoco.MjSpec
    cfg: SceneConfig
    demo: Demo
    plate_T: np.ndarray | None
    screw_xy: list
    obstacle_geoms: list          # names of the static obstacle geoms (table, plate, tray)
    names: dict


def build_scene(demo: Demo, cfg: SceneConfig, lead_hand: str = "left") -> Scene:
    spec = mujoco.MjSpec()
    spec.modelname = "newlithic_cell"
    spec.meshdir = str(MESH_DIR)
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1440
    spec.visual.headlight.ambient[:] = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse[:] = [0.6, 0.6, 0.6]
    spec.visual.map.znear = 0.005
    world = spec.worldbody
    world.add_light(pos=[0.5, -0.5, 1.2], dir=[-0.4, 0.4, -1.0], diffuse=[0.5, 0.5, 0.5], castshadow=True)
    world.add_light(pos=[-0.6, 0.6, 1.0], dir=[0.5, -0.5, -1.0], diffuse=[0.35, 0.35, 0.35], castshadow=False)
    z_table = demo.table_z - PLATE_THICKNESS
    obstacles = []
    sx, sy = cfg.table_size
    both = CONTACT_STRUCTURE | CONTACT_OBJECT
    g = world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, 0.01], pos=[cfg.table_centre[0], cfg.table_centre[1], z_table - 0.01],
                       rgba=[0.2, 0.2, 0.22, 1.0], friction=[0.8, 0.005, 0.0001], contype=both, conaffinity=both)
    obstacles.append("table")
    names = {}
    plate_T, screw_xy = None, []
    h = demo.hand(lead_hand)
    # the picks: the grip origin at the moment of CONTACT (the aperture event, not the first 'hold' frame 0.3 s later, by which
    # time the hand has already moved a millimetre or two)
    picks = [h.p[h.events[k]["contact"]] for k in h.holds() if h.height[h.contact_frame(k)] < 0.06 and -h.approach[h.contact_frame(k), 2] > 0.85]
    if cfg.plate and len(picks) >= 2:
        plate_T, mismatch = plate_pose_from_picks(picks[0], picks[1], demo.table_z)
        V, F = read_stl(ASSETS / "cad" / "plate_tray" / "Part_1.stl")
        V = V / 1000.0
        pf = MESH_DIR / "plate.stl"
        if not pf.exists():
            write_stl(pf, V, F)
        m = spec.add_mesh(name="m_plate")
        m.file = "plate.stl"
        b = world.add_body(name="plate", pos=plate_T[:3, 3].tolist(), quat=_quat(plate_T[:3, :3]))
        # visual: the CAD mesh; collision: a convex hull would close the holes, so each of the nine cells is four box strips
        # around a 12 mm square hole (the nut passes, the screw stands on the table through it)
        b.add_geom(name="plate", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m_plate", rgba=[0.93, 0.93, 0.9, 1.0], contype=0, conaffinity=0)
        xs, ys = sorted(set(x for x, _ in PLATE_HOLES)), sorted(set(y for _, y in PLATE_HOLES))
        xb = [-0.035] + [(xs[i] + xs[i + 1]) / 2 for i in range(2)] + [0.035]
        yb = [-0.035] + [(ys[i] + ys[i + 1]) / 2 for i in range(2)] + [0.035]
        hh, t = 0.006, PLATE_THICKNESS
        k = 0
        for i, cx in enumerate(xs):
            for j, cy in enumerate(ys):
                x0, x1, y0, y1 = xb[i], xb[i + 1], yb[j], yb[j + 1]
                strips = [((x0 + x1) / 2, (y1 + cy + hh) / 2, (x1 - x0) / 2, (y1 - cy - hh) / 2), ((x0 + x1) / 2, (y0 + cy - hh) / 2, (x1 - x0) / 2, (cy - hh - y0) / 2),
                          ((x0 + cx - hh) / 2, cy, (cx - hh - x0) / 2, hh), ((x1 + cx + hh) / 2, cy, (x1 - cx - hh) / 2, hh)]
                for (sx_, sy_, hx, hy) in strips:
                    if hx > 1e-4 and hy > 1e-4:
                        b.add_geom(name=f"plate_c{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, t / 2], pos=[sx_, sy_, t / 2],
                                   rgba=[0.9, 0.9, 0.85, 0.0], group=3, contype=both, conaffinity=both)
                        obstacles.append(f"plate_c{k}")
                        k += 1
        names["plate_mismatch_mm"] = mismatch
        hole_w = [plate_T[:3, :3] @ np.array([x, y, 0.0]) + plate_T[:3, 3] for x, y in PLATE_HOLES]
        screw_xy = [hole_w[0][:2], hole_w[4][:2]]
    if cfg.tray:
        V, F = read_stl(ASSETS / "cad" / "plate_tray" / "Part_2.stl")
        V = V / 1000.0
        tf = MESH_DIR / "tray.stl"
        if not tf.exists():
            write_stl(tf, V - V.mean(0) * [1, 1, 0], F)
        m = spec.add_mesh(name="m_tray")
        m.file = "tray.stl"
        if plate_T is not None:
            pos = plate_T[:3, :3] @ np.array([0.104, 0.0, 0.0]) + plate_T[:3, 3]
            b = world.add_body(name="tray", pos=pos.tolist(), quat=_quat(plate_T[:3, :3]))
            b.add_geom(name="tray", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m_tray", rgba=[0.93, 0.93, 0.9, 1.0])
            obstacles.append("tray")
    if cfg.screws and screw_xy:
        Vn, Fn = _hex_prism(BOLT["nut_af"], BOLT["nut_h"])
        m = spec.add_mesh(name="m_nut")
        m.uservert = Vn.ravel().tolist()
        m.userface = Fn.ravel().tolist()
        for k, (x, y) in enumerate(screw_xy):
            # standing in the hole: the nut rests on the table, nut bottom at z_table; the bolt's thread tip is at the nut's
            # bottom + engagement, its head above the free shank
            z0 = z_table
            bolt = world.add_body(name=f"bolt{k}", pos=[x, y, z0 + BOLT["nut_h"] - BOLT["nut_engaged"]])
            if cfg.physics_objects:
                bolt.add_freejoint(name=f"bolt{k}_free")
            bolt.add_geom(name=f"bolt{k}_shank", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BOLT["shank_r"], BOLT["length"] / 2, 0],
                          pos=[0, 0, BOLT["length"] / 2], rgba=[0.2, 0.2, 0.2, 1], mass=0.002 if cfg.physics_objects else None,
                          contype=CONTACT_OBJECT, conaffinity=CONTACT_OBJECT)
            bolt.add_geom(name=f"bolt{k}_head", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BOLT["head_r"], BOLT["head_h"] / 2, 0],
                          pos=[0, 0, BOLT["length"] + BOLT["head_h"] / 2], rgba=[0.08, 0.08, 0.08, 1], mass=0.0025 if cfg.physics_objects else None,
                          condim=4, friction=[1.0, 0.01, 0.0001], contype=CONTACT_OBJECT, conaffinity=CONTACT_OBJECT)
            nut = bolt.add_body(name=f"nut{k}", pos=[0, 0, BOLT["nut_engaged"] - BOLT["nut_h"]])
            if cfg.physics_objects:
                nut.add_joint(name=f"nut{k}_turn", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=[1e-5, 0, 0])
                nut.add_joint(name=f"nut{k}_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                              range=[-BOLT["length"], 0.0], damping=[1e-4, 0, 0])
                spec.add_equality(name=f"nut{k}_thread", type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                                  name1=f"nut{k}_slide", name2=f"nut{k}_turn", data=[0.0, -BOLT["pitch"] / (2 * np.pi), 0, 0, 0] + [0.0] * 6)
            nut.add_geom(name=f"nut{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m_nut", rgba=[0.75, 0.75, 0.78, 1],
                         mass=0.0015 if cfg.physics_objects else None, condim=4, friction=[1.0, 0.01, 0.0001], contype=CONTACT_OBJECT, conaffinity=CONTACT_OBJECT)
        names["screws"] = [f"bolt{k}" for k in range(len(screw_xy))]
    # ------------------------------------------------------------ the arms
    for name, pl in cfg.arms.items():
        arm_spec, cad, parts = load_arm_spec(cfg.arm_cfg)
        T = pl.T_world_base(demo.table_z)
        frame = world.add_frame(name=f"{name}_base_frame", pos=T[:3, 3].tolist(), quat=_quat(T[:3, :3]))
        spec.attach(arm_spec, prefix=f"{name}_", frame=frame)
        names[f"{name}_T_world_base"] = T
    # grasp emulation: an (inactive) weld between every gripper and every object it may hold; the replay switches them on
    # when the pads close on the object during a demonstrated hold and off when they open (see replay.GraspEmulator)
    if cfg.physics_objects and cfg.screws:
        names["welds"] = {}
        for name in cfg.arms:
            for k in range(len(screw_xy)):
                for obj in (f"bolt{k}", f"nut{k}"):
                    # the nut welds are softer: with the bolt in the other gripper, the nut can only turn and slide on its thread,
                    # so the two grippers' relative-pose residual has to be absorbed somewhere - here, not at the bolt's pads
                    e = spec.add_equality(name=f"weld_{name}_{obj}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                          name1=f"{name}_gripper_link", name2=obj, active=False,
                                          solref=[0.004, 1.0] if obj.startswith("bolt") else [0.03, 1.0])
                    names["welds"][(name, obj)] = e.name
    model = spec.compile()
    return Scene(model, spec, cfg, demo, plate_T, screw_xy, obstacles, names)


def save_scene(scene: Scene, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scene.spec.to_xml(), encoding="utf-8")
    return path
