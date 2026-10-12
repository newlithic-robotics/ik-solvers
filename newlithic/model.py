"""The SO-ARM101 with the Newlithic gripper as a MuJoCo model, built from lerobot's URDF and the CAD.

What is taken from the URDF: the five arm joints, their limits, the links' meshes and inertials (onshape-to-robot export).
What is replaced: everything hanging off the wrist-roll flange (`gripper_link`): the stock jaw servo, the stock gripper body,
the moving jaw and its joint.  What is added there: the Newlithic parts (retarget.cad.PARTS) as mesh geoms with their masses,
two jaw bodies on hinges (REV_L = q, REV_R = -q, coupled by an equality), the jaw servo, the X3 camera (as a mass and as a
MuJoCo camera looking along the finger-side lens axis), and sites:

  flange   the gripper_link origin (wrist-roll flange face, z toward the wrist)
  grip     the grip origin at the closed pose (reference; the true grip point moves with the jaw angle - use pad_L/pad_R)
  tips     midpoint of the pad tips at the closed pose (the pipeline's "TCP")
  pad_L/R  the pad contact-strip centres, on the jaw bodies (midpoint = the grip origin at any jaw angle)
  camera   the X3 centre

Actuators: position servos on every joint (kp from the servo's stiffness, force range = the STS3215 stall torque at the
configured voltage); the two jaws share one actuator through the coupling.

Nothing here is specific to one episode; the scene (cell, second arm, objects) is built on top of this in scene.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .cad import ASSETS, LEROBOT_URDF, MESH_DIR, GripperCAD, Part, load_rig, prepare_meshes

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
JAW_JOINTS = ("jaw_L", "jaw_R")
MODEL_DIR = ASSETS / "model"
CONTACT_STRUCTURE, CONTACT_OBJECT = 1, 2     # contype/conaffinity bits: arm + cell structure, manipulated objects
PAD_FACE_HALF = (0.001, 0.0065, 0.014)     # half sizes of a pad's flat contact face: thickness, along the finger, across (m)
PAD_FACE_SHIFT = 0.003                     # the face is centred this far from the strip centre toward the tips (it runs almost to the tips)
TABLE_Z_BASE = -0.01755     # the bare table top in the base frame as measured on the real arm in 2026-09 (the CAD base bottom is
PLATE_THICKNESS = 0.010     # at -2.4 mm: the real arm stands ~15 mm higher than the URDF origin suggests; re-measure on the new arm)
STS3215 = {  # Feetech STS3215 (vendor listings, 2026-10): stall torque, no-load speed, rated torque
    "12V": dict(stall_Nm=30.0 * 0.0980665, rated_Nm=10.0 * 0.0980665, speed_rad_s=np.radians(60.0 / 0.222)),
    "7.4V": dict(stall_Nm=16.5 * 0.0980665, rated_Nm=6.0 * 0.0980665, speed_rad_s=np.radians(60.0 / 0.25)),
}


@dataclass
class ArmConfig:
    voltage: str = "12V"
    kp: float = 70.0            # N m / rad of the position servo: ~P 48 on the real STS3215 (3.3 mm tool droop at P 64, 11 at P 16
    kv: float = 1.0             #   were measured on the stock arm, 2026-09); stall torque at 2.4 deg of error
    jaw_kp: float = 40.0        # the jaw servo on its 95 mm lever: 1 mm of pad squeeze = 0.6 deg = 0.4 N m = 4 N per pad
    jaw_kv: float = 0.4
    use_urdf_limits: bool = True
    collision_meshes: bool = True
    name: str = "so101_newlithic"
    extra: dict = field(default_factory=dict)


def _urdf_with_extension(tmp: Path) -> Path:
    """lerobot's URDF with the MuJoCo extension, and a dummy root above base_link so that base_link stays a body of its
    own (the importer otherwise merges the root link into the world, and its servo then collides with the shoulder)."""
    urdf = LEROBOT_URDF.read_text(encoding="utf-8")
    meshdir = str(MESH_DIR).replace("\\", "/")
    ext = (f'<mujoco><compiler meshdir="{meshdir}" balanceinertia="true" discardvisual="true" strippath="true" fusestatic="false"/></mujoco>'
           '<link name="root"/><joint name="root_fixed" type="fixed"><parent link="root"/><child link="base_link"/></joint>')
    k = urdf.index(">", urdf.index("<robot")) + 1
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(urdf[:k] + ext + urdf[k:], encoding="utf-8")
    return tmp


def _quat_wxyz(R: np.ndarray) -> list:
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _frame_quat(z_axis: np.ndarray, x_hint: np.ndarray) -> list:
    z = np.asarray(z_axis, float); z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, float); x = x - z * (x @ z); x /= np.linalg.norm(x)
    return _quat_wxyz(np.stack([x, np.cross(z, x), z], axis=1))


def _inertial_from_parts(parts: list[Part]):
    """(mass, ipos, iquat, diagonal inertia) of one body made of these mesh parts, computed by MuJoCo itself."""
    s = mujoco.MjSpec()
    s.meshdir = str(MESH_DIR)
    b = s.worldbody.add_body(name="b")
    b.add_joint(name="j", type=mujoco.mjtJoint.mjJNT_HINGE)
    for p in parts:
        m = s.add_mesh(name="m_" + p.name)
        m.file = p.file
        b.add_geom(name="g_" + p.name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=m.name, mass=p.mass_kg)
    mod = s.compile()
    return float(mod.body_mass[1]), mod.body_ipos[1].copy(), mod.body_iquat[1].copy(), mod.body_inertia[1].copy()


def load_arm_spec(cfg: ArmConfig | None = None) -> tuple[mujoco.MjSpec, GripperCAD, list[Part]]:
    """The one-arm spec (base_link at the origin), not yet compiled."""
    cfg = cfg or ArmConfig()
    cad = load_rig()
    parts = prepare_meshes(cad)
    tmp = MODEL_DIR / "so_arm101_mj.urdf"
    _urdf_with_extension(tmp)
    try:
        spec = mujoco.MjSpec.from_file(str(tmp))
    except Exception:
        model = mujoco.MjModel.from_xml_path(str(tmp))
        mjcf = MODEL_DIR / "so_arm101_stock.xml"
        mujoco.mj_saveLastXML(str(mjcf), model)
        spec = mujoco.MjSpec.from_file(str(mjcf))
    spec.modelname = cfg.name
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1440
    spec.visual.headlight.ambient[:] = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse[:] = [0.6, 0.6, 0.6]
    spec.compiler.balanceinertia = True
    spec.compiler.discardvisual = False
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO   # the URDF importer pins explicit inertials
    spec.meshdir = str(MESH_DIR)                      # our meshes by file; the URDF ones were loaded with their own dir

    # ------------------------------------------------------------ strip the stock gripper
    grip = spec.body("gripper_link")
    for g in list(grip.geoms):
        spec.delete(g)
    for b in list(grip.bodies):
        spec.delete(b)                                # moving jaw (and its joint)
    for b in list(spec.bodies):
        if b.name == "gripper_frame_link":
            spec.delete(b)
    # the URDF importer leaves an explicit inertial that the compiler keeps even when told not to: replace it with the one
    # MuJoCo computes from the new parts (a throw-away spec with just those geoms)
    mass, ipos, iquat, inertia = _inertial_from_parts([p for p in parts if p.body == "gripper_link"])
    grip.explicitinertial = True
    grip.mass, grip.ipos = mass, ipos
    grip.iquat, grip.inertia = np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)
    Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, iquat); Rm = Rm.reshape(3, 3)
    I = Rm @ np.diag(inertia) @ Rm.T                  # the URDF importer filled fullinertia; keep using that field
    grip.fullinertia = np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])
    for m in list(spec.meshes):                       # the stock gripper's meshes are no longer referenced
        if m.name in ("moving_jaw_so101_v1", "wrist_roll_follower_so101_v1"):
            spec.delete(m)

    # ------------------------------------------------------------ the Newlithic parts
    for p in parts:
        m = spec.add_mesh(name=f"m_{p.name}".replace(" ", "_").replace("(", "").replace(")", ""))
        m.file = p.file
        m.maxhullvert = 128
        body = grip if p.body == "gripper_link" else None
        if body is None:
            body = next((b for b in grip.bodies if b.name == p.body), None)
            if body is None:
                body = grip.add_body(name=p.body, pos=cad.pivot_g_m(p.body[-1]).tolist())
                lo, hi = (0.0, cad.open_stop_deg) if p.body == "jaw_L" else (-cad.open_stop_deg, 0.0)
                body.add_joint(name=p.body, type=mujoco.mjtJoint.mjJNT_HINGE, axis=cad.hinge_axis_g.tolist(),
                               range=[np.radians(lo), np.radians(hi)], armature=1e-4, damping=[0.01, 0.0, 0.0])
                body.add_site(name=f"pad_{p.body[-1]}", pos=cad.pad_site_local_m(p.body[-1]).tolist(), size=[0.002, 0.002, 0.002],
                              rgba=[0.1, 0.9, 0.2, 1], group=1)
        geom = body.add_geom(name=f"g_{p.name}".replace(" ", "_").replace("(", "").replace(")", ""), type=mujoco.mjtGeom.mjGEOM_MESH,
                             meshname=m.name, rgba=list(p.rgba), mass=p.mass_kg)
        geom.contype, geom.conaffinity = (CONTACT_STRUCTURE, CONTACT_STRUCTURE) if cfg.collision_meshes else (0, 0)
    # the pads' contact faces: a thin flat box at each contact strip, the only part of the gripper that touches the manipulated
    # objects (the convex hull of a curved pad would touch a screw head along one edge; the real TPU pad flattens onto it)
    Rbox = _quat_wxyz(cad.R_Dg.T)                       # box axes = D axes (x opening, y along the finger, z across the pad)
    for jaw, sgn in (("L", 1.0), ("R", -1.0)):
        body = next(b for b in grip.bodies if b.name == f"jaw_{jaw}")
        c = cad.pad_site_local_m(jaw)
        off = cad.vec_D_to_g([-sgn * PAD_FACE_HALF[0], PAD_FACE_SHIFT, 0.0])   # face flush with the strip, box inside the pad, toward the tips
        g = body.add_geom(name=f"pad_face_{jaw}", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(PAD_FACE_HALF), pos=(c + off).tolist(), quat=Rbox,
                          rgba=[0.2, 0.2, 0.2, 1.0], friction=[1.2, 0.02, 0.0002], condim=4, priority=1, solref=[0.004, 1.0])
        g.contype, g.conaffinity = CONTACT_STRUCTURE | CONTACT_OBJECT, CONTACT_STRUCTURE | CONTACT_OBJECT
        g.mass = 0.0
    # the two jaws overlap at the gear teeth (meshing gears): never a collision
    spec.add_exclude(name="jaws", bodyname1="jaw_L", bodyname2="jaw_R")
    # base_link is static (welded to the world), so MuJoCo's parent-child filter does not cover its servo inside the shoulder
    spec.add_exclude(name="base_shoulder", bodyname1="base_link", bodyname2="shoulder_link")
    spec.add_equality(name="jaw_couple", type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT, name1="jaw_R", name2="jaw_L",
                      data=[0.0, -1.0, 0.0, 0.0, 0.0] + [0.0] * 6)

    # ------------------------------------------------------------ sites and the camera
    T = cad.grip_triad_g()                             # columns approach, opening, normal (in g)
    q_grip = _quat_wxyz(np.stack([T[:, 1], T[:, 2], T[:, 0]], axis=1))   # site frame: x = opening, y = normal, z = approach
    grip.add_site(name="flange", pos=[0, 0, 0], size=[0.003] * 3, rgba=[0.9, 0.2, 0.2, 1], group=1)
    grip.add_site(name="grip", pos=cad.grip_point_g_m(0.0).tolist(), quat=q_grip, size=[0.0025] * 3, rgba=[0.2, 0.4, 1.0, 1], group=1)
    grip.add_site(name="tips", pos=cad.tips_g_m().tolist(), quat=q_grip, size=[0.002] * 3, rgba=[1.0, 0.3, 0.9, 1], group=1)
    c, lens, body_axis = cad.camera_g_m()
    grip.add_site(name="camera", pos=c.tolist(), quat=_frame_quat(lens, body_axis), size=[0.004] * 3, rgba=[0.2, 0.9, 0.9, 1], group=1)
    # MuJoCo cameras look along their -z; put -z on the finger-side lens axis, y "up" along the camera body
    grip.add_camera(name="x3_fingers", pos=c.tolist(), quat=_frame_quat(-lens, np.cross(body_axis, -lens)), fovy=110)

    # ------------------------------------------------------------ joints: limits, actuators
    servo = STS3215[cfg.voltage]
    for j in spec.joints:
        if j.name in JOINTS:
            j.armature = 2e-4
            j.damping = [0.05, 0.0, 0.0]
            j.frictionloss = 0.02
    for name in JOINTS:
        a = spec.add_actuator(name=f"act_{name}", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=cfg.kp, kv=cfg.kv)
        a.forcerange = [-servo["stall_Nm"], servo["stall_Nm"]]
        a.ctrlrange = list(spec.joint(name).range)
        a.inheritrange = 0.0
    a = spec.add_actuator(name="act_jaw", target="jaw_R", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.set_to_position(kp=cfg.jaw_kp, kv=cfg.jaw_kv)
    a.forcerange = [-servo["stall_Nm"], servo["stall_Nm"]]
    a.ctrlrange = [-np.radians(cad.open_stop_deg), 0.0]
    return spec, cad, parts


def compile_arm(cfg: ArmConfig | None = None, save: bool = True):
    spec, cad, parts = load_arm_spec(cfg)
    model = spec.compile()
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        (MODEL_DIR / "so101_newlithic.xml").write_text(spec.to_xml(), encoding="utf-8")
    return model, spec, cad, parts


# ---------------------------------------------------------------------------- queries on a compiled model
def joint_ids(model: mujoco.MjModel, prefix: str = "") -> dict:
    out = {}
    for n in JOINTS + JAW_JOINTS:
        out[n] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)
    return out


def set_arm(model: mujoco.MjModel, data: mujoco.MjData, q_deg, jaw_deg: float, prefix: str = ""):
    """Joint angles [deg] (URDF zero = lerobot calibrated zero) and the jaw opening q [deg] (REV_L)."""
    ids = joint_ids(model, prefix)
    for n, v in zip(JOINTS, q_deg):
        data.qpos[model.jnt_qposadr[ids[n]]] = np.radians(v)
    data.qpos[model.jnt_qposadr[ids["jaw_L"]]] = np.radians(jaw_deg)
    data.qpos[model.jnt_qposadr[ids["jaw_R"]]] = -np.radians(jaw_deg)


def base_bottom_z(model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "") -> float:
    """The lowest point of base_link's meshes in its own frame: where the arm stands on the table."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link")
    zmin = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        V = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        R = model.geom_quat[g]
        M = np.zeros(9); mujoco.mju_quat2Mat(M, R)
        P = V @ M.reshape(3, 3).T + model.geom_pos[g]
        zmin = min(zmin, P[:, 2].min())
    return float(zmin)


def gravity_torque(model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "") -> dict:
    """Joint torque [N m] that holds the current pose still against gravity (qfrc_bias at zero velocity), per arm joint."""
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    ids = joint_ids(model, prefix)
    return {n: float(data.qfrc_bias[model.jnt_dofadr[ids[n]]]) for n in JOINTS + JAW_JOINTS}


def contact_pairs(model: mujoco.MjModel, data: mujoco.MjData) -> list:
    """(body1, body2, geom1, geom2, penetration mm) for every contact in data."""
    out = []
    for c in data.contact[:data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        out.append((model.body(b1).name, model.body(b2).name, model.geom(c.geom1).name, model.geom(c.geom2).name, round(-c.dist * 1000, 2)))
    return out


def mass_summary(model: mujoco.MjModel, prefix: str = "") -> str:
    names = [prefix + n for n in ("base_link", "shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "jaw_L", "jaw_R")]
    lines = []
    tot = 0.0
    for n in names:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if b < 0:
            continue
        tot += model.body_mass[b]
        lines.append(f"  {n:16s} {model.body_mass[b]*1000:7.1f} g   com (body frame) {np.round(model.body_ipos[b]*1000, 1)} mm")
    lines.append(f"  total {tot*1000:.0f} g")
    return "\n".join(lines)
