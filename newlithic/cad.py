"""The Newlithic (UMI-style) gripper as mounted on the SO-ARM101, read from the CAD.

Sources (all in `assets/`):
  * gripper_cad_arm_rig.yaml      measured in Fusion on the document "robot arm newlithic gripper on SO-ARM101" (2026-10-10):
                                  the transform between the document frame D and the URDF gripper_link frame g, the jaw
                                  pivots, the pad contact-strip polar law, the camera, the jaw servo.
  * cad/arm_gripper/*.stl         every solid of that document tessellated (tools/step_to_meshes.py), document frame D, mm,
                                  jaws CLOSED (q = 0).
  * masses.yaml                   what each part weighs, or the density to derive it from its CAD volume.

Frames
  D   Fusion document frame (mm): fingertips toward +y, camera toward +z, the L jaw on the -x side.
  g   URDF gripper_link frame (m): origin on the wrist-roll axis at the servo flange face, z toward the wrist servo, so the
      gripper hangs along -z.  p_D = R_Dg p_g + t_Dg.
  jaw frames   origin on the jaw's pivot axis, axes parallel to g; the hinge axis is z_D expressed in g.

Jaw law  the pads are on arms pivoting about the two jaw axes (30 mm apart); the pad contact-strip centre of jaw j sits at
radius r_j from its pivot at polar angle phi0_j +/- q.  The GAP between the two strip centres is what the UMI pipeline calls
the aperture; it grows 3.3 mm per degree of opening (0.18 mm closed, 22.4 mm in the docking jig at q = 6.69 deg, 144 mm at the
45.6 deg stop).

Grip frame  origin = midpoint of the two pad contact centres (moves slightly with q), approach = from the pivot midpoint
toward the pad centres (along the fingers), opening = from the R pivot toward the L pivot (the handheld CAD's +x_B), normal =
approach x opening = toward the camera.  This is the same triad the demonstration is expressed in
(state-estimation-debugging/retarget/demo.py), so the two correspond 1:1 - the whole point of a UMI-matching end effector.
(Both rigs name the jaws from the camera's viewpoint looking along the fingers: L on the viewer's left.)
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RIG_YAML = ASSETS / "gripper_cad_arm_rig.yaml"
CAD_DIR = ASSETS / "cad" / "arm_gripper"
MESH_DIR = ASSETS / "meshes"
LEROBOT_URDF = ASSETS / "model" / "so_arm101.urdf"      # lerobot's SO-ARM101 URDF (Apache-2.0), vendored; its meshes are in MESH_DIR

# which CAD solid belongs to which body; the servo is the lerobot STL placed per the YAML
PARTS = {
    "CASE": "gripper_link",
    "UPPER PLATE": "gripper_link",
    "Part 1": "gripper_link",               # the bracket: mount plate -> case, with the camera cradle
    "WRIST_MOUNT_PLATE": "gripper_link",
    "X3_CAMERA": "gripper_link",
    "SERVO_HOLDER": "gripper_link",
    "Spur gear (20 teeth) (1)": "jaw_L",    # the L gear with its arm
    "FLAP_L_ver3_AIRoA2": "jaw_L",
    "FINGER_PAD_L": "jaw_L",
    "Spur gear (20 teeth)": "jaw_R",
    "ATTACHMENT_METAL_R": "jaw_R",
    "FLAP_R_ver3_AIRoA": "jaw_R",
    "FINGER_PAD_R": "jaw_R",
    "HORN_TO_GEAR_ADAPTER": "jaw_R",
}
COLOURS = {"X3_CAMERA": (0.12, 0.12, 0.13, 1), "ATTACHMENT_METAL_R": (0.75, 0.76, 0.78, 1), "STS3215": (0.1, 0.1, 0.1, 1),
           "FINGER_PAD_L": (1.0, 0.78, 0.1, 1), "FINGER_PAD_R": (1.0, 0.78, 0.1, 1), "FLAP_L_ver3_AIRoA2": (1.0, 0.8, 0.15, 1),
           "FLAP_R_ver3_AIRoA": (1.0, 0.8, 0.15, 1)}
DEFAULT_COLOUR = (0.92, 0.9, 0.85, 1)

DEFAULT_MASSES = {
    "notes": "mass_g overrides density; density in g/cm3 is applied to the CAD volume.  Printed parts: PLA 1.24 g/cm3 x 0.8 "
             "(walls + infill) = 1.0 g/cm3 - an assumption, weigh the real parts and edit this file.",
    "densities_g_cm3": {"printed": 1.0, "aluminium": 2.7},
    "parts": {
        "X3_CAMERA": {"mass_g": 180.0, "why": "Insta360 X3 specification"},
        "STS3215": {"mass_g": 55.0, "why": "Feetech specification"},
        "ATTACHMENT_METAL_R": {"material": "aluminium"},
    },
    "default_material": "printed",
}


def orthonormal(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    return Q


def rotz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class GripperCAD:
    R_Dg: np.ndarray                 # columns = g axes in D
    t_Dg: np.ndarray                 # g origin in D [mm]
    pivot_D: dict                    # "L"/"R" -> (x, y) mm
    polar: dict                      # "L"/"R" -> (r_mm, phi0_deg)
    sign: dict                       # "L": +1, "R": -1 (phi = phi0 + sign * q)
    strip_z_D: float                 # z_D of the contact strips
    open_stop_deg: float
    camera_center_D: np.ndarray
    lens_axis_D: np.ndarray          # finger-side lens axis
    body_axis_D: np.ndarray
    servo_rotz_deg: float
    servo_t_D: np.ndarray
    yaml: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ frames
    def D_to_g(self, p_mm) -> np.ndarray:
        """Points (…,3) from D [mm] to g [mm]."""
        return (np.asarray(p_mm, float) - self.t_Dg) @ self.R_Dg

    def vec_D_to_g(self, v) -> np.ndarray:
        return np.asarray(v, float) @ self.R_Dg

    @property
    def hinge_axis_g(self) -> np.ndarray:
        return self.vec_D_to_g([0.0, 0.0, 1.0])

    def pivot_g_m(self, jaw: str) -> np.ndarray:
        x, y = self.pivot_D[jaw]
        return self.D_to_g([x, y, self.strip_z_D]) / 1000.0

    # ------------------------------------------------------------------ the jaw law
    def contact_centres_D(self, q_deg):
        """(cL, cR) (…,3) in D [mm] at jaw opening q [deg] (REV_L = q, REV_R = -q)."""
        q = np.asarray(q_deg, float)
        out = []
        for jaw in ("L", "R"):
            r, phi0 = self.polar[jaw]
            phi = np.radians(phi0 + self.sign[jaw] * q)
            x, y = self.pivot_D[jaw]
            out.append(np.stack([x + r * np.cos(phi), y + r * np.sin(phi), np.broadcast_to(self.strip_z_D, q.shape)], -1))
        return out[0], out[1]

    def gap_mm(self, q_deg):
        cL, cR = self.contact_centres_D(q_deg)
        return np.linalg.norm(cL - cR, axis=-1)

    def q_from_gap(self, gap_mm):
        """Jaw angle [deg] for a pad gap [mm] (monotone law inverted on a fine grid)."""
        if not hasattr(self, "_grid"):
            qs = np.linspace(0.0, self.open_stop_deg, 6000)
            self._grid = (qs, self.gap_mm(qs))
        qs, gs = self._grid
        return np.interp(np.asarray(gap_mm, float), gs, qs)

    def pad_site_local_m(self, jaw: str) -> np.ndarray:
        """The contact-strip centre in the jaw's own frame (origin at the pivot, axes parallel to g), closed pose, metres."""
        cL, cR = self.contact_centres_D(0.0)
        c = cL if jaw == "L" else cR
        x, y = self.pivot_D[jaw]
        return self.vec_D_to_g(c - np.array([x, y, self.strip_z_D])) / 1000.0

    def grip_point_g_m(self, q_deg):
        """The grip origin (midpoint of the pad centres) in g [m] at jaw angle q."""
        cL, cR = self.contact_centres_D(q_deg)
        return self.D_to_g(0.5 * (cL + cR)) / 1000.0

    def grip_triad_g(self) -> np.ndarray:
        """Columns: approach (along the fingers), opening (from the R pivot toward the L pivot: the demonstration's +x_B, the
        viewer's left when looking along the fingers with the camera up), normal = approach x opening (toward the camera); in g."""
        pL, pR = (np.array([*self.pivot_D[j], self.strip_z_D]) for j in ("L", "R"))
        cL, cR = self.contact_centres_D(0.0)
        a = 0.5 * (cL + cR) - 0.5 * (pL + pR)
        a /= np.linalg.norm(a)
        o = pL - pR
        o = o - a * (o @ a)
        o /= np.linalg.norm(o)
        n = np.cross(a, o)
        return self.R_Dg.T @ np.stack([a, o, n], axis=1)

    def tips_g_m(self) -> np.ndarray:
        """Midpoint of the pad tips (closed) in g [m]: the pipeline's 'TCP'."""
        return self.D_to_g(self.yaml["jaws"]["closed_pose_tips_D"]) / 1000.0 if "closed_pose_tips_D" in self.yaml.get("jaws", {}) \
            else self.D_to_g([11.52, 90.94, -0.42]) / 1000.0

    def camera_g_m(self):
        """Camera centre [m] and (lens axis finger side, body axis) in g."""
        return self.D_to_g(self.camera_center_D) / 1000.0, self.vec_D_to_g(self.lens_axis_D), self.vec_D_to_g(self.body_axis_D)


def load_rig(path: Path = RIG_YAML) -> GripperCAD:
    y = yaml.safe_load(open(path, encoding="utf-8"))
    rows = np.array(y["in_URDF_gripper_link_frame"]["T_D_to_g_rows"], float)
    R, t = orthonormal(rows[:, :3]), rows[:, 3]
    jaws = y["jaws"]
    return GripperCAD(
        R_Dg=R, t_Dg=t,
        pivot_D={k: tuple(float(x) for x in v) for k, v in jaws["pivots_D"].items()},
        polar={k: (float(v["r"]), float(v["phi0_deg"])) for k, v in jaws["contact_polar"].items()},
        sign={"L": 1.0, "R": -1.0},
        strip_z_D=float(jaws["contact_strip_centre_closed_D"]["L"][2]),
        open_stop_deg=float(jaws["open_stop_deg"]),
        camera_center_D=np.array(y["camera"]["center_D"], float),
        lens_axis_D=np.array(y["camera"]["lens_axis_finger_side_D"], float),
        body_axis_D=np.array(y["camera"]["body_axis_D"], float),
        servo_rotz_deg=float(y["actuator"]["servo_transform_D"]["rotation_z_deg"]),
        servo_t_D=np.array(y["actuator"]["servo_transform_D"]["translation"], float),
        yaml=y)


# ---------------------------------------------------------------------------- meshes
def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    b = np.fromfile(path, dtype=np.uint8)
    n = int(np.frombuffer(b[80:84].tobytes(), "<u4")[0])
    tri = np.frombuffer(b[84:84 + 50 * n].tobytes(), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    V = tri["v"].reshape(-1, 3).astype(float)
    F = np.arange(len(V)).reshape(-1, 3)
    return V, F


def write_stl(path: Path, V: np.ndarray, F: np.ndarray):
    tri = np.zeros(len(F), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    P = V[F]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    tri["n"], tri["v"] = n, P
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(F)))
        f.write(tri.tobytes())


def dedupe(V: np.ndarray, F: np.ndarray, tol: float = 1e-6):
    key = np.round(V / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return V[idx], inv.reshape(-1)[F]


def mesh_volume_centroid(V: np.ndarray, F: np.ndarray):
    P = V[F]
    v = np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])) / 6.0
    vol = v.sum()
    c = ((P[:, 0] + P[:, 1] + P[:, 2]) / 4.0 * v[:, None]).sum(0) / vol if abs(vol) > 1e-12 else P.reshape(-1, 3).mean(0)
    return float(vol), c


def load_masses(path: Path = ASSETS / "masses.yaml") -> dict:
    if not path.exists():
        path.write_text(yaml.safe_dump(DEFAULT_MASSES, sort_keys=False), encoding="utf-8")
    return yaml.safe_load(open(path, encoding="utf-8"))


@dataclass
class Part:
    name: str
    body: str
    file: str                  # STL in MESH_DIR, metres, in the body's frame
    mass_kg: float
    volume_cm3: float
    centroid_m: np.ndarray     # in the body's frame
    rgba: tuple


def prepare_meshes(cad: GripperCAD, out_dir: Path = MESH_DIR, rebuild: bool = False) -> list[Part]:
    """Every gripper part as a metre-scale STL in its body's frame (gripper_link, jaw_L or jaw_R), with its mass."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(CAD_DIR / "manifest.json", encoding="utf-8"))
    masses = load_masses()
    dens = masses["densities_g_cm3"]
    parts = []
    origins = {"gripper_link": (cad.t_Dg, "g"), "jaw_L": (np.array([*cad.pivot_D["L"], cad.strip_z_D]), "g"),
               "jaw_R": (np.array([*cad.pivot_D["R"], cad.strip_z_D]), "g")}

    def place(name, V_D, F, body, volume_mm3=None):
        o, _ = origins[body]
        V = (V_D - o) @ cad.R_Dg / 1000.0
        V, F = dedupe(V, F)
        fname = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name) + ".stl"
        if rebuild or not (out_dir / fname).exists():
            write_stl(out_dir / fname, V, F)
        vol_m3, c = mesh_volume_centroid(V, F)
        vol_cm3 = (volume_mm3 / 1000.0) if volume_mm3 else vol_m3 * 1e6
        spec = masses["parts"].get(name, {})
        if "mass_g" in spec:
            m = spec["mass_g"] / 1000.0
        else:
            m = dens[spec.get("material", masses["default_material"])] * vol_cm3 / 1000.0
        parts.append(Part(name, body, fname, m, vol_cm3, c, COLOURS.get(name, DEFAULT_COLOUR)))

    by_name = {m.get("name_ascii", m["name"]): m for m in manifest}
    for name, body in PARTS.items():
        m = by_name.get(name) or by_name.get(name.replace(" ", "_"))
        if m is None:
            cands = [k for k in by_name if k.startswith(name.split(" ")[0])]
            raise KeyError(f"CAD part {name!r} not in the manifest; have {sorted(by_name)} (candidates {cands})")
        V, F = read_stl(CAD_DIR / m["file"])
        place(name, V, F, body, m["volume_mm3"])
    # the jaw servo: lerobot's STS3215 mesh (metres, its own frame) placed per the YAML (mm, D frame)
    V, F = read_stl(MESH_DIR / "sts3215_03a_v1.stl")
    V_D = (V * 1000.0) @ rotz(cad.servo_rotz_deg).T + cad.servo_t_D
    place("STS3215", V_D, F, "gripper_link")
    return parts


def describe_parts(parts: list[Part]) -> str:
    lines = []
    tot = 0.0
    for p in parts:
        tot += p.mass_kg
        lines.append(f"  {p.name:28s} -> {p.body:12s} {p.volume_cm3:7.1f} cm3  {p.mass_kg*1000:6.1f} g  centroid {np.round(p.centroid_m*1000, 1)} mm")
    lines.append(f"  total {tot*1000:.0f} g")
    return "\n".join(lines)
