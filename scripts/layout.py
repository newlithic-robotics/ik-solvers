"""The cell layout as a dimensioned drawing, so the real arms, plate and jigs can be placed where the solution put them.

    .venv/Scripts/python.exe scripts/layout.py --left-traj out/bimanual/Episode-1_left_bimanual.npz --right-traj out/bimanual/Episode-1_right_bimanual.npz --out out/layout

Everything is expressed in the CELL frame: origin = the LEFT arm's shoulder-pan axis on the table, x = the direction the left
arm faces (its base_link +x, pan joint at zero), y = to the left of that, z up from the table top.  Outputs:
  cell_layout.pdf / .png     the dimensioned top view (A3 landscape, scale 1:2.5)
  template_1_left_base_A3.pdf   1:1 print (landscape): left base footprint, pan-axis cross, facing arrow, left rest point
  template_2_plate_A3.pdf       1:1 print (portrait): left pan-axis cross + sight line, plate outline, the nine holes (two screws), put-downs
  template_3_right_base_A3.pdf  1:1 print (landscape): right base footprint, pan-axis cross, right rest point, nut put-down
  cell_points.csv            every reference point (cell frame and demo-world frame)
  cell_layout.md             the tables and the setup procedure
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newlithic.cad import load_rig, rotz                                        # noqa: E402
from newlithic.demo import PHASE, Placement, load_demo                          # noqa: E402
from newlithic.kin import ArmKinematics                                         # noqa: E402
from newlithic.model import PLATE_THICKNESS, TABLE_Z_BASE, compile_arm, ArmConfig   # noqa: E402
from newlithic.scene import PLATE_HOLES, SceneConfig, build_scene               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--left-traj", default="out/bimanual/Episode-1_left_bimanual.npz")
ap.add_argument("--right-traj", default="out/bimanual/Episode-1_right_bimanual.npz")
ap.add_argument("--out", default="out/layout")
args = ap.parse_args()

import mujoco  # noqa: E402

demo = load_demo()
cad = load_rig()
arms, trajs = {}, {}
for name, f in (("left", args.left_traj), ("right", args.right_traj)):
    meta = json.loads(Path(f).with_suffix(".json").read_text())
    pl = meta["placement"]
    arms[name] = Placement(pl["x_m"], pl["y_m"], pl["yaw_deg"], pl["surface_z_base"])
    trajs[name] = np.load(f)
scene = build_scene(demo, SceneConfig(arms=arms))
model = scene.model
single, _, _, _ = compile_arm(ArmConfig(), save=False)
kin1 = ArmKinematics(single, cad=cad)
pan_xy = kin1.pan_axis_xy                                   # pan axis in base_link (m)

# ---------------------------------------------------------------- frames
T_wb = {n: arms[n].T_world_base(demo.table_z) for n in arms}    # world <- base
# cell frame: origin at the left pan axis, axes of the left base, z from the table top
T_wc = T_wb["left"].copy()
T_wc[:3, 3] = T_wb["left"][:3, :3] @ np.array([pan_xy[0], pan_xy[1], TABLE_Z_BASE]) + T_wb["left"][:3, 3]
T_cw = np.linalg.inv(T_wc)
w2c = lambda p: (T_cw[:3, :3] @ np.asarray(p, float) + T_cw[:3, 3]) * 1000.0           # world m -> cell mm
heading_c = lambda yaw_world: (yaw_world - arms["left"].yaw_deg + 180) % 360 - 180    # a world heading in the cell frame


# ---------------------------------------------------------------- base footprint (from the base meshes, base_link frame)
def base_footprint_mm():
    d = mujoco.MjData(single)
    mujoco.mj_forward(single, d)
    bid = mujoco.mj_name2id(single, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    P = []
    for g in range(single.ngeom):
        if single.geom_bodyid[g] != bid or single.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = single.geom_dataid[g]
        V = single.mesh_vert[single.mesh_vertadr[mid]:single.mesh_vertadr[mid] + single.mesh_vertnum[mid]]
        R = d.geom_xmat[g].reshape(3, 3)
        P.append(V @ R.T + d.geom_xpos[g])
    P = np.vstack(P) * 1000.0
    zmin = P[:, 2].min()
    low = P[P[:, 2] < zmin + 6.0]                             # the bottom 6 mm: what touches the table
    from scipy.spatial import ConvexHull
    hull = ConvexHull(low[:, :2])
    return low[hull.vertices, :2], zmin


foot_b, base_bottom = base_footprint_mm()                      # base_link mm
foot_pan = foot_b - pan_xy * 1000.0                            # relative to the pan axis, base axes
bb = dict(x_min=float(foot_pan[:, 0].min()), x_max=float(foot_pan[:, 0].max()), y_min=float(foot_pan[:, 1].min()), y_max=float(foot_pan[:, 1].max()))

# ---------------------------------------------------------------- reference points
pts = {}                                                       # name -> (cell xyz mm, world xyz m, note)
L, R = demo.hand("left"), demo.hand("right")


def add(name, p_world, note=""):
    pts[name] = (w2c(p_world), np.asarray(p_world, float), note)


add("left_pan_axis", T_wc[:3, 3], "left arm shoulder-pan axis, on the table")
pr = T_wb["right"][:3, :3] @ np.array([pan_xy[0], pan_xy[1], TABLE_Z_BASE]) + T_wb["right"][:3, 3]
add("right_pan_axis", pr, "right arm shoulder-pan axis, on the table")
# plate
Tp = scene.plate_T
add("plate_centre", Tp[:3, 3] + np.array([0, 0, PLATE_THICKNESS]), "plate centre, top face")
for k, (hx, hy) in enumerate(PLATE_HOLES):
    add(f"plate_hole_{k}", Tp[:3, :3] @ np.array([hx, hy, PLATE_THICKNESS]) + Tp[:3, 3], "hole centre, plate top")
plate_yaw_world = np.degrees(np.arctan2(Tp[1, 0], Tp[0, 0]))
# picks (demo contact), put-downs (demo releases)
table_holds = [k for k in L.holds() if L.height[L.contact_frame(k)] < 0.06 and -L.approach[L.contact_frame(k), 2] > 0.85]
for i, k in enumerate(table_holds):
    add(f"pick_{i+1}", L.p[L.events[k]["contact"]], f"left hand grasps screw {i+1} (demo contact, hold {k})")
    add(f"putdown_{i+1}", L.p[L.events[k]["release"]], f"left hand releases screw {i+1} (demo release, hold {k})")
nut_holds = [k for k in R.holds() if R.height[np.where(R.hold_id == k)[0][-1]] < 0.06]
for i, k in enumerate(nut_holds):
    add(f"nut_putdown_{i+1}", R.p[R.events[k]["release"]], f"right hand releases a nut (demo release, hold {k})")
# rest (docked) points and headings, demo frame 0
for name, h in (("left", L), ("right", R)):
    add(f"{name}_rest_demo", h.p[0], f"{name} hand docked in its jig at t=0 (pad midpoint)")
# the arms' own first-frame grip poses (what the solution starts from)
kins = {n: ArmKinematics(model, prefix=f"{n}_", cad=cad) for n in arms}
rest_arm = {}
for n in arms:
    Rg, pg = kins[n].grip(trajs[n]["q"][0], float(trajs[n]["jaw"][0]))
    add(f"{n}_rest_arm", pg, f"{n} arm's grip point at the solution's first frame")
    rest_arm[n] = (Rg, pg)
# in-air work centroid
rel = (L.phase == PHASE["hold"]) & (L.height > 0.06)
add("in_air_centroid", 0.5 * (L.p[rel].mean(0) + R.p[rel].mean(0)), "centre of the in-air unscrewing (both hands)")

headings = {"left": 0.0, "right": heading_c(arms["right"].yaw_deg), "plate": heading_c(plate_yaw_world)}
bearing = float(np.degrees(np.arctan2(pts["right_pan_axis"][0][1], pts["right_pan_axis"][0][0])))     # left pan axis -> right pan axis
off_right = float((headings["right"] - (bearing + 180.0) + 180.0) % 360.0 - 180.0)                   # right's facing vs the line back to the left
rest_heading = {}
for name, h in (("left", L), ("right", R)):
    a = T_cw[:3, :3] @ h.approach[0]
    rest_heading[f"{name}_demo"] = (np.degrees(np.arctan2(a[1], a[0])), np.degrees(np.arcsin(np.clip(-a[2], -1, 1))))
    a = T_cw[:3, :3] @ rest_arm[name][0][:, 0]
    rest_heading[f"{name}_arm"] = (np.degrees(np.arctan2(a[1], a[0])), np.degrees(np.arcsin(np.clip(-a[2], -1, 1))))

# ---------------------------------------------------------------- tables
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
import pandas as pd
rows = []
for k, (c, w, note) in pts.items():
    rows.append(dict(point=k, x_mm=round(c[0], 1), y_mm=round(c[1], 1), z_mm=round(c[2], 1), dist_from_left_pan_mm=round(float(np.hypot(c[0], c[1])), 1),
                     dist_from_right_pan_mm=round(float(np.hypot(*(c[:2] - pts["right_pan_axis"][0][:2]))), 1),
                     world_x_m=round(w[0], 4), world_y_m=round(w[1], 4), world_z_m=round(w[2], 4), note=note))
df = pd.DataFrame(rows)
df.to_csv(out / "cell_points.csv", index=False)
key = ["left_pan_axis", "right_pan_axis", "plate_centre", "pick_1", "pick_2", "left_rest_demo", "right_rest_demo"]
dist = pd.DataFrame({a: {b: round(float(np.hypot(*(pts[a][0][:2] - pts[b][0][:2]))), 0) for b in key} for a in key})
dist.to_csv(out / "cell_distances.csv")

md = [f"# Cell layout for Episode-1 (bimanual solution of {Path(args.left_traj).name} / {Path(args.right_traj).name})", "",
      "Cell frame: origin = the LEFT arm's shoulder-pan axis projected onto the table top; x = the direction the left arm faces at pan 0; "
      "y = to its left; z = up from the table top.  All mm.", "",
      "## Arms", "",
      f"* Left arm: pan axis at (0, 0), heading 0 deg.  Right arm: pan axis at ({pts['right_pan_axis'][0][0]:.0f}, {pts['right_pan_axis'][0][1]:.0f}), "
      f"heading {headings['right']:.1f} deg (its facing direction, measured from the cell x axis, counter-clockwise positive).",
      f"* Distance between the two pan axes: {np.hypot(*pts['right_pan_axis'][0][:2]):.0f} mm.  Bearing of the right pan axis from the left: "
      f"{bearing:.1f} deg from the cell x axis.  The left arm faces {bearing:.1f} deg to the right of the line to the other arm; the right arm faces "
      f"{off_right:.1f} deg off the line back to the left arm (positive = to its left).",
      f"* Base footprint (both arms, from the URDF base meshes), relative to the pan axis in the arm's own axes (x ahead): "
      f"x {bb['x_min']:.0f} .. {bb['x_max']:.0f} mm, y {bb['y_min']:.0f} .. {bb['y_max']:.0f} mm.  The pan axis is {-bb['x_min']:.0f} mm from the rear edge and "
      f"{bb['x_max']:.0f} mm from the front edge of the footprint, {(bb['y_max'] + bb['y_min'])/2:+.0f} mm off its centre line.",
      f"* Height: the solution assumes the plate top {-(TABLE_Z_BASE + PLATE_THICKNESS)*1000:.2f} mm below the base_link origin and the bare table "
      f"{-TABLE_Z_BASE*1000:.2f} mm below it (as measured on the stock arm in September); the CAD base bottom is {-base_bottom:.1f} mm below base_link. "
      f"If the new arm stands directly on the table, re-solve with `--surface-z-mm {(base_bottom/1000 + PLATE_THICKNESS)*1000:.1f}` or shim the bases by {(TABLE_Z_BASE*1000 + -base_bottom) * -1 - 0:.0f} mm.",
      "", "## Plate (70 x 70 x 10 mm, nine d12 holes)", "",
      f"* Centre at ({pts['plate_centre'][0][0]:.0f}, {pts['plate_centre'][0][1]:.0f}), rotated {headings['plate']:.1f} deg (the plate's own x axis vs the cell x axis).  "
      f"Seen from the left pan axis the centre lies at bearing {np.degrees(np.arctan2(pts['plate_centre'][0][1], pts['plate_centre'][0][0])):.1f} deg, "
      f"{np.hypot(*pts['plate_centre'][0][:2]):.0f} mm away; the plate's hole rows (its x axis) are turned "
      f"{(headings['plate'] - np.degrees(np.arctan2(pts['plate_centre'][0][1], pts['plate_centre'][0][0])) + 180) % 360 - 180:+.1f} deg from that sight line.",
      f"* Screw 1 stands in hole 0 (plate corner) = pick_1 at ({pts['pick_1'][0][0]:.0f}, {pts['pick_1'][0][1]:.0f}); screw 2 in hole 4 (plate centre) = pick_2 at "
      f"({pts['pick_2'][0][0]:.0f}, {pts['pick_2'][0][1]:.0f}).  The plate is placed so both holes sit on the demonstrated picks "
      f"({scene.names['plate_mismatch_mm']:+.1f} mm of hole-spacing mismatch split between them).",
      f"* Put-downs (where the demonstrator released the screws): ({pts['putdown_1'][0][0]:.0f}, {pts['putdown_1'][0][1]:.0f}) and "
      f"({pts['putdown_2'][0][0]:.0f}, {pts['putdown_2'][0][1]:.0f}); nut put-down at ({pts['nut_putdown_1'][0][0]:.0f}, {pts['nut_putdown_1'][0][1]:.0f}).",
      "", "## Jigs (docked start / end)", "",
      "The recording does not say where the gripper sits inside the jig, so the jig cannot be drawn exactly.  What is known is the "
      "docked GRIP POINT (pad midpoint) and the finger heading of each hand at t = 0:", ""]
for name in ("left", "right"):
    c = pts[f"{name}_rest_demo"][0]
    hd, el = rest_heading[f"{name}_demo"]
    ca = pts[f"{name}_rest_arm"][0]
    ha, ea = rest_heading[f"{name}_arm"]
    md.append(f"* {name}: demonstrated rest point ({c[0]:.0f}, {c[1]:.0f}) at {c[2]:.0f} mm above the table, fingers heading {hd:.0f} deg in the cell frame, "
              f"{el:.0f} deg below horizontal.  The arm's own first-frame grip point: ({ca[0]:.0f}, {ca[1]:.0f}) at {ca[2]:.0f} mm, heading {ha:.0f} deg, "
              f"{ea:.0f} deg below horizontal (the arm cannot reproduce the docked orientation: it reaches the point, not the lean).")
md += ["", "Practical: drive the arm to the solution's first frame (joint angles below), dock the gripper in the jig by hand, and slide the jig "
       "on the table until the gripper seats; mark the jig outline.  That makes the jig the arm's start pose, which is all the policy needs.", ""]
for n in arms:
    q0 = np.degrees(trajs[n]["q"][0])
    md.append(f"* {n} first-frame joints (deg, lerobot convention): pan {q0[0]:.1f}, lift {q0[1]:.1f}, elbow {q0[2]:.1f}, wrist_flex {q0[3]:.1f}, wrist_roll {q0[4]:.1f}, "
              f"jaw {np.degrees(trajs[n]['jaw'][0]):.1f} deg (gap {cad.gap_mm(np.degrees(trajs[n]['jaw'][0])):.1f} mm)")
md += ["", "## Distances between reference points (mm, on the table plane)", "", dist.to_markdown(), "",
       "## All points", "", df[["point", "x_mm", "y_mm", "z_mm", "dist_from_left_pan_mm", "dist_from_right_pan_mm", "note"]].to_markdown(index=False), "",
       "## Setup procedure", "",
       "1. Mark the left pan axis on the table (point O) and a straight line through it for the cell x axis (the left arm's facing direction).",
       "2. Stand the left arm so its pan axis is on O and it faces along x: the base footprint corners are at the offsets given above "
       "(rear edge %.0f mm behind the axis, front edge %.0f mm ahead, half-widths %.0f / %.0f mm)." % (-bb['x_min'], bb['x_max'], -bb['y_min'], bb['y_max']),
       "3. Mark the right pan axis: %.0f mm from O at %.1f deg from the x axis (counter-clockwise), or by the two distances in the table. "
       "Stand the right arm on it facing %.1f deg (cell frame)." % (np.hypot(*pts['right_pan_axis'][0][:2]), np.degrees(np.arctan2(pts['right_pan_axis'][0][1], pts['right_pan_axis'][0][0])), headings['right']),
       "4. Place the plate with its centre at (%.0f, %.0f) rotated %.1f deg; check hole 0 and hole 4 against pick_1 / pick_2 (distances from both pan axes in the table). "
       "Stand screw 1 (nut on) in hole 0 and screw 2 in hole 4." % (pts['plate_centre'][0][0], pts['plate_centre'][0][1], headings['plate']),
       "5. Jigs: as in the section above.", "6. Print template_1_left_base_A3.pdf, template_2_plate_A3.pdf (portrait) and template_3_right_base_A3.pdf at 100 % and lay them on the table: "
       "template 2 carries the left pan-axis cross and the sight line to the plate, so it registers to template 1 by the pan-axis cross."]
(out / "cell_layout.md").write_text("\n".join(md), encoding="utf-8")

# ---------------------------------------------------------------- drawing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrow, Rectangle


def footprint_cell(name):
    T = T_wb[name]
    P = np.column_stack([foot_b / 1000.0, np.full(len(foot_b), TABLE_Z_BASE)])     # base_link m
    W = P @ T[:3, :3].T + T[:3, 3]
    return np.array([w2c(p)[:2] for p in W])


def plate_outline_cell():
    sq = np.array([[-35, -35], [35, -35], [35, 35], [-35, 35]]) / 1000.0
    W = np.array([Tp[:3, :3] @ np.array([x, y, PLATE_THICKNESS]) + Tp[:3, 3] for x, y in sq])
    return np.array([w2c(p)[:2] for p in W])


def draw(ax, scale_note=True):
    for name, col in (("left", "#1f77b4"), ("right", "#d62728")):
        fp = footprint_cell(name)
        ax.add_patch(Polygon(fp, closed=True, fill=False, lw=1.5, ec=col))
        c = pts[f"{name}_pan_axis"][0]
        ax.plot([c[0] - 12, c[0] + 12], [c[1], c[1]], color=col, lw=1); ax.plot([c[0], c[0]], [c[1] - 12, c[1] + 12], color=col, lw=1)
        ax.add_patch(Circle(c[:2], 4, fill=False, ec=col))
        hd = np.radians(headings[name])
        ax.add_patch(FancyArrow(c[0], c[1], 90 * np.cos(hd), 90 * np.sin(hd), width=2, head_width=12, color=col, length_includes_head=True))
        ax.text(c[0] + 100 * np.cos(hd), c[1] + 100 * np.sin(hd), f"{name} arm faces {headings[name]:.1f}°", color=col, fontsize=7, ha="center")
        ax.text(c[0], c[1] - 22, f"{name} pan axis ({c[0]:.0f}, {c[1]:.0f})", color=col, fontsize=7, ha="center", va="top")
    po = plate_outline_cell()
    ax.add_patch(Polygon(po, closed=True, fill=False, lw=1.2, ec="k"))
    for k in range(9):
        h = pts[f"plate_hole_{k}"][0]
        ax.add_patch(Circle(h[:2], 6, fill=(k in (0, 4)), fc="#ffb000" if k in (0, 4) else "none", ec="k", lw=0.8))
    for name, mk, col in (("pick_1", "x", "k"), ("pick_2", "x", "k"), ("putdown_1", "s", "#2ca02c"), ("putdown_2", "s", "#2ca02c"), ("nut_putdown_1", "D", "#d62728"),
                          ("in_air_centroid", "*", "#9467bd")):
        c = pts[name][0]
        ax.plot(c[0], c[1], mk, color=col, ms=7 if mk != "*" else 11, mew=1.5)
        dx, dy = {"pick_1": (10, -22), "pick_2": (10, 8), "putdown_1": (8, 12), "putdown_2": (8, -16), "nut_putdown_1": (8, 10), "in_air_centroid": (10, -16)}.get(name, (8, 8))
        ax.text(c[0] + dx, c[1] + dy, f"{name} ({c[0]:.0f}, {c[1]:.0f})" + (f", {c[2]:.0f} mm up" if name == "in_air_centroid" else ""), fontsize=6.5, color=col)
    for name, col in (("left", "#1f77b4"), ("right", "#d62728")):
        c = pts[f"{name}_rest_demo"][0]
        hd, el = rest_heading[f"{name}_demo"]
        ax.plot(c[0], c[1], "o", color=col, ms=6, mfc="none")
        ax.add_patch(FancyArrow(c[0], c[1], 45 * np.cos(np.radians(hd)), 45 * np.sin(np.radians(hd)), width=1, head_width=8, color=col, length_includes_head=True))
        ax.text(c[0] + 10, c[1] - 14, f"{name} rest (dock) ({c[0]:.0f}, {c[1]:.0f}), {c[2]:.0f} mm up, fingers {hd:.0f}° / {el:.0f}° down", fontsize=6.5, color=col)
        ca = pts[f"{name}_rest_arm"][0]
        ax.plot(ca[0], ca[1], "+", color=col, ms=8)
    # dimension lines
    rp = pts["right_pan_axis"][0]
    ax.annotate("", xy=rp[:2], xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8))
    ax.text(rp[0] / 2, rp[1] / 2 + 10, f"{np.hypot(rp[0], rp[1]):.0f} mm, bearing {np.degrees(np.arctan2(rp[1], rp[0])):.1f}°", color="gray", fontsize=7, rotation=np.degrees(np.arctan2(rp[1], rp[0])), ha="center")
    pc = pts["plate_centre"][0]
    for base, col in (("left_pan_axis", "#1f77b4"), ("right_pan_axis", "#d62728")):
        b = pts[base][0]
        ax.annotate("", xy=pc[:2], xytext=b[:2], arrowprops=dict(arrowstyle="<->", color=col, lw=0.6, ls="--"))
        ax.text((pc[0] + b[0]) / 2, (pc[1] + b[1]) / 2 - 12, f"{np.hypot(*(pc[:2] - b[:2])):.0f} mm", color=col, fontsize=7, ha="center")
    ax.text(pc[0], pc[1] + 45, f"plate centre ({pc[0]:.0f}, {pc[1]:.0f}), rotated {headings['plate']:.1f}°", fontsize=7, ha="center")
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_xlabel("x [mm]  (left arm's facing direction)")
    ax.set_ylabel("y [mm]  (left of the left arm)")


allp = np.array([pts[k][0][:2] for k in pts] + [p for n in arms for p in footprint_cell(n)])
xmin, ymin = allp.min(0) - 80; xmax, ymax = allp.max(0) + 80
fig, ax = plt.subplots(figsize=(420 / 25.4, 297 / 25.4))
draw(ax)
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_title(f"Episode-1 cell layout - cell frame on the left arm's pan axis - both arms with the Newlithic gripper (bases {np.hypot(*pts['right_pan_axis'][0][:2]):.0f} mm apart)", fontsize=10)
fig.text(0.01, 0.01, "x: filled holes = the two screws (hole 0 corner, hole 4 centre).  o + arrow: demonstrated docked pad midpoint and finger heading.  + : the arm's own first-frame grip point.  "
         "squares: screw put-downs, diamond: nut put-down, star: in-air unscrewing centre.  Heights are above the table top.", fontsize=7)
fig.tight_layout()
fig.savefig(out / "cell_layout.pdf"); fig.savefig(out / "cell_layout.png", dpi=150)
plt.close(fig)

# 1:1 templates (A3, print at 100 %): explicit pages, each a region that fits
def page(fname, frame_T, xlim, ylim, items, outlines, title, holes=False, sight_line=False):
    """Points are given in the cell frame (mm) and drawn in `frame_T` (world <- frame).  xlim/ylim: the page window in that
    frame; the figure is exactly that size in mm, so the print is 1:1."""
    Tinv = np.linalg.inv(frame_T)

    def to_f(c):
        c = np.asarray(c, float)
        if len(c) == 2:
            c = np.array([c[0], c[1], pts["plate_centre"][0][2]])
        w = T_wc[:3, :3] @ (c / 1000.0) + T_wc[:3, 3]
        return (Tinv @ np.array([*w, 1.0]))[:3] * 1000.0
    W, H = xlim[1] - xlim[0], ylim[1] - ylim[0]
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
    missing = []
    for o, col, lw in outlines:
        ax.add_patch(Polygon(np.array([to_f(q)[:2] for q in o]), closed=True, fill=False, lw=lw, ec=col))
    if holes:
        for k in range(9):
            h = to_f(pts[f"plate_hole_{k}"][0])
            ax.add_patch(Circle(h[:2], 6, fill=(k in (0, 4)), fc="#ffb000" if k in (0, 4) else "none", ec="k", lw=0.8))
            ax.text(h[0], h[1] - 9, f"h{k}", fontsize=5, ha="center", va="top")
    for k, (mk, col, lab) in items:
        c = to_f(pts[k][0])
        if not (xlim[0] < c[0] < xlim[1] and ylim[0] < c[1] < ylim[1]):
            missing.append(k); continue
        ax.plot(c[0], c[1], mk, color=col, ms=7, mfc="none", mew=1.2)
        right_side = c[0] < 0.5 * (xlim[0] + xlim[1]) + 40
        dy = -12 if k.endswith("_rest_arm") else 5                    # the arm's first-frame point coincides with the demo's rest point
        fmt = lambda v: f"{v:.0f}".replace("-0", "0") if abs(v) < 0.5 else f"{v:.0f}"
        ax.text(c[0] + (5 if right_side else -5), c[1] + dy, f"{lab} ({fmt(c[0])}, {fmt(c[1])}; {c[2]:.0f} up)", fontsize=6.5, color=col, ha="left" if right_side else "right")
    for n in arms:
        c = to_f(pts[f"{n}_pan_axis"][0])
        if xlim[0] < c[0] < xlim[1] and ylim[0] < c[1] < ylim[1]:
            col = "#1f77b4" if n == "left" else "#d62728"
            ax.plot([c[0] - 15, c[0] + 15], [c[1], c[1]], color=col, lw=0.8); ax.plot([c[0], c[0]], [c[1] - 15, c[1] + 15], color=col, lw=0.8)
            ax.add_patch(Circle(c[:2], 3, fill=False, ec=col))
            a_f = Tinv[:3, :3] @ (T_wb[n][:3, :3] @ np.array([1.0, 0.0, 0.0]))
            ax.add_patch(FancyArrow(c[0], c[1], 60 * a_f[0], 60 * a_f[1], width=0.8, head_width=6, color=col, length_includes_head=True))
            ax.text(c[0], c[1] - 20, f"{n} pan axis ({abs(c[0]):.0f}, {abs(c[1]):.0f}); arrow = the arm's facing direction", color=col, fontsize=7, ha="center", va="top")
    if sight_line:
        o, pc = to_f(pts["left_pan_axis"][0]), to_f(pts["plate_centre"][0])
        ax.plot([o[0], pc[0]], [o[1], pc[1]], "--", color="gray", lw=0.6)
        ax.text(0.5 * (o[0] + pc[0]) + 6, 0.5 * (o[1] + pc[1]), f"{np.hypot(*(pc[:2] - o[:2])):.0f} mm", fontsize=7, color="gray")
    xs0, ys0 = xlim[1] - 115, ylim[0] + 8
    for k in range(11):
        ax.plot([xs0 + k * 10, xs0 + k * 10], [ys0, ys0 + (6 if k % 5 == 0 else 3)], "k-", lw=0.6)
    ax.plot([xs0, xs0 + 100], [ys0, ys0], "k-", lw=0.8); ax.text(xs0 + 50, ys0 + 8, "100 mm check scale", fontsize=7, ha="center")
    ax.text(xlim[0] + 6, ylim[1] - 6, title + "  Print at 100 % (A3) and check the scale bar.  Coordinates (x, y; height) in mm.", fontsize=7, va="top")
    fig.savefig(out / fname); plt.close(fig)
    if missing:
        print(f"  {fname}: outside the page: {missing}")


T_right = T_wb["right"].copy()
T_right[:3, 3] = T_wb["right"][:3, :3] @ np.array([pan_xy[0], pan_xy[1], TABLE_Z_BASE]) + T_wb["right"][:3, 3]
page("template_1_left_base_A3.pdf", T_wc, (-100, 320), (-100, 197),
     [("left_rest_demo", ("o", "#1f77b4", "left docked pad midpoint (demo)")), ("left_rest_arm", ("+", "#1f77b4", "left arm first-frame grip point"))],
     [(footprint_cell("left"), "#1f77b4", 1.2)],
     "1:1 template 1 of 3: LEFT arm base.  Cell frame: x = the left arm's facing direction, origin on its pan axis.")
page("template_2_plate_A3.pdf", T_wc, (-55, 242), (-20, 400),
     [("pick_1", ("x", "k", "pick_1 = hole 0, screw 1")), ("pick_2", ("x", "k", "pick_2 = hole 4, screw 2")),
      ("putdown_1", ("s", "#2ca02c", "screw 1 put-down")), ("putdown_2", ("s", "#2ca02c", "screw 2 put-down")),
      ("nut_putdown_1", ("D", "#d62728", "nut 1 put-down")), ("nut_putdown_2", ("D", "#d62728", "nut 2 put-down")),
      ("left_rest_demo", ("o", "#1f77b4", "left docked pad midpoint (demo)"))],
     [(plate_outline_cell(), "k", 1.0)],
     "1:1 template 2 of 3: the PLATE and the table work, cell frame (x = the left arm's facing direction, origin on its pan axis).", holes=True, sight_line=True)
page("template_3_right_base_A3.pdf", T_right, (-100, 320), (-120, 177),
     [("right_rest_demo", ("o", "#d62728", "right docked pad midpoint (demo)")), ("right_rest_arm", ("+", "#d62728", "right arm first-frame grip point")),
      ("nut_putdown_3", ("D", "#d62728", "nut put-down (hold 9)"))],
     [(footprint_cell("right"), "#d62728", 1.2)],
     "1:1 template 3 of 3: RIGHT arm base, in the right arm's own axes (x = its facing direction, origin on its pan axis).")
for f in ("template_left_A3.pdf", "template_right_A3.pdf"):
    (out / f).unlink(missing_ok=True)
# ---------------------------------------------------------------- the vertical stack (elevation)
d1 = mujoco.MjData(single)
mujoco.mj_forward(single, d1)
z_bl = -(arms["left"].surface_z_base - PLATE_THICKNESS) * 1000.0       # base_link origin above the table top [mm]
base_geoms = [g for g in range(single.ngeom) if single.body(single.geom_bodyid[g]).name == "base_link" and single.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
ztop = -np.inf
for g in base_geoms:
    mid = single.geom_dataid[g]
    V = single.mesh_vert[single.mesh_vertadr[mid]:single.mesh_vertadr[mid] + single.mesh_vertnum[mid]]
    ztop = max(ztop, (V @ d1.geom_xmat[g].reshape(3, 3).T + d1.geom_xpos[g])[:, 2].max() * 1000.0)
jid = lambda n: mujoco.mj_name2id(single, mujoco.mjtObj.mjOBJ_JOINT, n)
z_pan = d1.xanchor[jid("shoulder_pan")][2] * 1000.0 + z_bl
z_lift = d1.xanchor[jid("shoulder_lift")][2] * 1000.0 + z_bl
from newlithic.scene import BOLT
z_head_top = (BOLT["nut_h"] - BOLT["nut_engaged"] + BOLT["length"] + BOLT["head_h"]) * 1000.0
z_head_bot = z_head_top - BOLT["head_h"] * 1000.0
heights = [
    ("table top (reference)", 0.0, "the surface the plate and the shims stand on"),
    ("plate top", PLATE_THICKNESS * 1000.0, "the demonstration's reference surface (the hands' fingertips touched it)"),
    ("screw head, bottom / top", f"{z_head_bot:.1f} / {z_head_top:.1f}", "M5 x 12 standing nut-down in a hole, nut on the table, two threads engaged"),
    ("base bottom (shim thickness)", z_bl + base_bottom, "the gap under each arm's base in the model: put a shim of this thickness under both bases"),
    ("base_link origin", z_bl, "the URDF/lerobot base frame origin (TABLE_Z_BASE = %.2f mm below it)" % (-TABLE_Z_BASE * 1000)),
    ("top of the base housing", z_bl + ztop, "highest point of the base meshes"),
    ("shoulder-pan joint anchor", z_pan, "on the pan axis"),
    ("shoulder-lift axis", z_lift, "the first pitch axis: the easiest height to check with a ruler against the servo horn centre"),
    ("pick 1 / pick 2 grip origin", f"{pts['pick_1'][0][2]:.1f} / {pts['pick_2'][0][2]:.1f}", "pad-strip midpoint at the grasp; the pad tips are %.1f mm lower (at the plate top)" % (L.pad_to_tip_m * 1000)),
    ("screw put-downs (grip origin)", f"{pts['putdown_1'][0][2]:.1f} / {pts['putdown_2'][0][2]:.1f}", "release heights"),
    ("docked rest, left / right (grip origin)", f"{pts['left_rest_demo'][0][2]:.1f} / {pts['right_rest_demo'][0][2]:.1f}", "pad-strip midpoint of the gripper docked in its jig"),
    ("in-air unscrewing (grip origins)", pts["in_air_centroid"][0][2], "median height of the pair while the nut is turned"),
]
md += ["", "## Heights (mm above the table top)", "", "| level | z | what |", "|---|---:|---|"]
for name_, z, what in heights:
    md.append(f"| {name_} | {z if isinstance(z, str) else f'{z:.1f}'} | {what} |")
md += ["", f"The model stands each arm's base {z_bl + base_bottom:.1f} mm above the table (base bottom {-base_bottom:.1f} mm below the base_link origin in the CAD; "
       f"the origin {z_bl:.2f} mm above the table as measured on the stock arm in 2026-09).  Reproduce it with a {z_bl + base_bottom:.1f} mm shim under both bases, "
       f"or re-solve with `--surface-z-mm {(base_bottom + PLATE_THICKNESS * 1000):.1f}` for bases standing directly on the table and re-run this script."]
(out / "cell_layout.md").write_text("\n".join(md), encoding="utf-8")

fig, ax = plt.subplots(figsize=(420 / 25.4, 230 / 25.4))
ax.axhline(0, color="k", lw=2)
ax.add_patch(Rectangle((-200, -12), 820, 12, fc="#d9d9d9", ec="none"))
ax.text(-190, -6, "table", fontsize=8, va="center")
# the shim and the base (schematic, along the sight line to the plate)
shim = z_bl + base_bottom
ax.add_patch(Rectangle((bb["x_min"], 0), bb["x_max"] - bb["x_min"], shim, fc="#f4c27a", ec="k", lw=0.8, hatch="//"))
ax.add_patch(Rectangle((bb["x_min"], shim), bb["x_max"] - bb["x_min"], z_bl + ztop - shim, fc="#ffd34d", ec="k", lw=0.8))
ax.plot([pan_xy[0] * 0, 0], [shim, z_lift + 30], color="#1f77b4", lw=1, ls="--")
ax.plot(0, z_pan, "o", color="#1f77b4", ms=5); ax.plot(0, z_lift, "s", color="#1f77b4", ms=6)
ax.text(4, z_lift + 2, "shoulder-lift axis", fontsize=7, color="#1f77b4")
ax.text(4, z_pan - 9, "pan joint anchor", fontsize=7, color="#1f77b4")
ax.text(-55, z_bl + ztop + 4, "left arm base (schematic)", fontsize=7)
ax.text(bb["x_max"] + 4, shim / 2, f"shim {shim:.1f} mm", fontsize=8, va="center", color="#8a5a00")
# the plate with a screw, at the sight-line distance
xp = np.hypot(*pts["plate_centre"][0][:2])
ax.add_patch(Rectangle((xp - 35, 0), 70, PLATE_THICKNESS * 1000, fc="#f2f2f2", ec="k", lw=0.8))
ax.add_patch(Rectangle((xp - 6, 0), 12, PLATE_THICKNESS * 1000, fc="white", ec="none"))          # the hole
ax.add_patch(Rectangle((xp - 4, 0), 8, BOLT["nut_h"] * 1000, fc="#bfbfbf", ec="k", lw=0.5))        # nut
ax.add_patch(Rectangle((xp - 2.5, BOLT["nut_h"] * 1000 - BOLT["nut_engaged"] * 1000), 5, BOLT["length"] * 1000, fc="#404040", ec="k", lw=0.5))
ax.add_patch(Rectangle((xp - 4.25, z_head_bot), 8.5, BOLT["head_h"] * 1000, fc="#202020", ec="k", lw=0.5))
ax.text(xp, -8, f"plate + screw, {xp:.0f} mm from the pan axis", fontsize=7, ha="center", va="top")
# grip heights
for name_, x_, col, lab in (("pick_1", xp, "k", "pick: grip origin"), ("left_rest_demo", np.hypot(*pts["left_rest_demo"][0][:2]), "#1f77b4", "left docked rest"),
                            ("in_air_centroid", np.hypot(*pts["in_air_centroid"][0][:2]), "#9467bd", "in-air unscrewing")):
    z = pts[name_][0][2]
    ax.plot(x_, z, "x", color=col, ms=7, mew=1.5)
    ax.text(x_ + 6, z + 3, f"{lab} {z:.0f} mm", fontsize=7, color=col)
    if name_ == "pick_1":
        ax.plot(x_, z - L.pad_to_tip_m * 1000, "v", color=col, ms=5)
        ax.text(x_ + 6, z - L.pad_to_tip_m * 1000 - 3, f"pad tips {z - L.pad_to_tip_m*1000:.0f} mm", fontsize=7, color=col, va="top")
# vertical dimension ladder on the left
# dimension ladder: one arrow per level, texts staggered so the levels a few mm apart stay readable
levels = sorted([(PLATE_THICKNESS * 1000, f"plate top {PLATE_THICKNESS*1000:.0f}"), (shim, f"base bottom / shim {shim:.1f}"), (z_bl, f"base_link origin {z_bl:.2f}"),
                 (z_head_top, f"screw head top {z_head_top:.1f}"), (z_pan, f"pan joint anchor {z_pan:.1f}"), (z_lift, f"shoulder-lift axis {z_lift:.1f}")])
ty = []
for z, _ in levels:
    ty.append(max(z, (ty[-1] + 9) if ty else z))
for k, ((z, lab), yt) in enumerate(zip(levels, ty)):
    xd = -120 - 14 * k
    ax.annotate("", xy=(xd, z), xytext=(xd, 0), arrowprops=dict(arrowstyle="<->", color="gray", lw=0.7))
    ax.plot([xd, 0 if z > 25 else xp], [z, z], color="gray", lw=0.4, ls=":")
    ax.plot([xd - 4, xd - 14], [z, yt], color="gray", lw=0.5)
    ax.text(xd - 16, yt, lab, fontsize=7, ha="right", va="center")
ax.set_xlim(-330, 480); ax.set_ylim(-20, 230)
ax.set_aspect("equal"); ax.grid(True, lw=0.3, alpha=0.4)
ax.set_xlabel("horizontal distance from the left pan axis [mm] (schematic section toward the plate)"); ax.set_ylabel("height above the table top [mm]")
ax.set_title(f"Vertical stack: each arm base stands {shim:.1f} mm above the table in the solution (shim), plate 10 mm, screw head to {z_head_top:.1f} mm, picks at {pts['pick_1'][0][2]:.0f} mm", fontsize=9)
fig.tight_layout(); fig.savefig(out / "cell_elevation.pdf"); fig.savefig(out / "cell_elevation.png", dpi=150); plt.close(fig)
print("heights:", [(n, z) for n, z, _ in heights])
print("wrote", out, "\n", "\n".join(md[:14]))
