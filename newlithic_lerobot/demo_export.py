"""Stage 0: the state-estimation pipeline's outputs -> the demonstration file the solvers read.

Input: the results directory of `newlithic-state-estimation` for one episode (`pose` and `bimanual` runs for both clips, and the
`aperture` run with the factory lens model):

    <results>/pose/<episode>/bimanual_summary.json           registration of the right rig's world into the left's, clock offset
    <results>/pose/<episode>/<side>/<stem>_pose.csv          per frame: pose instants, lens / IMU / tcp poses, validity
    <results>/pose/<episode>/<side>/<stem>_pose_summary.json fps, T_lens1_base, readout, ...
    <results>/pose/<episode>/<side>/<stem>_map.npz           landmarks (for the table height)
    <results>/aperture*/<episode>/<side>/<stem>_aperture.csv jaw angles per frame
    <results>/aperture*/<episode>/<side>/<stem>_summary.json sticker size (the gripper geometry's scale)

with side = LHS (the left hand) and RHS.  Output: `<episode>.npz` + `.json`, per hand on ONE clock (the left rig's pose
instants) in ONE world (the left rig's, gravity aligned, z up):

    t (n,) s, R (n,3,3) world <- grip (columns approach, opening, normal), p (n,3) m the grip origin = midpoint of the two
    pad contact centres, gap (n,) m between the pad centres, phase (n,) codes, hold_id (n,), valid (n,), height / tips_height
    (n,) m above the reference surface, and for the episode table_z (m), fps and notes.

The grip frame convention is the one the arm's gripper model uses (`newlithic.cad`): approach along the fingers, opening from
the R jaw toward the L jaw (the handheld CAD's +x), normal toward the camera.

Facts to keep in mind (from the retargeting work):
  * the "table" the map gives is the dominant plane the fingers touched - the plate top in Episode-1 - fitted to about a
    millimetre; measure the real surface when replaying on hardware;
  * the aperture is calibrated per hand on the docking jig (`jig_gap_mm`): the rest window before the first hold is the one
    aperture reference a recording contains;
  * a short closing of both grippers at the same moment is the operator's synchronisation clap, relabelled transport.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

PHASE = {"transport": 0, "grasp": 1, "hold": 2, "release": 3, "approach": 4}
PHASE_NAME = {v: k for k, v in PHASE.items()}
SIDES = {"left": "LHS", "right": "RHS"}
GROUPS = ("lens1", "tcp")


# ----------------------------------------------------------------------------- the pipeline's files
def find_clip(results: Path, episode: str, side: str) -> str:
    """The clip stem of a side from its pose csv."""
    d = Path(results) / "pose" / episode / side
    hits = sorted(d.glob("*_pose.csv"))
    if not hits:
        raise FileNotFoundError(f"no *_pose.csv in {d}")
    if len(hits) > 1:
        raise ValueError(f"several clips in {d}: {[h.name for h in hits]}; keep one per side")
    return hits[0].name[: -len("_pose.csv")]


def find_aperture_dir(results: Path, episode: str, side: str, stem: str) -> Path:
    for d in sorted(Path(results).glob("aperture*")):
        if (d / episode / side / f"{stem}_aperture.csv").exists():
            return d / episode / side
    raise FileNotFoundError(f"no aperture*/{episode}/{side}/{stem}_aperture.csv under {results}")


def find_insv(data: Optional[Path], stem: str, lens: int) -> Optional[Path]:
    """The raw Insta360 file of a lens (0 = screen side, 1 = behind the jaws), anywhere under `data`."""
    if data is None:
        return None
    hits = sorted(Path(data).rglob(f"{stem}_{lens}0_*.insv"))
    if not hits and lens == 1:
        # 3K mode: one _00_ file holds both lenses side by side (lens 1 in the right half); wrist.lens1_window crops it
        hits = sorted(Path(data).rglob(f"{stem}_00_*.insv"))
    return hits[0] if hits else None


@dataclass
class Rig:
    side: str
    name: str                    # left / right
    stem: str
    n: int
    fps: float
    t: np.ndarray                # (n,) common clock (the left rig's pose instants)
    valid: np.ndarray
    R: dict                      # group -> (n,3,3) world <- frame, common world
    p: dict                      # group -> (n,3) m
    q: np.ndarray                # (n,2) jaw angles L, R [rad]
    geom: object                 # state_estimation.gripper.Geometry
    T_lens_base: np.ndarray      # (4,4) gripper base frame in the lens-1 frame
    lever_scale: float
    map_X: np.ndarray            # (L,3) landmarks, common world
    summary: dict
    files: dict = field(default_factory=dict)       # lens -> raw .insv path (if the data directory was given)
    frame0: int = 0                                  # video frame of row 0 (a segment of a clip solved per episode window)

    def frame_at(self, t) -> np.ndarray:
        """Index of the row whose instant is nearest to common-clock time t (its video frame is frame0 + that)."""
        t = np.atleast_1d(np.asarray(t, float))
        i = np.clip(np.searchsorted(self.t, t), 1, self.n - 1)
        return np.where(np.abs(self.t[i - 1] - t) <= np.abs(self.t[i] - t), i - 1, i)


def segment_of_episodes(results: Path, episode: str) -> Optional[dict]:
    """episode index -> segment, for a recording estimated per episode window (state-estimation pose with windows); None for one
    estimated as a whole."""
    b = json.loads((Path(results) / "pose" / episode / "bimanual_summary.json").read_text(encoding="utf-8"))
    if not b.get("windowed"):
        return None
    return {int(i): int(s["segment"]) for s in b["segments"] for i in s["episodes"]}


def load_registration(results: Path, episode: str, segment: Optional[int] = None) -> dict:
    b = json.loads((Path(results) / "pose" / episode / "bimanual_summary.json").read_text(encoding="utf-8"))
    if b.get("windowed"):                                                    # one registration and clock offset per episode window
        if segment is None:
            raise ValueError(f"{episode} was estimated per episode window: name the segment")
        seg = next((s for s in b["segments"] if int(s["segment"]) == int(segment)), None)
        if seg is None:
            raise ValueError(f"{episode} segment {segment} was not put on one clock: {b.get('segments_failed', {}).get(str(segment), 'not merged')}")
        reg, offset = seg["registration"], seg["clock_offset_s"]
    else:
        reg, offset = b["registration"], b["clock_offset_s"]
    return dict(scale=float(reg["scale_b_over_a"]), R=np.array(reg["R"], float), t=np.array(reg["t"], float),
                clock_offset_s=float(offset), summary=b, segment=segment)


def load_rig(results: Path, episode: str, name: str, data: Optional[Path] = None, registration: Optional[dict] = None,
             segment: Optional[int] = None) -> Rig:
    """segment: for a clip solved per episode window, only that window's rows (in its own world); Rig.frame0 is its first frame."""
    import pandas as pd
    from scipy.spatial.transform import Rotation as Rot
    from state_estimation.gripper import load_geometry

    side = SIDES[name]
    results = Path(results)
    stem = find_clip(results, episode, side)
    pose_dir = results / "pose" / episode / side
    tab = pd.read_csv(pose_dir / f"{stem}_pose.csv")
    summ = json.loads((pose_dir / f"{stem}_pose_summary.json").read_text(encoding="utf-8"))
    if "t_pose_video_s" not in tab:
        raise ValueError(f"{stem}_pose.csv has no t_pose_video_s column: re-run `pose` with a current state-estimation")
    frame0 = 0
    if segment is not None:
        rows = np.flatnonzero(tab["segment"].values == int(segment))
        if not len(rows):
            raise ValueError(f"{stem}: no rows of segment {segment}")
        frame0 = int(rows[0])
        tab = tab.iloc[rows[0]:rows[-1] + 1].reset_index(drop=True)
    reg = registration or load_registration(results, episode, segment)
    if side == "LHS":
        s, Rr, tr = 1.0, np.eye(3), np.zeros(3)
        t_common = tab.t_pose_video_s.values
    else:
        s, Rr, tr = reg["scale"], reg["R"], reg["t"]
        t_common = tab.t_pose_video_s.values - reg["clock_offset_s"]          # t_right = t_left + offset
    R, p = {}, {}
    for g in GROUPS:
        R[g] = Rr @ Rot.from_quat(tab[[f"{g}_qx", f"{g}_qy", f"{g}_qz", f"{g}_qw"]].values).as_matrix()
        p[g] = s * tab[[f"{g}_x", f"{g}_y", f"{g}_z"]].values @ Rr.T + tr
    ap_dir = find_aperture_dir(results, episode, side, stem)
    ap = pd.read_csv(ap_dir / f"{stem}_aperture.csv").set_index("frame").reindex(range(frame0, frame0 + len(tab)))
    q = np.radians(ap[["q_left_deg", "q_right_deg"]].interpolate(limit_direction="both").values)
    ap_meta = json.loads((ap_dir / f"{stem}_summary.json").read_text(encoding="utf-8"))
    geom = load_geometry(marker_size_mm=ap_meta["sticker_size_mm"])
    m = np.load(pose_dir / f"{stem}_map.npz")
    files = {lens: find_insv(data, stem, lens) for lens in (0, 1)} if data else {}
    return Rig(side=side, name=name, stem=stem, n=len(tab), fps=float(summ["fps"]), t=t_common.astype(float),
               valid=tab.valid.values.astype(bool), R=R, p=p, q=q, geom=geom, T_lens_base=np.array(summ["T_lens1_base"], float),
               lever_scale=s, map_X=s * m["landmarks" if segment is None else f"s{int(segment)}_landmarks"] @ Rr.T + tr, summary=summ, files=files,
               frame0=frame0)


# ----------------------------------------------------------------------------- phases from the gap
def two_level_threshold(gap: np.ndarray) -> tuple:
    """Hysteresis thresholds between an "open" and a "closed" level of the gap, from its own distribution (2-means)."""
    g = gap[np.isfinite(gap)]
    lo, hi = np.percentile(g, 10), np.percentile(g, 90)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        a, b = g[g < mid], g[g >= mid]
        if not len(a) or not len(b):
            break
        lo2, hi2 = a.mean(), b.mean()
        if abs(lo2 - lo) < 1e-6 and abs(hi2 - hi) < 1e-6:
            break
        lo, hi = lo2, hi2
    sep = hi - lo
    return lo + 0.35 * sep, lo + 0.55 * sep              # close below the first, open above the second


def phases_from_gap(gap: np.ndarray, t: np.ndarray, min_hold_s: float = 0.3, plateau_mm: float = 1.0, grasp_before_s: float = 0.4,
                    grasp_after_s: float = 0.3, approach_s: float = 1.5, release_after_s: float = 0.5):
    """Holds are stretches where the gap is at its closed level (two-level hysteresis).  Inside a hold the CONTACT is the first
    frame at which the gap is within `plateau_mm` of the hold's median - a hand may close most of the way long before it
    reaches the object - and the release is the last such frame.  Returns (phase, hold_id, thresholds, events) with
    events[k] = dict(cross=, contact=, release=, open=, gap_mm=) frame indices."""
    close_below, open_above = two_level_threshold(gap)
    closed = np.zeros(len(gap), bool)
    state = False
    for i, g in enumerate(gap):
        if not np.isfinite(g):
            closed[i] = state
            continue
        if state and g > open_above:
            state = False
        elif not state and g < close_below:
            state = True
        closed[i] = state
    phase = np.full(len(gap), PHASE["transport"])
    hold_id = np.full(len(gap), -1)
    events = []
    dt = np.median(np.diff(t))
    f = lambda s: int(round(s / dt))
    k, i = 0, 0
    while i < len(gap):
        if closed[i]:
            j = i
            while j + 1 < len(gap) and closed[j + 1]:
                j += 1
            if t[j] - t[i] >= min_hold_s:
                g_hold = np.nanmedian(gap[i:j + 1])
                on = np.where(gap[i:j + 1] <= g_hold + plateau_mm / 1000)[0]
                c, r = i + int(on[0]), i + int(on[-1])
                a0 = max(0, min(i - f(0.2), c - f(approach_s)))
                phase[a0:c] = PHASE["approach"]
                hold_id[a0:c] = k
                g0, g1 = max(0, c - f(grasp_before_s)), min(len(gap), c + f(grasp_after_s) + 1)
                phase[g0:g1] = PHASE["grasp"]
                hold_id[g0:g1] = k
                phase[g1:max(g1, r - f(0.2))] = PHASE["hold"]
                hold_id[g1:max(g1, r - f(0.2))] = k
                r0, r1 = max(g1, r - f(0.2)), min(len(gap), max(j + 1, r + f(release_after_s) + 1))
                phase[r0:r1] = PHASE["release"]
                hold_id[r0:r1] = k
                events.append(dict(cross=i, contact=c, release=r, open=j, gap_mm=float(g_hold * 1000)))
                k += 1
            i = j + 1
        else:
            i += 1
    return phase, hold_id, (float(close_below), float(open_above)), events


# ----------------------------------------------------------------------------- the table and the hands
def table_height_from_map(rigs: list) -> float:
    """z of the reference surface in the world: the dominant plane of the map around the workspace (the plate top in Episode-1)."""
    P = np.concatenate([r.map_X for r in rigs])
    ws = np.mean([r.p["tcp"].mean(0) for r in rigs], axis=0)
    band = (np.hypot(P[:, 0] - ws[0], P[:, 1] - ws[1]) < 0.45) & (P[:, 2] > -0.30) & (P[:, 2] < -0.08)
    if band.sum() < 50:
        raise ValueError("too few map points around the workspace to find the table")
    h, e = np.histogram(P[band, 2], bins=np.arange(-0.30, -0.08, 0.005))
    z0 = e[np.argmax(h)] + 0.0025
    return float(np.median(P[band & (np.abs(P[:, 2] - z0) < 0.012), 2]))


@dataclass
class Hand:
    name: str
    t: np.ndarray
    R: np.ndarray
    p: np.ndarray
    gap: np.ndarray
    phase: np.ndarray
    hold_id: np.ndarray
    valid: np.ndarray
    height: np.ndarray
    tips_height: np.ndarray
    pad_to_tip_m: float
    thresholds: tuple
    events: list

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def approach(self) -> np.ndarray:
        return self.R[:, :, 0]


def skeleton_tcp(geom) -> tuple:
    """(hinge midpoint, tcp) in the gripper's base frame at closed jaws: the finger direction."""
    from state_estimation.gripper import jaw_points, tips
    hinges = []
    for jaw in ("L", "R"):
        j = geom.jaws[jaw]
        hinge = np.array([j.pivot[0], j.pivot[1], j.marker_center[2]])
        hinges.append(jaw_points(geom, jaw, hinge[None], 0.0)[0])
    tL, tR = tips(geom, 0.0, 0.0)
    return 0.5 * (hinges[0] + hinges[1]), 0.5 * (np.asarray(tL).reshape(3) + np.asarray(tR).reshape(3))


def hand_from_rig(rig: Rig, table_z: float, gap_offset_m: float) -> Hand:
    from state_estimation.gripper import contact_centers, tips
    geom = rig.geom
    hinge_mid, tcp = skeleton_tcp(geom)
    a_B = tcp - hinge_mid
    a_B /= np.linalg.norm(a_B)                                           # out along the fingers, in the CAD base frame
    o_B = np.array([1.0, 0.0, 0.0])                                      # +x is the opening direction (the CAD's convention)
    o_B = o_B - a_B * (o_B @ a_B)
    o_B /= np.linalg.norm(o_B)
    C = np.stack([a_B, o_B, np.cross(a_B, o_B)], axis=1)
    cL, cR = contact_centers(geom, rig.q[:, 0], rig.q[:, 1])             # (n,3) pad centres, CAD base frame
    tL, tR = tips(geom, rig.q[:, 0], rig.q[:, 1])
    pad_mid, tip_mid = 0.5 * (cL + cR), 0.5 * (tL + tR)
    R_WB = rig.R["tcp"]                                                  # the tcp group carries the base frame's rotation
    p_tcp = rig.p["tcp"]                                                 # ... and the tip midpoint's position
    p_grip = p_tcp + np.einsum("nij,nj->ni", R_WB, pad_mid - tip_mid)
    gap = np.linalg.norm(cL - cR, axis=1) + gap_offset_m
    phase, hold_id, thr, events = phases_from_gap(gap, rig.t)
    return Hand(name=rig.name, t=rig.t.astype(float), R=R_WB @ C, p=p_grip, gap=gap, phase=phase, hold_id=hold_id, valid=rig.valid.copy(),
                height=p_grip[:, 2] - table_z, tips_height=p_tcp[:, 2] - table_z,
                pad_to_tip_m=float(np.median((tip_mid - pad_mid) @ a_B)), thresholds=thr, events=events)


def rest_window(h: Hand, speed_max: float = 0.02, margin_s: float = 0.3) -> np.ndarray:
    """Frames before the first hold in which the hand is still: the gripper docked in its jig."""
    sp = np.linalg.norm(np.gradient(h.p, h.t, axis=0), axis=1)
    first = h.events[0]["cross"] if h.events else len(h.t)
    return (h.t < h.t[first] - margin_s) & h.valid & (sp < speed_max)


def drop_sync_claps(hands: dict, max_s: float = 1.6, within_s: float = 0.6) -> list:
    """A short closing of BOTH grippers at the same moment is the operator's synchronisation clap, not a grasp: it is relabelled
    transport so no solver weight is spent on it.  Returns what was dropped."""
    if len(hands) < 2:
        return []
    dropped = []
    names = list(hands)
    for a, b in ((names[0], names[1]), (names[1], names[0])):
        ha, hb = hands[a], hands[b]
        for k, ev in enumerate(ha.events):
            ta, dur = ha.t[ev["contact"]], ha.t[ev["release"]] - ha.t[ev["contact"]]
            if dur > max_s:
                continue
            partner = [e for e in hb.events if abs(hb.t[e["contact"]] - ta) < within_s and hb.t[e["release"]] - hb.t[e["contact"]] <= max_s]
            if partner:
                m = ha.hold_id == k
                ha.phase[m] = PHASE["transport"]
                ha.hold_id[m] = -1
                ev["sync_clap"] = True
                dropped.append((a, k, float(ta), float(dur)))
    return dropped


# ----------------------------------------------------------------------------- the export
def build_demo(results: Path, episode: str, data: Optional[Path] = None, jig_gap_mm: Optional[float] = 22.4,
               gap_offset_m=0.0, sync_claps: bool = True, hands=("left", "right"), segment: Optional[int] = None) -> dict:
    """Returns dict(episode, fps, table_z, hands: name -> Hand, notes, rigs: name -> Rig).  segment: one episode window of a
    recording estimated per window (its own world, so its own table height and jig calibration)."""
    reg = load_registration(results, episode, segment)
    rigs = {name: load_rig(results, episode, name, data, reg, segment) for name in hands}
    table_z = table_height_from_map(list(rigs.values()))
    out, offsets = {}, {}
    for name, rig in rigs.items():
        off = gap_offset_m[name] if isinstance(gap_offset_m, dict) else float(gap_offset_m)
        h = hand_from_rig(rig, table_z, 0.0)
        if jig_gap_mm is not None:
            m = rest_window(h)
            if m.sum() >= 5:
                off = jig_gap_mm / 1000 - float(np.median(h.gap[m]))
        offsets[name] = off
        out[name] = hand_from_rig(rig, table_z, off)
    claps = drop_sync_claps(out) if sync_claps else []
    fps = float(rigs[hands[0]].fps)
    notes = dict(gap_offset_m=offsets, jig_gap_mm=jig_gap_mm, sync_claps=claps, clock="left rig pose clock",
                 world="left rig world, gravity aligned, z up", clock_offset_s=reg["clock_offset_s"],
                 registration_scale=reg["scale"], clips={n: r.stem for n, r in rigs.items()}, segment=segment)
    return dict(episode=episode, fps=fps, table_z=table_z, hands=out, notes=notes, rigs=rigs)


def describe(demo: dict) -> str:
    lines = [f"{demo['episode']}: {demo['fps']:.2f} fps, reference surface at z = {demo['table_z'] * 1000:.0f} mm in the world"]
    n = demo["notes"]
    if n.get("jig_gap_mm"):
        lines.append(f"  aperture calibrated on the jig ({n['jig_gap_mm']:g} mm while docked): offsets "
                     + ", ".join(f"{k} {v * 1000:+.2f} mm" for k, v in n["gap_offset_m"].items()))
    for name, k, t, dur in n.get("sync_claps", []):
        lines.append(f"  {name} hold {k} at t {t:.2f} s ({dur:.1f} s) is a synchronisation clap, not a grasp: relabelled transport")
    for name, h in demo["hands"].items():
        holds = sorted(set(h.hold_id[h.hold_id >= 0].tolist()))
        lines.append(f"  {name}: {h.n} frames, t {h.t[0]:.2f}..{h.t[-1]:.2f} s, gap {h.gap.min() * 1000:.1f}..{h.gap.max() * 1000:.1f} mm "
                     f"(closed < {h.thresholds[0] * 1000:.1f}, open > {h.thresholds[1] * 1000:.1f}); {len(holds)} holds")
        for k in holds:
            ev = h.events[k]
            c, r = ev["contact"], ev["release"]
            el = lambda i: np.degrees(np.arccos(np.clip(-h.approach[i, 2], -1, 1)))
            lines.append(f"     hold {k}: contact t {h.t[c]:.2f} s at {h.height[c] * 1000:.0f} mm, fingers {el(c):.0f} deg off vertical, "
                         f"gap {ev['gap_mm']:.1f} mm; release t {h.t[r]:.2f} s at {h.height[r] * 1000:.0f} mm")
    return "\n".join(lines)


def write_demo(demo: dict, out: Path) -> Path:
    """Writes <out>.npz and <out>.json in the format `newlithic.demo.load_demo` reads."""
    out = Path(out).with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, dict(episode=demo["episode"], fps=demo["fps"], table_z=demo["table_z"], notes=demo["notes"], phase_codes=PHASE, hands={})
    for name, h in demo["hands"].items():
        for k in ("t", "R", "p", "gap", "phase", "hold_id", "valid", "height", "tips_height"):
            arrays[f"{name}_{k}"] = np.asarray(getattr(h, k))
        meta["hands"][name] = dict(n=int(h.n), pad_to_tip_m=float(h.pad_to_tip_m), thresholds_m=[float(x) for x in h.thresholds],
                                  events=[{k: (bool(v) if isinstance(v, bool) else float(v) if isinstance(v, float) else int(v))
                                           for k, v in e.items()} for e in h.events])
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=_jsonable), encoding="utf-8")
    return out.with_suffix(".npz")


def export_demo(results: Path, episode: str, out: Path, data: Optional[Path] = None, jig_gap_mm: Optional[float] = 22.4,
                sync_claps: bool = True, log=print) -> Path:
    demo = build_demo(results, episode, data, jig_gap_mm=jig_gap_mm, sync_claps=sync_claps)
    log(describe(demo))
    p = write_demo(demo, out)
    log(f"wrote {p} and {p.with_suffix('.json')}")
    return p


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
