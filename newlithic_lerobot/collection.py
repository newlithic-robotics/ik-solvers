"""A whole collection of recordings -> one dataset: many reviewed episodes per recording, any recording mode.

`pipeline` turns ONE recording into ONE episode.  The glove sessions are different: each recording (one LHS + one RHS clip,
3K at 100 fps, four to six minutes) holds about nine demonstrations, cut by the operator's spoken "start" / "end" and reviewed
by a person.  The state-estimation work folder of a session has

    <session>/pose/<recording>/...  aperture/<recording>/...  raw/<recording>/<LHS|RHS>/*.insv     as `demo-export` reads them
    <session>/episodes/episodes_reviewed.json      the reviewed episodes: recording, index, task_id, start_s, end_s (left clip's clock)

Per recording (`prepare`): the demonstration of the whole recording is built once (the aperture calibration on the docking jig
and the phase labels need the whole signal), then every episode is cut out of it on an exact `fps` grid (default 30): poses
interpolated (slerp), labels taken from the nearest frame, holds renumbered.  Its two wrist videos are rendered for the same
instants, one frame per demonstration frame.

Placement.  Every recording has its own world: the left rig's, whose origin and heading are set while that rig sits in its
docking jig.  The arm bases must stand the same way relative to the scene in every recording, or the same motion gives
different joint angles from one recording to the next.  So the placements of the findings (Episode-1) are carried over through
the LEFT dock: the planar transform that maps Episode-1's docked left grip (position, heading of the fingers) onto this
recording's is applied to both bases.  One fixed pair of arms; the right jig, which the operator puts down by hand, is wherever
it was that day.

Then `solve` per episode (in parallel processes) and one `export` over all of them.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from .demo_export import PHASE, build_demo, rest_window, write_demo

# Bases relative to the docked left grip, written in Episode-1's world.  Head-on to the two-handed work of the glove
# collection (the hands meet at about (-40, 170) mm, the left approaching along 67 deg, the right along -112 deg, to within
# 5 deg from episode to episode): each base 350-400 mm behind its hand on that line, which leaves 9 / 14 deg of tilt the
# 5-DoF arm cannot reach instead of 30.  docs/findings.md's Episode-1 bases are (-40.3, -171.5, 37.5), (336.5, 525.8, 225.0).
REF_PLACEMENTS_MM = {"left": (-190.0, -180.0, 67.0), "right": (121.0, 552.0, -112.0)}
DEFAULT_TASKS = {0: "Unscrew the M5 nut from the bolt held by the other hand"}


# ----------------------------------------------------------------------------- placement through the left dock
def episode_rest(h, t_start: float, before_s: float = 2.5, after_s: float = 0.5, speed_max: float = 0.02, margin_s: float = 0.3) -> np.ndarray:
    """The docked rest just before an episode: the still frames from `before_s` ahead of its start to `after_s` after it, and
    before its first grasp.  Each episode is measured from its own dock, so where the hand rested between episodes (it varies by
    centimetres and degrees) does not shift the episode."""
    sp = np.linalg.norm(np.gradient(h.p, h.t, axis=0), axis=1)
    cross = [h.t[ev["cross"]] for ev in h.events if h.t[ev["cross"]] >= t_start - before_s]
    end = min(t_start + after_s, (cross[0] if cross else h.t[-1]) - margin_s)
    return (h.t >= t_start - before_s) & (h.t < end) & h.valid & (sp < speed_max)


def _anchor(p: np.ndarray, R: np.ndarray, rest: np.ndarray) -> dict:
    """The docked grip of a hand: planar position [m] and the heading of the fingers [rad] in the world."""
    if rest.sum() < 5:
        raise ValueError("no still stretch before the first grasp: the docked rest is the placement's reference")
    a = np.median(R[rest][:, :, 0], axis=0)
    return dict(xy=np.median(p[rest, :2], axis=0), heading=float(np.arctan2(a[1], a[0])), frames=int(rest.sum()))


def reference_anchor() -> dict:
    """Episode-1's docked left grip, from the demonstration shipped in assets/."""
    from newlithic.demo import load_demo
    h = load_demo(drop_non_grasps=False).hand("left")
    sp = np.linalg.norm(np.gradient(h.p, h.t, axis=0), axis=1)
    first = int(np.where(h.hold_id >= 0)[0][0])
    return _anchor(h.p, h.R, (h.t < h.t[first] - 0.3) & h.valid & (sp < 0.02))


def anchored_placements(anchor: dict, ref: dict, placements_mm: dict = REF_PLACEMENTS_MM) -> dict:
    """The reference placements moved by the planar transform that takes the reference dock onto this recording's."""
    d = anchor["heading"] - ref["heading"]
    c, s = np.cos(d), np.sin(d)
    Rz = np.array([[c, -s], [s, c]])
    out = {}
    for name, (x, y, yaw) in placements_mm.items():
        xy = anchor["xy"] + Rz @ (np.array([x, y]) / 1000.0 - ref["xy"])
        out[name] = (float(xy[0] * 1000), float(xy[1] * 1000), float((yaw + np.degrees(d) + 180) % 360 - 180))
    return out


# ----------------------------------------------------------------------------- one episode out of a recording
def slice_hand(h, grid: np.ndarray):
    """A demo_export.Hand on the instants `grid` (inside its span): continuous quantities interpolated, labels nearest."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    g = np.clip(grid, h.t[0], h.t[-1])
    i = np.clip(np.searchsorted(h.t, g), 1, h.n - 1)
    near = np.where(np.abs(h.t[i - 1] - g) <= np.abs(h.t[i] - g), i - 1, i)
    lin = lambda a: np.stack([np.interp(g, h.t, a[:, k]) for k in range(a.shape[1])], 1) if a.ndim == 2 else np.interp(g, h.t, a)
    phase, hold = h.phase[near].copy(), h.hold_id[near].copy()
    ids = [int(k) for k in sorted(set(hold[hold >= 0].tolist()))]
    events = []
    for new, k in enumerate(ids):
        ev = dict(h.events[k])
        for key in ("cross", "contact", "release", "open"):
            if key in ev:
                ev[key] = int(np.argmin(np.abs(g - h.t[int(ev[key])])))
        events.append(ev)
        hold[h.hold_id[near] == k] = new
    return SimpleNamespace(name=h.name, t=grid.astype(float), R=Slerp(h.t, Rot.from_matrix(h.R))(g).as_matrix(), p=lin(h.p), gap=lin(h.gap),
                           phase=phase, hold_id=hold, valid=h.valid[near].copy(), height=lin(h.height), tips_height=lin(h.tips_height),
                           pad_to_tip_m=h.pad_to_tip_m, thresholds=h.thresholds, events=events, n=len(grid))


def render_episode(rigs: dict, views: dict, grid: np.ndarray, paths: dict, fps: int, crf: int = 18) -> None:
    """The two wrist videos of an episode, one frame per instant of `grid` (each rig's nearest frame)."""
    from .wrist import Reader, writer
    idx = {n: r.frame_at(grid) + r.frame0 for n, r in rigs.items()}          # video frames (a rig may start mid-clip)
    rd = {n: Reader(rigs[n].files[1], idx[n][0], views[n].src_scale, views[n].x0, views[n].lens_w) for n in rigs}
    ff = {n: writer(paths[n], views[n].W, views[n].H, crf=crf, fps=str(fps)) for n in rigs}
    try:
        for k in range(len(grid)):
            for n in rigs:
                ff[n].stdin.write(views[n].render(rd[n].get(idx[n][k])).tobytes())
    finally:
        for n in rigs:
            ff[n].stdin.close()
            ff[n].wait()
            rd[n].cap.release()


def prepare_recording(session: str, recording: str, episodes: list, out_dir: str, fps: int = 30, jig_gap_mm: Optional[float] = 22.4,
                      tasks: Optional[dict] = None, render: bool = True, force: bool = False, skipped: Optional[dict] = None) -> list:
    """Writes <out_dir>/<recording>_eNN.npz/.json (+ _wrist_left/right.mp4) for every reviewed episode; returns their records.
    A recording estimated per episode window is prepared window by window, each in its own world with its own dock frame, table
    height and jig calibration; an episode whose window could not be prepared goes into `skipped` (name -> why)."""
    from .demo_export import segment_of_episodes
    session = Path(session)
    seg_of = segment_of_episodes(session, recording)
    if seg_of is None:
        return _prepare_part(session, recording, episodes, out_dir, fps, jig_gap_mm, tasks, render, force, None)
    records = []
    by_seg: dict = {}
    for e in episodes:
        if int(e["index"]) not in seg_of:
            (skipped if skipped is not None else {})[f"{recording}_e{int(e['index']):02d}"] = "its episode window was not estimated or not put on one clock"
            continue
        by_seg.setdefault(seg_of[int(e["index"])], []).append(e)
    for sg, eps in sorted(by_seg.items()):
        try:
            records += _prepare_part(session, recording, eps, out_dir, fps, jig_gap_mm, tasks, render, force, sg)
        except Exception as ex:  # noqa: BLE001 - one window must not cost the recording's others
            for e in eps:
                (skipped if skipped is not None else {})[f"{recording}_e{int(e['index']):02d}"] = f"{type(ex).__name__}: {ex}"
    if not records and episodes:
        raise RuntimeError("no episode window of this recording could be prepared: " + "; ".join(sorted(set((skipped or {}).values())))[:500])
    return records


def _prepare_part(session: Path, recording: str, episodes: list, out_dir: str, fps: int, jig_gap_mm: Optional[float], tasks: Optional[dict],
                  render: bool, force: bool, segment: Optional[int]) -> list:
    """prepare_recording for one world: the whole recording, or one episode window (segment)."""
    from .wrist import WristView
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = {int(k): v for k, v in (tasks or DEFAULT_TASKS).items()}
    data = session / "raw" / recording
    demo = build_demo(session, recording, data if render else None, jig_gap_mm=jig_gap_mm, segment=segment)
    left = demo["hands"]["left"]
    views: dict = {}                                                         # made when a video has to be rendered (needs the raw clips)
    t_lo = max(h.t[0] for h in demo["hands"].values())
    t_hi = min(h.t[-1] for h in demo["hands"].values())
    records = []
    for e in episodes:
        name = f"{recording}_e{int(e['index']):02d}"
        a, b = max(float(e["start_s"]), t_lo), min(float(e["end_s"]), t_hi)
        grid = a + np.arange(int(np.floor((b - a) * fps)) + 1) / fps
        if len(grid) < 2 * fps:
            raise ValueError(f"{name}: only {len(grid)} frames between {a:.2f} and {b:.2f} s")
        hands = {n: slice_hand(h, grid) for n, h in demo["hands"].items()}
        valid = float(np.mean([h.valid.mean() for h in hands.values()]))
        rest = episode_rest(left, float(e["start_s"]))
        if rest.sum() < 5:
            raise ValueError(f"{name}: no still stretch of the left hand just before the episode (its dock)")
        anchor = _anchor(left.p, left.R, rest)
        anchor_rec = dict(xy=anchor["xy"].tolist(), heading=anchor["heading"], frames=anchor["frames"], table_z=float(demo["table_z"]),
                          rest="episode", **({} if segment is None else {"segment": int(segment)}))
        notes = {**demo["notes"], "recording": recording, "session": f"{session.parent.name}/{session.name}", "episode_index": int(e["index"]),
                 "window_s": [a, b], "cue_s": [e.get("cue_start_s"), e.get("cue_stop_s")], "task_id": e.get("task_id"),
                 "resampled_fps": fps, "source_fps": demo["fps"],
                 "placement_anchor": dict(xy_m=anchor["xy"].tolist(), heading_deg=float(np.degrees(anchor["heading"])), frames=anchor["frames"])}
        demo_path = out / f"{name}.npz"
        if force or not demo_path.exists():
            write_demo(dict(episode=name, fps=float(fps), table_z=demo["table_z"], hands=hands, notes=notes), demo_path)
        videos = {n: str(out / f"{name}_wrist_{n}.mp4") for n in hands}
        if render and (force or not all(Path(v).exists() and Path(v).stat().st_size > 1000 for v in videos.values())):
            if not views:
                views.update({n: WristView(r) for n, r in demo["rigs"].items()})
            render_episode(demo["rigs"], views, grid, videos, fps)
        task_id = e.get("task_id")
        records.append(dict(name=name, session=notes["session"], recording=recording, index=int(e["index"]), demo=str(demo_path), videos=videos,
                            frames=int(len(grid)), duration_s=float(grid[-1] - grid[0]), valid_fraction=valid, anchor=anchor_rec,
                            task_id=task_id, task=tasks.get(task_id if task_id is not None else 0, f"task {task_id}"),
                            holds={n: len(h.events) for n, h in hands.items()}))
    return records


def _prepare_job(kw):
    try:
        skipped: dict = {}
        return dict(ok=True, recording=kw["recording"], records=prepare_recording(**kw, skipped=skipped), skipped=skipped)
    except Exception as ex:  # noqa: BLE001 - one bad recording must not stop the collection
        return dict(ok=False, recording=kw["recording"], error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc())


def _solve_job(rec: dict, solve_kw: dict, force: bool, ref_placements: dict, ref: dict):
    from .solve import QPConfig, SmoothConfig, solve_episode
    out = Path(rec["demo"]).with_name(rec["name"] + "_retargeted.npz")
    if out.exists() and not force:
        return dict(ok=True, name=rec["name"], path=str(out), skipped=True)
    try:
        t0 = time.time()
        lines = []
        anchor = dict(xy=np.array(rec["anchor"]["xy"]), heading=rec["anchor"]["heading"])
        placements = anchored_placements(anchor, ref, ref_placements)
        ep = solve_episode(Path(rec["demo"]), placements, qp=QPConfig(), smooth=SmoothConfig(**solve_kw), task=rec["task"], log=lines.append)
        ep.save(out)
        Path(rec["demo"]).with_name(rec["name"] + "_solve.log").write_text("\n".join(lines + [ep.describe()]), encoding="utf-8")
        return dict(ok=True, name=rec["name"], path=str(out), seconds=time.time() - t0, summary={n: _brief(ep, n) for n in ep.hands})
    except Exception as ex:  # noqa: BLE001
        return dict(ok=False, name=rec["name"], error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc())


def _brief(ep, n: str) -> dict:
    h = ep.hands[n]
    task = h.phase != PHASE["transport"]
    sp = np.abs(np.diff(h.q_deg, axis=0) / np.diff(ep.t)[:, None])
    return dict(pos_err_mm_median_task=float(np.median(h.pos_err_mm[task])) if task.any() else None, pos_err_mm_median=float(np.median(h.pos_err_mm)),
                pos_err_mm_p95=float(np.percentile(h.pos_err_mm, 95)), pos_err_mm_max=float(h.pos_err_mm.max()),
                tilt_deg_median_task=float(np.median(h.tilt_err_deg[task])) if task.any() else None,
                speed_deg_s_p95=float(np.percentile(sp, 95)), speed_deg_s_max=float(sp.max()), q_start_deg=h.q_deg[0].round(1).tolist())


# ----------------------------------------------------------------------------- the collection
def load_reviewed(session: Path) -> dict:
    """recording -> its reviewed episodes."""
    d = json.loads((Path(session) / "episodes" / "episodes_reviewed.json").read_text(encoding="utf-8"))
    out: dict = {}
    for e in d["episodes"]:
        out.setdefault(e["recording"], []).append(e)
    return out


def run_collection(sessions: list, work: Path, dataset: Optional[Path] = None, repo_id: Optional[str] = None, stages=("prepare", "solve", "export"),
                   jobs: int = 8, fps: int = 30, tasks: Optional[dict] = None, only: Optional[str] = None, force: bool = False,
                   solve_kw: Optional[dict] = None, export_kw: Optional[dict] = None, render: bool = True,
                   placements_mm: Optional[dict] = None, ee_dataset: Optional[Path] = None, ee_repo_id: Optional[str] = None,
                   log=print) -> dict:
    """placements_mm: hand -> (x_mm, y_mm, yaw_deg) of the bases in the reference frame (Episode-1's world, i.e. relative to
    the left dock); default: the findings' Episode-1 placements.
    Stages: prepare, solve, export (the joint dataset of the solved episodes), export-ee (the end-effector training dataset of
    the prepared episodes at `ee_dataset`, no solve needed; see ee_dataset.py)."""
    work = Path(work)
    ep_dir = work / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"episodes": {}, "failed": {}}
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):           # the workers are the parallelism
        os.environ.setdefault(var, "2")

    if "prepare" in stages:
        todo = []
        for s in sessions:
            for rec, eps in sorted(load_reviewed(Path(s)).items()):
                if only and only not in rec:
                    continue
                todo.append(dict(session=str(s), recording=rec, episodes=eps, out_dir=str(ep_dir), fps=fps, tasks=tasks, force=force, render=render))
        log(f"prepare: {len(todo)} recordings, {sum(len(t['episodes']) for t in todo)} episodes, {min(jobs, len(todo))} at a time")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=max(1, min(jobs, len(todo)))) as ex:
            for fut in as_completed([ex.submit(_prepare_job, kw) for kw in todo]):
                r = fut.result()
                if r["ok"]:
                    for rec in r["records"]:
                        manifest["episodes"][rec["name"]] = {**manifest["episodes"].get(rec["name"], {}), **rec}
                        manifest["failed"].pop(rec["name"], None)
                    manifest["failed"].pop(r["recording"], None)
                    for name, why in r.get("skipped", {}).items():                # one episode window that could not be prepared
                        manifest["failed"][name] = why
                        log(f"  {name}: SKIPPED {why}")
                    log(f"  {r['recording']}: {len(r['records'])} episodes prepared ({time.time() - t0:.0f} s)")
                else:
                    manifest["failed"][r["recording"]] = r["error"]
                    log(f"  {r['recording']}: FAILED {r['error']}\n{r['trace']}")
                manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    names = sorted(n for n in manifest["episodes"] if not only or only in n)
    if "solve" in stages:
        ref_placements = {k: tuple(v) for k, v in (placements_mm or REF_PLACEMENTS_MM).items()}
        ref = reference_anchor()
        manifest["placements_reference_mm"] = ref_placements
        log(f"solve: {len(names)} episodes, {jobs} at a time; bases relative to the left dock (reference frame): {ref_placements}")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=max(1, min(jobs, len(names)))) as ex:
            futs = [ex.submit(_solve_job, manifest["episodes"][n], solve_kw or {}, force, ref_placements, ref) for n in names]
            for k, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                e = manifest["episodes"][r["name"]]
                if r["ok"]:
                    e["retargeted"] = r["path"]
                    e.pop("solve_error", None)
                    if not r.get("skipped"):
                        e["solve"] = r["summary"]
                        e["solve_seconds"] = r["seconds"]
                        log(f"  [{k}/{len(names)}] {r['name']}: " + ", ".join(
                            f"{n} {v['pos_err_mm_median']:.2f} mm (p95 {v['pos_err_mm_p95']:.1f}), {v['speed_deg_s_p95']:.0f} deg/s" for n, v in r["summary"].items())
                            + f"  ({r['seconds']:.0f} s, {time.time() - t0:.0f} s elapsed)")
                else:
                    e["solve_error"] = r["error"]
                    log(f"  [{k}/{len(names)}] {r['name']}: FAILED {r['error']}\n{r['trace']}")
                if k % 5 == 0 or k == len(names):
                    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    if "export" in stages:
        from .export import ExportConfig, write_dataset
        good = [n for n in names if manifest["episodes"][n].get("retargeted") and not manifest["episodes"][n].get("solve_error")]
        if not good:
            raise ValueError("nothing solved to export")
        cameras = {"left_wrist": str(ep_dir / "{episode}_wrist_left.mp4"), "right_wrist": str(ep_dir / "{episode}_wrist_right.mp4")}
        log(f"export: {len(good)} episodes -> {dataset}")
        write_dataset([manifest["episodes"][n]["retargeted"] for n in good], Path(dataset), repo_id, ExportConfig(fps=fps, **(export_kw or {})),
                      cameras=cameras, log=log)
        manifest["dataset"] = dict(root=str(dataset), repo_id=repo_id, episodes=good)
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if "reanchor-ee" in stages:                                                # the dataset's poses in each episode's own dock frame
        from .ee_dataset import reanchor_ee_dataset
        res = reanchor_ee_dataset(Path(ee_dataset), {n: e["anchor"] for n, e in manifest["episodes"].items() if e.get("anchor")}, ee_repo_id, log=log)
        manifest["ee_dataset"] = {**manifest.get("ee_dataset", {}), "root": res["root"], "reanchored": res}
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if "export-ee" in stages:
        from .ee_dataset import write_ee_dataset
        ready = [manifest["episodes"][n] for n in names if Path(manifest["episodes"][n].get("demo", "")).exists()
                 and all(Path(v).exists() for v in manifest["episodes"][n].get("videos", {}).values())]
        if not ready:
            raise ValueError("no prepared episodes (with wrist videos) to export")
        log(f"export-ee: {len(ready)} prepared episodes -> {ee_dataset}")
        res = write_ee_dataset(ready, Path(ee_dataset), ee_repo_id, fps=fps, vcodec=(export_kw or {}).get("vcodec", "libsvtav1"), log=log)
        manifest["ee_dataset"] = dict(root=res["root"], repo_id=ee_repo_id, episodes=res["episodes"], previous=res.get("previous"))
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest
