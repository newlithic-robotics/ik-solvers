"""The training dataset: reviewed demonstrations -> LeRobotDataset v3.0 of what the glove measured, no joints.

The arms' joints are not part of it: a policy trained on it outputs grip poses and openings, and whoever runs it on an arm
adds the inverse kinematics of that arm at inference time (the IK solver of this repository is kept for QA: can an arm follow
the demonstration?).

Per frame, on the prepared demonstration's grid (collection.prepare_recording: `fps`, both hands on one clock):

    observation.state (20)   per hand, left then right:
                               x, y, z [m]     the grip origin (midpoint of the two pad contact centres)
                               r1x .. r2z      6D rotation: the first two columns of the grip's rotation (approach along the fingers,
                                               opening from the R jaw toward the L jaw); the third is their cross product
                               gap [m]         pad-to-pad opening, calibrated on the docking jig
    action (20)              the state one frame later (the last frame repeats): the pose the hands move to next
    observation.images.left_wrist / right_wrist   the rendered wrist views of the same instant
    observation.phase (2)    transport 0, grasp 1, hold 2, release 3, approach 4, per hand (context, not for control)

All poses are in the episode's LEFT DOCK frame: origin at the left grip as it rested just before the episode (its x, y;
collection.episode_rest) on the reference surface (z of the table the fingers touched), x along those resting fingers' heading,
z up.  Every episode starts from its own dock: where the hand rested between episodes varies by centimetres and degrees, and an
episode estimated in its own window (state-estimation, after QA) has no other reference.  The same motion has the same
coordinates in every episode.  reanchor_ee_dataset rewrites a dataset's poses when the dock frames change, keeping its videos.

Adding sessions: the current dataset is copied to <root>.next, the new episodes are appended there, the copy is reloaded and
counted, and only then does it replace <root>; the replaced dataset is kept as <root>.previous-<time>.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__

HANDS = ("left", "right")
FIELDS = ("x", "y", "z", "r1x", "r1y", "r1z", "r2x", "r2y", "r2z", "gap")
ROBOT_TYPE = "newlithic_glove_bimanual_ee"
CAMERAS = {"left_wrist": "left", "right_wrist": "right"}


def names() -> list:
    return [f"{h}_{f}" for h in HANDS for f in FIELDS]


def dock_frame(anchor: dict) -> tuple:
    """(Rz, origin): world -> dock is  p_dock = Rz @ (p_world - origin),  R_dock = Rz @ R_world."""
    c, s = np.cos(anchor["heading"]), np.sin(anchor["heading"])
    Rz = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])           # rotation by -heading about z
    origin = np.array([anchor["xy"][0], anchor["xy"][1], anchor["table_z"]], float)
    return Rz, origin


def ee_frames(demo_path, anchor: dict, action_lead: int = 1) -> dict:
    """state, action (m, 20), phase (m, 2), valid (m, 2) of one prepared demonstration."""
    d = np.load(demo_path)
    Rz, origin = dock_frame(anchor)
    cols, phase, valid = [], [], []
    for h in HANDS:
        p, R = d[f"{h}_p"], d[f"{h}_R"]
        pd = (p - origin) @ Rz.T
        Rd = np.einsum("ij,njk->nik", Rz, R)
        cols += [pd, Rd[:, :, 0], Rd[:, :, 1], d[f"{h}_gap"][:, None]]
        phase.append(d[f"{h}_phase"].astype(float))
        valid.append(d[f"{h}_valid"])
    state = np.concatenate(cols, axis=1)
    if not np.isfinite(state).all():
        raise ValueError(f"{demo_path}: non-finite poses")
    lead = np.minimum(np.arange(len(state)) + action_lead, len(state) - 1)
    return {"state": state, "action": state[lead], "phase": np.stack(phase, 1), "valid": np.stack(valid, 1), "n": len(state)}


def features(cam_shapes: dict) -> dict:
    n = names()
    f = {"observation.state": {"dtype": "float32", "shape": (len(n),), "names": n},
         "action": {"dtype": "float32", "shape": (len(n),), "names": n},
         "observation.phase": {"dtype": "float32", "shape": (2,), "names": [f"{h}_phase" for h in HANDS]}}
    for cam, (hgt, wid) in cam_shapes.items():
        f[f"observation.images.{cam}"] = {"dtype": "video", "shape": (hgt, wid, 3), "names": ["height", "width", "channels"]}
    return f


def _write(ds, rec: dict, fr: dict, log) -> int:
    from .video import VideoFrames
    videos = {cam: VideoFrames(rec["videos"][hand]) for cam, hand in CAMERAS.items()}
    try:
        for cam, v in videos.items():
            if v.n != fr["n"]:
                log(f"  WARNING {rec['name']} {cam}: {v.n} video frames for {fr['n']} poses; frames are matched by index")
        for k in range(fr["n"]):
            frame = {"observation.state": fr["state"][k].astype(np.float32), "action": fr["action"][k].astype(np.float32),
                     "observation.phase": fr["phase"][k].astype(np.float32), "task": rec.get("task") or ""}
            for cam, v in videos.items():
                frame[f"observation.images.{cam}"] = v.read(min(k, v.n - 1))
            ds.add_frame(frame)
        ds.save_episode()
    finally:
        for v in videos.values():
            v.close()
    return fr["n"]


def write_ee_dataset(records: list, root: Path, repo_id: str, fps: int = 30, vcodec: str = "libsvtav1", image_writer_threads: int = 4,
                     log=print) -> dict:
    """Adds `records` (collection manifest episode records: demo, videos, anchor, task, name) to the dataset at `root`, creating
    it when absent.  Episodes already in it (by name, from meta/newlithic_ee.json) are skipped.  Returns the dataset summary."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .export import _codec_kwargs
    from .video import VideoFrames
    root = Path(root)
    side = "newlithic_ee.json"
    existing = json.loads((root / "meta" / side).read_text(encoding="utf-8")) if (root / "meta" / side).exists() else None
    have = {e["episode"] for e in existing["episodes"]} if existing else set()
    todo = [r for r in records if r["name"] not in have]
    if not todo:
        log(f"{root}: all {len(records)} episodes are already in the dataset")
        return {"root": str(root), "added": 0, "episodes": len(have)}
    with VideoFrames(todo[0]["videos"]["left"]) as v:
        shape = (v.height, v.width)
    feats = features({cam: shape for cam in CAMERAS})
    work = root.with_name(root.name + ".next")
    if work.exists():
        shutil.rmtree(work)                                                 # a run that did not finish
    if existing:
        log(f"copying {root.name} ({len(have)} episodes) to {work.name}")
        shutil.copytree(root, work)
        ds = LeRobotDataset(repo_id, root=work)
        if dict(ds.meta.features).get("observation.state", {}).get("names") != feats["observation.state"]["names"]:
            raise ValueError(f"{root} was written with a different state layout; export to a new dataset folder")
    else:
        ds = LeRobotDataset.create(repo_id, fps=fps, features=feats, root=work, robot_type=ROBOT_TYPE, use_videos=True,
                                   image_writer_threads=image_writer_threads, **_codec_kwargs(vcodec))
    record = existing or {"newlithic_lerobot": __version__, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "robot_type": ROBOT_TYPE, "frame": "left dock: origin at the docked left grip on the table plane, x along its heading, z up",
                          "state": names(), "action": "state one frame later", "fps": fps, "episodes": []}
    total = 0
    for i, rec in enumerate(todo, 1):
        fr = ee_frames(rec["demo"], rec["anchor"])
        n = _write(ds, rec, fr, log)
        total += n
        record["episodes"].append({"episode": rec["name"], "session": rec.get("session"), "recording": rec.get("recording"), "frames": n,
                                   "task": rec.get("task"), "task_id": rec.get("task_id"), "anchor": rec["anchor"], "demo": rec["demo"]})
        log(f"  [{i}/{len(todo)}] {rec['name']}: {n} frames")
    ds.finalize()
    (work / "meta" / side).write_text(json.dumps(record, indent=1), encoding="utf-8")
    check = LeRobotDataset(repo_id, root=work)                               # reload the way training will before it replaces anything
    want = len(have) + len(todo)
    if int(check.meta.total_episodes) != want:
        raise ValueError(f"{work} holds {check.meta.total_episodes} episodes, expected {want}; the current dataset was left alone")
    check[len(check) - 1]
    prev = None
    if root.exists():
        prev = root.with_name(f"{root.name}.previous-{datetime.now():%Y%m%d_%H%M%S}")
        root.rename(prev)
    work.rename(root)
    log(f"done: {len(todo)} episode(s), {total} frames added -> {root} ({want} episodes)" + (f"; the previous dataset is {prev.name}" if prev else ""))
    return {"root": str(root), "added": len(todo), "episodes": want, "previous": str(prev) if prev else None}


def reanchor_ee_dataset(root: Path, anchors: dict, repo_id: str | None = None, log=print) -> dict:
    """Rewrites the poses of an end-effector dataset in new dock frames and keeps everything else, the wrist videos above all
    (re-encoding them is most of an export's time).  anchors: episode name -> anchor (collection manifest records).

    Works on <root>.reanchor (videos and images hard-linked, the rest copied): observation.state and action of every frame, the
    per-episode statistics of those two features (meta/episodes), the dataset's (meta/stats.json, with lerobot's own functions)
    and the anchors in meta/newlithic_ee.json.  The copy is loaded with lerobot and checked against the new poses before it
    replaces the dataset; the dataset it replaces is kept as <root>.previous-<time>."""
    import os
    import shutil
    from datetime import datetime

    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats

    root = Path(root)
    side = json.loads((root / "meta" / "newlithic_ee.json").read_text(encoding="utf-8"))
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    eps = side["episodes"]
    if len(eps) != info["total_episodes"]:
        raise ValueError(f"meta/newlithic_ee.json lists {len(eps)} episodes, the dataset has {info['total_episodes']}")
    missing = [e["episode"] for e in eps if e["episode"] not in anchors]
    if missing:
        raise ValueError(f"{len(missing)} episode(s) without a new anchor, e.g. {missing[:3]}: prepare their sessions first")
    work = root.with_name(root.name + ".reanchor")
    if work.exists():
        shutil.rmtree(work)
    linked = 0
    for src in root.rglob("*"):
        dst = work / src.relative_to(root)
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.relative_to(root).parts[0] in ("videos", "images"):
            try:
                os.link(src, dst)
                linked += 1
                continue
            except OSError:
                pass
        shutil.copy2(src, dst)
    log(f"reanchor: {len(eps)} episodes; working copy {work.name} ({linked} video files linked, not copied)")

    frames = {k: ee_frames(e["demo"], anchors[e["episode"]]) for k, e in enumerate(eps)}
    feats = info["features"]
    keys = ("observation.state", "action")
    ep_stats = {k: compute_episode_stats({"observation.state": fr["state"].astype(np.float32), "action": fr["action"].astype(np.float32)},
                                         {key: feats[key] for key in keys}) for k, fr in frames.items()}

    def column(values: np.ndarray, typ):
        flat = pa.array(values.astype(np.float32).ravel(), type=pa.float32())
        if pa.types.is_fixed_size_list(typ):
            arr = pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
        else:
            arr = pa.ListArray.from_arrays(pa.array(np.arange(0, values.size + 1, values.shape[1]), type=pa.int32()), flat)
        return arr.cast(typ)

    rows_done = 0
    for f in sorted((work / "data").rglob("*.parquet")):
        t = pq.read_table(f)
        ep_idx, fr_idx = t.column("episode_index").to_numpy(), t.column("frame_index").to_numpy()
        new = {key: np.stack(t.column(key).to_numpy(zero_copy_only=False)).astype(np.float32) for key in keys}
        for k in np.unique(ep_idx):
            rows = np.flatnonzero(ep_idx == k)
            fr = frames[int(k)]
            if len(rows) != fr["n"] or fr_idx[rows].max() >= fr["n"]:
                raise ValueError(f"episode {eps[int(k)]['episode']}: {len(rows)} rows in the dataset, {fr['n']} frames in its demonstration")
            new["observation.state"][rows] = fr["state"][fr_idx[rows]]
            new["action"][rows] = fr["action"][fr_idx[rows]]
            rows_done += len(rows)
        for key in keys:
            i = t.schema.get_field_index(key)
            t = t.set_column(i, t.schema.field(i), column(new[key], t.schema.field(i).type))
        pq.write_table(t, f)
    if rows_done != info["total_frames"]:
        raise ValueError(f"{rows_done} rows rewritten, the dataset has {info['total_frames']}")

    for f in sorted((work / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(f)
        idx = t.column("episode_index").to_numpy()
        for key in keys:
            for stat in ep_stats[0][key]:
                name = f"stats/{key}/{stat}"
                if name not in t.column_names:
                    continue
                i = t.schema.get_field_index(name)
                vals = [np.asarray(ep_stats[int(k)][key][stat]).reshape(-1).tolist() for k in idx]
                t = t.set_column(i, t.schema.field(i), pa.array(vals).cast(t.schema.field(i).type))
        pq.write_table(t, f)
    stats = json.loads((work / "meta" / "stats.json").read_text(encoding="utf-8"))
    agg = aggregate_stats([ep_stats[k] for k in sorted(ep_stats)])
    for key in keys:
        stats[key] = {s: np.asarray(v).tolist() for s, v in agg[key].items()}
    (work / "meta" / "stats.json").write_text(json.dumps(stats, indent=4), encoding="utf-8")
    for k, e in enumerate(eps):
        e["anchor"] = anchors[e["episode"]]
    side["reanchored"] = datetime.now().isoformat(timespec="seconds")
    (work / "meta" / "newlithic_ee.json").write_text(json.dumps(side, indent=1), encoding="utf-8")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset                # the copy must load, and say what was written
    ds = LeRobotDataset(repo_id or side.get("repo_id") or "local/reanchor", root=work)
    if len(ds) != info["total_frames"]:
        raise ValueError(f"the rewritten dataset loads with {len(ds)} frames, not {info['total_frames']}")
    for k in sorted({0, len(eps) // 2, len(eps) - 1}):
        start = int(ds.meta.episodes[k]["dataset_from_index"]) if hasattr(ds.meta, "episodes") else None
        if start is None:
            continue
        got = np.asarray(ds.hf_dataset[start]["observation.state"], np.float32)
        if not np.allclose(got, frames[k]["state"][0].astype(np.float32), atol=1e-6):
            raise ValueError(f"episode {eps[k]['episode']}: the rewritten dataset does not hold the new poses")
    prev = root.with_name(f"{root.name}.previous-{datetime.now():%Y%m%d_%H%M%S}")
    root.rename(prev)
    work.rename(root)
    log(f"reanchor: done, {rows_done} frames of {len(eps)} episodes in their own dock frames -> {root}; the previous dataset is {prev.name}")
    return {"root": str(root), "episodes": len(eps), "frames": rows_done, "previous": str(prev)}

