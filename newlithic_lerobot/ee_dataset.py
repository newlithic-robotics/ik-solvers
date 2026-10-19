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

All poses are in the LEFT DOCK frame: origin at the docked left grip (its x, y) on the reference surface (z of the table the
fingers touched), x along the docked left fingers' heading, z up.  The same motion has the same coordinates in every recording
(each recording's own world starts wherever the left rig did).

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
