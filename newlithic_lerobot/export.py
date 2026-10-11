"""Stage 2: retargeted episodes (+ wrist videos) -> a LeRobotDataset v3.0.

Per output frame (uniform at `fps`):
    observation.state   the solved joints at that instant (what the arms are at), in the chosen units
    action              the state `action_lead` frames later: the setpoint the arms are moving toward.  lerobot-record stores
                        the leader arm's position next to the follower's, which is this same "where to go next" relation.
    observation.images.<camera>   the demonstration's camera frame nearest to that instant
    observation.grip_pose / phase / gap_mm   (optional) the demonstrated grip pose per hand in its arm's base frame, the
                        phase label and the pad gap - context for analysis and for policies that want it

Nothing here needs MuJoCo.  `write_dataset` needs lerobot; `build_frames` and `write_episode` do not (the dataset object is
duck-typed: add_frame / save_episode), so the frame logic is unit-testable without it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from .episode import RetargetedEpisode
from .features import ROBOT_TYPE, Units, dataset_features, motor_names
from .timing import Clock, interp, interp_pose7, make_clock

VALID_CODECS = ("libsvtav1", "h264", "hevc")


@dataclass
class ExportConfig:
    fps: int = 30
    action_lead: int = 1                       # frames between observation.state and action
    v_max_deg_s: float = 0.0                   # > 0: stretch the clock where a joint would exceed this (see timing.py)
    units: Units = field(default_factory=Units)
    extras: bool = True
    start_s: Optional[float] = None            # trim the episode (demonstration clock, relative to its first frame)
    end_s: Optional[float] = None
    vcodec: str = "libsvtav1"
    image_writer_threads: int = 4

    def __post_init__(self):
        if self.fps <= 0 or int(self.fps) != self.fps:
            raise ValueError("fps must be a positive integer (LeRobot stores timestamps as k / fps)")
        if self.action_lead < 0:
            raise ValueError("action_lead must be >= 0")
        if self.vcodec not in VALID_CODECS:
            raise ValueError(f"vcodec must be one of {VALID_CODECS}")


@dataclass
class EpisodeFrames:
    clock: Clock
    state: np.ndarray            # (m, 6 * hands)
    action: np.ndarray           # (m, 6 * hands)
    extras: dict                 # feature name -> (m, k)
    names: list

    @property
    def m(self) -> int:
        return len(self.state)


def _window(ep: RetargetedEpisode, cfg: ExportConfig) -> np.ndarray:
    t = ep.t - ep.t[0]
    keep = np.ones(ep.n, bool)
    if cfg.start_s is not None:
        keep &= t >= cfg.start_s
    if cfg.end_s is not None:
        keep &= t <= cfg.end_s
    if keep.sum() < 2:
        raise ValueError("the start/end window leaves fewer than two frames")
    return keep


def build_frames(ep: RetargetedEpisode, cfg: ExportConfig) -> EpisodeFrames:
    keep = _window(ep, cfg)
    t = (ep.t - ep.t[0])[keep]
    hands = list(ep.hands)
    Qs = [ep.hands[h].q_deg[keep] for h in hands]
    clock = make_clock(t, cfg.fps, Qs=Qs, v_max_deg_s=cfg.v_max_deg_s)
    clock.src_index = np.where(keep)[0][clock.src_index]            # back to indices into the full episode (the videos)
    open_stop = float(ep.gripper["open_stop_deg"])
    cols, extras_pose, extras_phase, extras_gap = [], [], [], []
    for h in hands:
        hd = ep.hands[h]
        q = interp(t, hd.q_deg[keep], clock.t_demo)
        jaw = interp(t, hd.jaw_deg[keep], clock.t_demo)
        cols.append(cfg.units.arm(q, h))
        cols.append(cfg.units.gripper(jaw, open_stop)[:, None])
        extras_pose.append(interp_pose7(t, hd.pose_demo[keep], clock.t_demo))
        extras_phase.append(hd.phase[clock.src_index][:, None].astype(float))
        extras_gap.append(interp(t, hd.gap_mm[keep], clock.t_demo)[:, None])
    state = np.concatenate(cols, axis=1)
    lead = np.minimum(np.arange(len(state)) + cfg.action_lead, len(state) - 1)
    action = state[lead]
    extras = {}
    if cfg.extras:
        extras = {"observation.grip_pose": np.concatenate(extras_pose, axis=1),
                  "observation.phase": np.concatenate(extras_phase, axis=1),
                  "observation.gap_mm": np.concatenate(extras_gap, axis=1)}
    return EpisodeFrames(clock=clock, state=state, action=action, extras=extras, names=motor_names(hands))


def write_episode(ds, frames: EpisodeFrames, videos: dict, task: str, log=None) -> int:
    """videos: camera name -> object with .read(index) -> (h,w,3) uint8.  Returns the number of frames written."""
    for k in range(frames.m):
        frame = {"observation.state": frames.state[k].astype(np.float32), "action": frames.action[k].astype(np.float32), "task": task}
        for name, arr in frames.extras.items():
            frame[name] = arr[k].astype(np.float32)
        for cam, src in videos.items():
            frame[f"observation.images.{cam}"] = src.read(int(frames.clock.src_index[k]))
        ds.add_frame(frame)
        if log and k % 300 == 0:
            log(f"  frame {k}/{frames.m}")
    ds.save_episode()
    return frames.m


def camera_paths(spec: dict, ep: RetargetedEpisode, path: Path) -> dict:
    """spec: camera name -> path pattern; `{episode}` is the episode name, `{stem}` the episode file's stem."""
    return {cam: Path(str(p).format(episode=ep.episode, stem=Path(path).with_suffix("").name)) for cam, p in spec.items()}


def write_dataset(episode_paths, root: Path, repo_id: str, cfg: ExportConfig, cameras: Optional[dict] = None,
                  task: Optional[str] = None, resume: bool = False, log=print) -> dict:
    """episode_paths: retargeted episodes (.npz).  cameras: name -> video path pattern (see camera_paths)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from .video import VideoFrames
    root = Path(root)
    cameras = dict(cameras or {})
    episodes = [(Path(p), RetargetedEpisode.load(p)) for p in episode_paths]
    if not episodes:
        raise ValueError("no episodes given")
    hands = list(episodes[0][1].hands)
    for p, ep in episodes:
        if list(ep.hands) != hands:
            raise ValueError(f"{p}: hands {list(ep.hands)} differ from the first episode's {hands}")
    cam_shapes = {}
    if cameras:
        first = camera_paths(cameras, episodes[0][1], episodes[0][0])
        for cam, vp in first.items():
            with VideoFrames(vp) as v:
                cam_shapes[cam] = (v.height, v.width)
    features = dataset_features(hands, cam_shapes, extras=cfg.extras)
    if resume and (root / "meta" / "info.json").exists():
        ds = LeRobotDataset(repo_id, root=root)
        log(f"resuming {root}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"{root} exists and is not empty (use --resume to append)")
        ds = LeRobotDataset.create(repo_id, fps=cfg.fps, features=features, root=root, robot_type=ROBOT_TYPE, use_videos=bool(cameras),
                                   image_writer_threads=cfg.image_writer_threads if cameras else 0, vcodec=cfg.vcodec)
    record = dict(newlithic_lerobot=__version__, created=datetime.now(timezone.utc).isoformat(timespec="seconds"), robot_type=ROBOT_TYPE,
                  export=_cfg_dict(cfg), cameras={k: str(v) for k, v in cameras.items()}, features=features, episodes=[])
    total = 0
    for path, ep in episodes:
        log(f"episode {ep.episode}: {ep.describe().splitlines()[0]}")
        frames = build_frames(ep, cfg)
        videos = {}
        try:
            for cam, vp in camera_paths(cameras, ep, path).items():
                v = VideoFrames(vp)
                if v.n != ep.n:
                    log(f"  WARNING {cam}: {v.n} video frames for {ep.n} demonstration frames ({vp}); frames are matched by index")
                if (v.height, v.width) != cam_shapes[cam]:
                    raise ValueError(f"{cam}: {vp} is {v.width}x{v.height}, the dataset expects {cam_shapes[cam][1]}x{cam_shapes[cam][0]}")
                videos[cam] = v
            n = write_episode(ds, frames, videos, task or ep.task or "", log=log)
        finally:
            for v in videos.values():
                v.close()
        total += n
        log(f"  wrote {n} frames at {cfg.fps} fps ({frames.clock.t_out[-1]:.1f} s; demonstration {ep.duration_s:.1f} s)")
        record["episodes"].append(dict(source=str(path), episode=ep.episode, frames=n, demo_frames=int(ep.n), task=task or ep.task,
                                       placements=ep.placements, gripper=ep.gripper, solver=ep.solver, provenance=ep.provenance,
                                       time_warped=bool(cfg.v_max_deg_s and cfg.v_max_deg_s > 0)))
    ds.finalize()
    side = root / "meta" / "newlithic_export.json"
    existing = json.loads(side.read_text(encoding="utf-8")) if side.exists() else None
    if existing and resume:
        existing["episodes"].extend(record["episodes"])
        record = existing
    side.write_text(json.dumps(record, indent=1, default=_jsonable), encoding="utf-8")
    log(f"done: {total} frames in {len(episodes)} episode(s) -> {root}")
    return record


def verify(root: Path, repo_id: str, log=print) -> dict:
    """Reload the dataset the way training will and decode one frame per camera."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=Path(root))
    m = ds.meta
    out = dict(episodes=int(m.total_episodes), frames=int(m.total_frames), fps=int(m.fps), robot_type=m.robot_type,
               features=list(m.features), video_keys=list(m.video_keys))
    log(f"{repo_id}: {out['episodes']} episode(s), {out['frames']} frames at {out['fps']} fps, robot {out['robot_type']}")
    item = ds[0]
    last = ds[len(ds) - 1]
    for k in ("observation.state", "action"):
        log(f"  {k}: {tuple(item[k].shape)} {m.features[k]['names']}")
        log(f"     first {np.round(item[k].numpy(), 2).tolist()}")
    for k in m.video_keys:
        log(f"  {k}: {tuple(item[k].shape)} (C,H,W) float in [{float(item[k].min()):.2f}, {float(item[k].max()):.2f}]")
        out[k] = list(item[k].shape)
    log(f"  timestamps {float(item['timestamp']):.3f} .. {float(last['timestamp']):.3f} s, task {item['task']!r}")
    return out


def _cfg_dict(cfg: ExportConfig) -> dict:
    d = asdict(cfg)
    d["units"] = cfg.units.describe()
    return d


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
