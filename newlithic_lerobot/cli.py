"""Command line: `newlithic-lerobot <stage>` (or `python -m newlithic_lerobot <stage>`).

    demo-export    state-estimation results -> demonstration file          (needs .[demo])
    render-wrist   state-estimation results + raw .insv -> wrist videos     (needs .[demo])
    solve          demonstration + base placements -> retargeted episode    (needs .[solve])
    export         retargeted episode(s) + videos -> LeRobotDataset v3.0    (needs .[export])
    verify         reload a dataset the way training will
    inspect        describe retargeted episode(s)
    pipeline       all of the above for one episode, in one go              (needs .[all])
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_TASK = "Unscrew the M5 nut from the bolt held by the other hand"


def _placement(s: str):
    try:
        x, y, yaw = (float(v) for v in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"placement must be x_mm,y_mm,yaw_deg, got {s!r}")
    return x, y, yaw


def _camera(s: str):
    if "=" not in s:
        raise argparse.ArgumentTypeError("camera must be name=path (path may contain {episode} or {stem})")
    name, path = s.split("=", 1)
    return name.strip(), path.strip()


def _frames(s: str):
    try:
        a, b = (int(v) for v in s.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"frames must be a:b, got {s!r}")
    return a, b


# ------------------------------------------------------------------------------ stages
def cmd_demo_export(a):
    from .demo_export import export_demo
    out = Path(a.out)
    if out.suffix not in (".npz", ".json"):
        out = out / f"{a.episode}.npz"
    export_demo(Path(a.results), a.episode, out, data=Path(a.data) if a.data else None, jig_gap_mm=None if a.no_jig else a.jig_gap_mm,
                sync_claps=not a.keep_sync_claps)


def cmd_render_wrist(a):
    from .wrist import render_wrist_videos
    render_wrist_videos(Path(a.results), a.episode, Path(a.data), Path(a.out), workers=a.workers, frames=a.frames, W=a.width, H=a.height,
                        hfov_deg=a.hfov_deg, tcp_row=a.tcp_row, src_scale=a.src_scale, crf=a.crf)


def cmd_solve(a):
    from .solve import QPConfig, SmoothConfig, solve_episode
    ep = solve_episode(a.demo, {"left": a.left, "right": a.right}, qp=QPConfig(iterations=a.iterations, rel_orient_deg=a.rel_orient_deg),
                       smooth=SmoothConfig(enabled=not a.no_smooth, v_max_deg_s=a.v_max_deg_s, a_max_deg_s2=a.a_max_deg_s2, max_nfev=a.max_nfev),
                       collisions=not a.no_collisions, task=a.task, frames=a.frames)
    out = Path(a.out)
    if out.suffix not in (".npz", ".json"):
        out = out / f"{ep.episode}_retargeted.npz"
    written = ep.save(out)
    print(ep.describe())
    print("wrote", written, "and", written.with_suffix(".json"))
    return written


def cmd_collection(a):
    from .collection import DEFAULT_TASKS, run_collection
    tasks = dict(DEFAULT_TASKS)
    for item in a.task:
        k, _, v = item.partition("=")
        tasks[int(k)] = v
    stages = tuple(s.strip() for s in a.stages.split(",") if s.strip())
    if "export" in stages and not (a.dataset and a.repo_id):
        raise SystemExit("the export stage needs --dataset and --repo-id")
    if "export-ee" in stages and not (a.ee_dataset and a.ee_repo_id):
        raise SystemExit("the export-ee stage needs --ee-dataset and --ee-repo-id")
    m = run_collection([Path(s) for s in a.session], Path(a.work), Path(a.dataset) if a.dataset else None, a.repo_id, stages=stages, jobs=a.jobs,
                       fps=a.fps, tasks=tasks, only=a.only, force=a.force, solve_kw=dict(max_nfev=a.max_nfev),
                       export_kw=dict(vcodec=a.vcodec, extras=not a.no_extras), render=not a.no_render,
                       placements_mm=({"left": a.left, "right": a.right} if a.left and a.right else None),
                       ee_dataset=Path(a.ee_dataset) if a.ee_dataset else None, ee_repo_id=a.ee_repo_id)
    print(f"{len(m['episodes'])} episodes in the manifest, {len(m['failed'])} recordings failed -> {Path(a.work) / 'manifest.json'}")


def _export_config(a):
    from .export import ExportConfig
    from .features import Units, load_calibration
    cal = None
    if a.calibration_dir:
        d = Path(a.calibration_dir)
        cal = {"left": load_calibration(d / a.calibration_left), "right": load_calibration(d / a.calibration_right)}
    units = Units(joints=a.joint_units, calibration=cal, gripper_open_percent=a.gripper_open_percent)
    v_max = getattr(a, "export_v_max_deg_s", None)
    if v_max is None:
        v_max = a.v_max_deg_s
    return ExportConfig(fps=a.fps, action_lead=a.action_lead, v_max_deg_s=v_max, units=units, extras=not a.no_extras,
                        start_s=a.start_s, end_s=a.end_s, vcodec=a.vcodec, image_writer_threads=a.image_writer_threads)


def cmd_export(a):
    from .export import verify, write_dataset
    write_dataset(a.episodes, Path(a.out), a.repo_id, _export_config(a), cameras=dict(a.camera or []), task=a.task, resume=a.resume)
    if a.verify:
        verify(Path(a.out), a.repo_id)


def cmd_verify(a):
    from .export import verify
    verify(Path(a.out), a.repo_id)


def cmd_inspect(a):
    from .episode import RetargetedEpisode
    for p in a.episodes:
        ep = RetargetedEpisode.load(p)
        print(ep.describe())
        if a.solver:
            import json
            print(json.dumps(ep.solver, indent=1, default=str))


def cmd_pipeline(a):
    """results + raw data -> demonstration -> wrist videos -> retargeted episode -> dataset, under one work directory."""
    from .demo_export import export_demo
    from .export import verify, write_dataset
    from .solve import QPConfig, SmoothConfig, solve_episode
    from .wrist import render_wrist_videos
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    demo = export_demo(Path(a.results), a.episode, work / f"{a.episode}.npz", data=Path(a.data), jig_gap_mm=None if a.no_jig else a.jig_gap_mm,
                       sync_claps=not a.keep_sync_claps)
    videos = render_wrist_videos(Path(a.results), a.episode, Path(a.data), work, workers=a.workers, frames=a.frames)
    ep = solve_episode(demo, {"left": a.left, "right": a.right}, qp=QPConfig(iterations=a.iterations, rel_orient_deg=a.rel_orient_deg),
                       smooth=SmoothConfig(enabled=not a.no_smooth, v_max_deg_s=a.v_max_deg_s, a_max_deg_s2=a.a_max_deg_s2, max_nfev=a.max_nfev),
                       collisions=not a.no_collisions, task=a.task, frames=a.frames)
    retargeted = ep.save(work / f"{a.episode}_retargeted.npz")
    print(ep.describe())
    if a.frames:
        # the videos were rendered for the same slice: they are matched by index after shifting to the slice's first frame
        print(f"NOTE: --frames {a.frames[0]}:{a.frames[1]}: the retargeted episode and the videos both cover only that slice")
    cameras = {"left_wrist": str(videos["left"]), "right_wrist": str(videos["right"])}
    write_dataset([retargeted], Path(a.dataset), a.repo_id, _export_config(a), cameras=cameras, task=a.task, resume=a.resume)
    verify(Path(a.dataset), a.repo_id)


# ------------------------------------------------------------------------------ argument groups
def _add_demo_args(p, with_data_required: bool):
    p.add_argument("--results", required=True, help="the state-estimation results directory (pose/, aperture*/)")
    p.add_argument("--episode", required=True, help="episode name as in the results directory, e.g. Episode-1")
    p.add_argument("--data", required=with_data_required, help="directory holding the raw Insta360 .insv files (searched recursively)")
    p.add_argument("--jig-gap-mm", type=float, default=22.4, help="the pad gap while docked in the jig: calibrates each hand's aperture")
    p.add_argument("--no-jig", action="store_true", help="do not calibrate the aperture on the rest window")
    p.add_argument("--keep-sync-claps", action="store_true", help="keep simultaneous short closings as grasps")


def _add_wrist_args(p):
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=880)
    p.add_argument("--hfov-deg", type=float, default=100.0)
    p.add_argument("--tcp-row", type=float, default=0.70, help="where the TCP sits in the image height (0 top .. 1 bottom)")
    p.add_argument("--src-scale", type=int, default=None, help="shrink the fisheye frames by this before remapping (default: to about 1440 px)")
    p.add_argument("--crf", type=int, default=18)


def _add_solve_args(p):
    p.add_argument("--left", required=True, type=_placement, help="left base x_mm,y_mm,yaw_deg in the demonstration world (scripts/place.py)")
    p.add_argument("--right", required=True, type=_placement, help="right base x_mm,y_mm,yaw_deg")
    p.add_argument("--iterations", type=int, default=3, help="QP sub-steps per frame")
    p.add_argument("--rel-orient-deg", type=float, default=4.0, help="sigma of the right-relative-to-left orientation task")
    p.add_argument("--no-smooth", action="store_true", help="skip the trajectory optimiser (QP result only)")
    p.add_argument("--v-max-deg-s", dest="v_max_deg_s", type=float, default=90.0, help="trajectory optimiser: speed the path should not need")
    p.add_argument("--a-max-deg-s2", type=float, default=600.0)
    p.add_argument("--max-nfev", type=int, default=40)
    p.add_argument("--no-collisions", action="store_true", help="skip the collision report")
    p.add_argument("--frames", type=_frames, help="a:b, solve only this slice of frames (smoke tests)")
    p.add_argument("--task", default=DEFAULT_TASK, help="task string stored with the episode")


def _add_export_args(p, v_max_dest: str = "v_max_deg_s"):
    p.add_argument("--repo-id", required=True, help="hub id the dataset is written under, e.g. newlithic/so101_unscrew")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--action-lead", type=int, default=1, help="action = state this many frames later")
    if v_max_dest == "v_max_deg_s":
        p.add_argument("--v-max-deg-s", dest="v_max_deg_s", type=float, default=0.0, help="> 0: stretch the clock where a joint would exceed this")
    else:
        p.add_argument("--export-v-max-deg-s", dest=v_max_dest, type=float, default=0.0, help="export: > 0 stretches the dataset clock where a joint would exceed this")
    p.add_argument("--joint-units", choices=("degrees", "range_m100_100"), default="degrees")
    p.add_argument("--calibration-dir", help="lerobot calibration directory (needed for range_m100_100)")
    p.add_argument("--calibration-left", default="left.json")
    p.add_argument("--calibration-right", default="right.json")
    p.add_argument("--gripper-open-percent", type=float, default=100.0, help="what the jaw stop maps to: 100 (default) or 0")
    p.add_argument("--no-extras", action="store_true", help="only state / action / images")
    p.add_argument("--start-s", type=float)
    p.add_argument("--end-s", type=float)
    p.add_argument("--vcodec", choices=("libsvtav1", "h264", "hevc"), default="libsvtav1")
    p.add_argument("--image-writer-threads", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="append to an existing dataset")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="newlithic-lerobot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo-export", help="state-estimation results -> demonstration file (grip poses, pad gap, phases)")
    _add_demo_args(d, with_data_required=False)
    d.add_argument("--out", required=True, help="output .npz, or a directory (then <episode>.npz)")
    d.set_defaults(fn=cmd_demo_export)

    w = sub.add_parser("render-wrist", help="state-estimation results + raw .insv -> the two wrist-camera videos")
    _add_demo_args(w, with_data_required=True)
    _add_wrist_args(w)
    w.add_argument("--frames", type=_frames, help="a:b sub-range of the left clip's frames (tests)")
    w.add_argument("--out", required=True, help="output directory")
    w.set_defaults(fn=cmd_render_wrist)

    s = sub.add_parser("solve", help="demonstration -> retargeted episode (bimanual QP + trajectory optimiser)")
    s.add_argument("--demo", required=True, help="the demonstration file (.npz with its .json next to it)")
    _add_solve_args(s)
    s.add_argument("--out", required=True, help="output .npz, or a directory (then <episode>_retargeted.npz)")
    s.set_defaults(fn=cmd_solve)

    e = sub.add_parser("export", help="retargeted episode(s) + videos -> LeRobotDataset v3.0")
    e.add_argument("episodes", nargs="+", help="retargeted episode .npz files, in dataset order")
    e.add_argument("--out", required=True, help="dataset root directory")
    e.add_argument("--camera", action="append", type=_camera, metavar="NAME=PATH",
                   help="a video on the demonstration's clock; PATH may contain {episode} or {stem}; repeat per camera")
    _add_export_args(e)
    e.add_argument("--task", help="task string (default: the episode's)")
    e.add_argument("--verify", action="store_true", help="reload the dataset afterwards and decode a frame")
    e.set_defaults(fn=cmd_export)

    v = sub.add_parser("verify", help="reload a dataset the way training will")
    v.add_argument("--out", required=True)
    v.add_argument("--repo-id", required=True)
    v.set_defaults(fn=cmd_verify)

    i = sub.add_parser("inspect", help="describe retargeted episode(s)")
    i.add_argument("episodes", nargs="+")
    i.add_argument("--solver", action="store_true", help="also print the solver record")
    i.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("pipeline", help="everything for one episode: results + raw data -> dataset")
    _add_demo_args(p, with_data_required=True)
    _add_wrist_args(p)
    _add_solve_args(p)
    _add_export_args(p, v_max_dest="export_v_max_deg_s")
    p.add_argument("--work", required=True, help="directory for the demonstration file, the videos and the retargeted episode")
    p.add_argument("--dataset", required=True, help="dataset root directory")
    p.set_defaults(fn=cmd_pipeline)

    c = sub.add_parser("collection", help="sessions of reviewed episodes (many per recording, any recording mode) -> one dataset")
    c.add_argument("--session", action="append", required=True, help="a state-estimation session folder (pose/, aperture/, raw/, "
                                                                     "episodes/episodes_reviewed.json); repeat per session")
    c.add_argument("--work", required=True, help="directory for the per-episode demonstrations, wrist videos and retargeted episodes")
    c.add_argument("--dataset", help="dataset root directory (export stage)")
    c.add_argument("--repo-id", help="hub id the dataset is written under")
    c.add_argument("--stages", default="prepare,solve,export", help="comma-separated subset of prepare,solve,export,export-ee,reanchor-ee "
                                                                     "(reanchor-ee: the end-effector dataset's poses in the anchors of the last prepare)")
    c.add_argument("--ee-dataset", help="end-effector training dataset root (export-ee stage): grip poses and openings, no joints")
    c.add_argument("--ee-repo-id", help="hub id the end-effector dataset is written under")
    c.add_argument("--jobs", type=int, default=8, help="recordings prepared / episodes solved in parallel")
    c.add_argument("--fps", type=int, default=30)
    c.add_argument("--task", action="append", default=[], metavar="ID=TEXT", help="task string for a spoken task id; repeat per id")
    c.add_argument("--only", help="only recordings / episodes whose name contains this text")
    c.add_argument("--force", action="store_true", help="redo what already exists")
    c.add_argument("--max-nfev", type=int, default=40, help="trajectory optimiser budget per arm")
    c.add_argument("--vcodec", choices=("libsvtav1", "h264", "hevc"), default="libsvtav1")
    c.add_argument("--no-extras", action="store_true", help="only state / action / images in the dataset")
    c.add_argument("--no-render", action="store_true", help="prepare the demonstrations only, no wrist videos")
    c.add_argument("--left", type=_placement, help="left base x_mm,y_mm,yaw_deg relative to the left dock (Episode-1's world); default: the findings'")
    c.add_argument("--right", type=_placement, help="right base, same frame")
    c.set_defaults(fn=cmd_collection)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
