"""Command line: `newlithic-lerobot <stage>` (or `python -m newlithic_lerobot <stage>`).

    demo-export    state-estimation results -> demonstration file          (needs .[demo])
    render-wrist   state-estimation results + raw .insv -> wrist videos     (needs .[demo])
    solve          demonstration + base placements -> retargeted episode    (needs .[solve]; the IK check)
    inspect        describe retargeted episode(s)
    collection     sessions of reviewed episodes -> prepared episodes, the IK check, and the end-effector training
                   dataset (export-ee, needs .[export]); what the data station runs
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
    if "export-ee" in stages and not (a.ee_dataset and a.ee_repo_id):
        raise SystemExit("the export-ee stage needs --ee-dataset and --ee-repo-id")
    m = run_collection([Path(s) for s in a.session], Path(a.work), stages=stages, jobs=a.jobs,
                       fps=a.fps, tasks=tasks, only=a.only, force=a.force, solve_kw=dict(max_nfev=a.max_nfev),
                       export_kw=dict(vcodec=a.vcodec), render=not a.no_render,
                       placements_mm=({"left": a.left, "right": a.right} if a.left and a.right else None),
                       ee_dataset=Path(a.ee_dataset) if a.ee_dataset else None, ee_repo_id=a.ee_repo_id)
    print(f"{len(m['episodes'])} episodes in the manifest, {len(m['failed'])} recordings failed -> {Path(a.work) / 'manifest.json'}")


def cmd_inspect(a):
    from .episode import RetargetedEpisode
    for p in a.episodes:
        ep = RetargetedEpisode.load(p)
        print(ep.describe())
        if a.solver:
            import json
            print(json.dumps(ep.solver, indent=1, default=str))


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

    i = sub.add_parser("inspect", help="describe retargeted episode(s)")
    i.add_argument("episodes", nargs="+")
    i.add_argument("--solver", action="store_true", help="also print the solver record")
    i.set_defaults(fn=cmd_inspect)

    c = sub.add_parser("collection", help="sessions of reviewed episodes (many per recording, any recording mode) -> one dataset")
    c.add_argument("--session", action="append", required=True, help="a state-estimation session folder (pose/, aperture/, raw/, "
                                                                     "episodes/episodes_reviewed.json); repeat per session")
    c.add_argument("--work", required=True, help="directory for the per-episode demonstrations, wrist videos and retargeted episodes")
    c.add_argument("--stages", default="prepare,solve", help="comma-separated subset of prepare,solve,export-ee,reanchor-ee "
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
    c.add_argument("--no-render", action="store_true", help="prepare the demonstrations only, no wrist videos")
    c.add_argument("--left", type=_placement, help="left base x_mm,y_mm,yaw_deg relative to the left dock (Episode-1's world); default: the findings'")
    c.add_argument("--right", type=_placement, help="right base, same frame")
    c.set_defaults(fn=cmd_collection)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
