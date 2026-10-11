"""Command line: `newlithic-lerobot solve | export | verify | inspect` (or `python -m newlithic_lerobot ...`)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


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


def cmd_solve(a):
    from .solve import QPConfig, SmoothConfig, controller_root, solve_episode
    root = controller_root(a.controller_root)
    placements = {"left": a.left, "right": a.right}
    ep = solve_episode(a.demo, placements, root, qp=QPConfig(iterations=a.iterations, rel_orient_deg=a.rel_orient_deg),
                       smooth=SmoothConfig(enabled=not a.no_smooth, v_max_deg_s=a.v_max_deg_s, a_max_deg_s2=a.a_max_deg_s2, max_nfev=a.max_nfev),
                       collisions=not a.no_collisions, task=a.task)
    out = Path(a.out)
    if out.suffix not in (".npz", ".json"):
        out = out / f"{ep.episode}_retargeted.npz"
    written = ep.save(out)
    print(ep.describe())
    print("wrote", written, "and", written.with_suffix(".json"))


def cmd_export(a):
    from .export import ExportConfig, verify, write_dataset
    from .features import Units, load_calibration
    cal = None
    if a.calibration_dir:
        d = Path(a.calibration_dir)
        cal = {"left": load_calibration(d / a.calibration_left), "right": load_calibration(d / a.calibration_right)}
    units = Units(joints=a.joint_units, calibration=cal, gripper_open_percent=a.gripper_open_percent)
    cfg = ExportConfig(fps=a.fps, action_lead=a.action_lead, v_max_deg_s=a.v_max_deg_s, units=units, extras=not a.no_extras,
                       start_s=a.start_s, end_s=a.end_s, vcodec=a.vcodec, image_writer_threads=a.image_writer_threads)
    write_dataset(a.episodes, Path(a.out), a.repo_id, cfg, cameras=dict(a.camera or []), task=a.task, resume=a.resume)
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="newlithic-lerobot", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve", help="demonstration export -> retargeted episode (needs the controller package: MuJoCo + mink)")
    s.add_argument("--demo", required=True, help="the demonstration export (.npz with its .json next to it)")
    s.add_argument("--left", required=True, type=_placement, help="left base x_mm,y_mm,yaw_deg in the demonstration world")
    s.add_argument("--right", required=True, type=_placement, help="right base x_mm,y_mm,yaw_deg")
    s.add_argument("--out", required=True, help="output .npz, or a directory (then <episode>_retargeted.npz)")
    s.add_argument("--controller-root", help="checkout of the controller repository (default: $NEWLITHIC_CONTROLLER)")
    s.add_argument("--iterations", type=int, default=3, help="QP sub-steps per frame")
    s.add_argument("--rel-orient-deg", type=float, default=4.0, help="sigma of the right-relative-to-left orientation task")
    s.add_argument("--no-smooth", action="store_true", help="skip the trajectory optimiser (QP result only)")
    s.add_argument("--v-max-deg-s", type=float, default=90.0, help="trajectory optimiser: speed the path should not need")
    s.add_argument("--a-max-deg-s2", type=float, default=600.0)
    s.add_argument("--max-nfev", type=int, default=40)
    s.add_argument("--no-collisions", action="store_true", help="skip the collision report")
    s.add_argument("--task", default="Unscrew the M5 nut from the bolt held by the other hand", help="task string stored with the episode")
    s.set_defaults(fn=cmd_solve)

    e = sub.add_parser("export", help="retargeted episode(s) + videos -> LeRobotDataset v3.0 (needs lerobot)")
    e.add_argument("episodes", nargs="+", help="retargeted episode .npz files, in dataset order")
    e.add_argument("--out", required=True, help="dataset root directory")
    e.add_argument("--repo-id", required=True, help="hub id the dataset is written under, e.g. newlithic/so101_unscrew")
    e.add_argument("--camera", action="append", type=_camera, metavar="NAME=PATH",
                   help="a video on the demonstration's clock; PATH may contain {episode} or {stem}. Repeat per camera, e.g. "
                        "left_wrist=out/{stem}_wrist_left.mp4")
    e.add_argument("--fps", type=int, default=30)
    e.add_argument("--action-lead", type=int, default=1, help="action = state this many frames later")
    e.add_argument("--v-max-deg-s", type=float, default=0.0, help="> 0: stretch the clock where a joint would exceed this")
    e.add_argument("--joint-units", choices=("degrees", "range_m100_100"), default="degrees")
    e.add_argument("--calibration-dir", help="lerobot calibration directory (needed for range_m100_100)")
    e.add_argument("--calibration-left", default="left.json")
    e.add_argument("--calibration-right", default="right.json")
    e.add_argument("--gripper-open-percent", type=float, default=100.0, help="what the jaw stop maps to: 100 (default) or 0")
    e.add_argument("--no-extras", action="store_true", help="only state / action / images")
    e.add_argument("--start-s", type=float)
    e.add_argument("--end-s", type=float)
    e.add_argument("--vcodec", choices=("libsvtav1", "h264", "hevc"), default="libsvtav1")
    e.add_argument("--image-writer-threads", type=int, default=4)
    e.add_argument("--task", help="task string (default: the episode's)")
    e.add_argument("--resume", action="store_true", help="append to an existing dataset at --out")
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

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
