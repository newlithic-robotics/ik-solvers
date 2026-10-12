"""Search a base placement for one hand with the Newlithic gripper.

    .venv/Scripts/python.exe scripts/place.py --hand left --step-mm 50 --radius-mm 450 --out out/place
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newlithic.demo import load_demo                                           # noqa: E402
from newlithic.kin import ArmKinematics                                        # noqa: E402
from newlithic.model import ArmConfig, compile_arm, PLATE_THICKNESS, TABLE_Z_BASE   # noqa: E402
from newlithic.placement import key_frames, search                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hand", default="left")
ap.add_argument("--step-mm", type=float, default=50.0)
ap.add_argument("--radius-mm", type=float, default=450.0)
ap.add_argument("--centre", default=None, help="x,y mm in the world to search around (default: the hand's task centroid)")
ap.add_argument("--headings", default=None, help="comma list of headings [deg]")
ap.add_argument("--refine", type=int, default=2)
ap.add_argument("--out", default="out/place")
args = ap.parse_args()

t0 = time.time()
model, spec, cad, parts = compile_arm(ArmConfig(), save=False)
kin = ArmKinematics(model, cad=cad)
demo = load_demo()
idx, w = key_frames(demo, args.hand)
print(f"{len(idx)} key frames, weights sum {w.sum():.1f}")
surface = TABLE_Z_BASE + PLATE_THICKNESS
centre = tuple(float(v) / 1000 for v in args.centre.split(",")) if args.centre else None
headings = [float(v) for v in args.headings.split(",")] if args.headings else None


def prog(n, total, c):
    if n >= 0:
        print(f"  {n}/{total}  last: ({c.placement.x_m*1000:.0f},{c.placement.y_m*1000:.0f},{c.placement.yaw_deg:.0f}) score {c.score:.2f}  [{time.time()-t0:.0f} s]")
    else:
        print(f"  refine {total}: best ({c.placement.x_m*1000:.0f},{c.placement.y_m*1000:.0f},{c.placement.yaw_deg:.1f}) score {c.score:.3f} pos med {c.pos_mm_med:.2f} max {c.pos_mm_max:.1f} mm tilt med {c.tilt_deg_med:.1f} limit {c.limit_min_deg:.1f}")


best = search(kin, demo, args.hand, cad, surface, centre, args.radius_mm / 1000, args.step_mm / 1000, headings, args.refine, progress=prog)
print("best placements (world mm, deg):")
for c in best:
    print(f"  ({c.placement.x_m*1000:7.1f}, {c.placement.y_m*1000:7.1f}, {c.placement.yaw_deg:6.1f})  score {c.score:7.3f}  pos med {c.pos_mm_med:5.2f} max {c.pos_mm_max:5.1f} mm  "
          f"tilt med {c.tilt_deg_med:5.1f} deg  limit margin {c.limit_min_deg:5.1f} deg")
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
(out / f"{args.hand}_placements.json").write_text(json.dumps([dict(x_mm=c.placement.x_m*1000, y_mm=c.placement.y_m*1000, yaw_deg=c.placement.yaw_deg,
                                                                      score=c.score, pos_mm_med=c.pos_mm_med, pos_mm_max=c.pos_mm_max,
                                                                      tilt_deg_med=c.tilt_deg_med, limit_min_deg=c.limit_min_deg) for c in best], indent=1))
print(f"done in {time.time()-t0:.0f} s")
