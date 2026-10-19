"""Judge base placements by what matters in the end: the full bimanual solve of a few episodes.

Candidates for one hand on rings around its task centroid (reference frame, see place_collection.py), facing it; the other
hand's base stays fixed.  Prints, per candidate, the task position error, tilt and collision frames over the sample.

    .venv/Scripts/python.exe scripts/try_placements.py --work <work> --hand right --centre=-22,203 --fixed=-40.3,-171.5,37.5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _job(a):
    rec, ref_pl, max_nfev = a
    from newlithic.demo import PHASE
    from newlithic_lerobot.collection import anchored_placements, reference_anchor
    from newlithic_lerobot.solve import QPConfig, SmoothConfig, solve_episode
    lines = []
    try:
        anchor = dict(xy=np.array(rec["anchor"]["xy"]), heading=rec["anchor"]["heading"])
        ep = solve_episode(Path(rec["demo"]), anchored_placements(anchor, reference_anchor(), ref_pl), qp=QPConfig(), smooth=SmoothConfig(max_nfev=max_nfev),
                           task=rec["task"], log=lines.append)
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")
    out = {}
    for n, h in ep.hands.items():
        task = h.phase != PHASE["transport"]
        out[n] = dict(pos_med=float(np.median(h.pos_err_mm[task])), pos_p95=float(np.percentile(h.pos_err_mm[task], 95)), pos_max=float(h.pos_err_mm.max()),
                      tilt=float(np.median(h.tilt_err_deg[task])), roll=float(np.median(h.roll_err_deg[task])))
    m = [re.search(r"collision frames (\d+)/(\d+), worst penetration ([\d.]+)", ln) for ln in lines]
    m = [x for x in m if x]
    out["collision"] = (int(m[-1][1]), float(m[-1][3])) if m else (0, 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--hand", default="right")
    ap.add_argument("--centre", required=True, help="x,y mm of the hand's task centroid in the reference frame")
    ap.add_argument("--fixed", required=True, help="x,y,yaw of the other hand's base")
    ap.add_argument("--dist", default="260,320")
    ap.add_argument("--bearings", default="0,45,90,135,180,225,270,315")
    ap.add_argument("--yaw-off", default="0", help="yaw offsets from facing the centroid [deg]")
    ap.add_argument("--list", default=None, help="explicit candidates x,y,yaw;x,y,yaw instead of the rings")
    ap.add_argument("--sample", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--max-nfev", type=int, default=40)
    a = ap.parse_args()
    man = json.loads((Path(a.work) / "manifest.json").read_text())
    names = sorted(man["episodes"])
    pick = [names[i] for i in np.linspace(3, len(names) - 4, a.sample).round().astype(int)]
    other = "left" if a.hand == "right" else "right"
    fixed = tuple(float(v) for v in a.fixed.split(","))
    c = np.array([float(v) for v in a.centre.split(",")])
    if a.list:
        cands = [tuple(float(v) for v in s.split(",")) for s in a.list.split(";")]
    else:
        cands = [(c[0] + d * np.cos(np.radians(b)), c[1] + d * np.sin(np.radians(b)), (b + 180 + y + 180) % 360 - 180)
                 for d in map(float, a.dist.split(",")) for b in map(float, a.bearings.split(",")) for y in map(float, a.yaw_off.split(","))]
    jobs = [(man["episodes"][n], {a.hand: pl, other: fixed}, a.max_nfev) for pl in cands for n in pick]
    print(f"{len(cands)} candidates x {len(pick)} episodes {pick}", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(_job, jobs))
    for i, pl in enumerate(cands):
        rs = res[i * len(pick):(i + 1) * len(pick)]
        if any("error" in r for r in rs):
            print(f"({pl[0]:7.1f},{pl[1]:7.1f},{pl[2]:6.1f})  {[r.get('error', 'ok')[:80] for r in rs]}", flush=True)
            continue
        s = "  ".join(f"{n}: pos {np.mean([r[n]['pos_med'] for r in rs]):5.2f}/{np.mean([r[n]['pos_p95'] for r in rs]):5.1f}/{max(r[n]['pos_max'] for r in rs):5.1f} mm "
                      f"tilt {np.mean([r[n]['tilt'] for r in rs]):4.1f} roll {np.mean([r[n]['roll'] for r in rs]):4.1f}" for n in ("left", "right"))
        print(f"({pl[0]:7.1f},{pl[1]:7.1f},{pl[2]:6.1f})  {s}  coll {sum(r['collision'][0] for r in rs)} fr, {max(r['collision'][1] for r in rs):.1f} mm", flush=True)


if __name__ == "__main__":
    main()
