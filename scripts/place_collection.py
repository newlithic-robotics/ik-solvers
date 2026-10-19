"""Search one base placement per hand for a whole collection of episodes.

Every recording has its own world, so a candidate base is expressed relative to the docked left grip (in Episode-1's world,
the frame of docs/findings.md) and moved into each episode's world through that episode's dock, exactly as the `collection`
driver does.  The score is the mean of `newlithic.placement.score_placement` over a sample of episodes.

    .venv/Scripts/python.exe scripts/place_collection.py --work <work> --hand right --sample 12 --jobs 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_S = {}


def _init(demos: list, hand: str):
    from newlithic.demo import load_demo
    from newlithic.kin import ArmKinematics
    from newlithic.model import ArmConfig, compile_arm, PLATE_THICKNESS, TABLE_Z_BASE
    from newlithic.placement import key_frames
    from newlithic.targets import Tolerances
    from newlithic_lerobot.collection import reference_anchor
    model, _, cad, _ = compile_arm(ArmConfig(), save=False)
    _S.update(kin=ArmKinematics(model, cad=cad), cad=cad, surface=TABLE_Z_BASE + PLATE_THICKNESS, hand=hand, tol=Tolerances(),
              ref=reference_anchor(), eps=[])
    for path, anchor in demos:
        d = load_demo(Path(path))
        idx, w = key_frames(d, hand)
        _S["eps"].append((d, dict(xy=np.array(anchor["xy"]), heading=anchor["heading"]), idx, w))


def _score(pl):
    from newlithic.demo import Placement
    from newlithic.placement import score_placement
    from newlithic_lerobot.collection import anchored_placements
    out = []
    for d, anchor, idx, w in _S["eps"]:
        x, y, yaw = anchored_placements(anchor, _S["ref"], {"h": pl})["h"]
        c = score_placement(_S["kin"], d, _S["hand"], Placement(x / 1000, y / 1000, yaw, _S["surface"]), _S["cad"], idx, w, _S["tol"])
        out.append((c.score, c.pos_mm_med, c.pos_mm_max, c.tilt_deg_med, c.limit_min_deg))
    a = np.array(out)
    return dict(pl=pl, score=float(a[:, 0].mean()), worst=float(a[:, 0].max()), pos_med=float(np.median(a[:, 1])), pos_max=float(a[:, 2].max()),
                tilt_med=float(np.median(a[:, 3])), limit_min=float(a[:, 4].min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--hand", default="right")
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--step-mm", type=float, default=50.0)
    ap.add_argument("--radius-mm", type=float, default=450.0)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--keep", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--also", default=None, help="x,y,yaw to score as well (e.g. the current placement)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from newlithic.demo import PHASE, load_demo
    from newlithic_lerobot.collection import reference_anchor
    man = json.loads((Path(a.work) / "manifest.json").read_text())
    names = sorted(man["episodes"])
    pick = [names[i] for i in np.linspace(0, len(names) - 1, a.sample).round().astype(int)]
    demos = [(man["episodes"][n]["demo"], man["episodes"][n]["anchor"]) for n in pick]
    ref = reference_anchor()
    cen = []
    for path, anchor in demos:                                    # the hand's task centroid, in the reference frame
        h = load_demo(Path(path)).hand(a.hand)
        m = h.phase != PHASE["transport"]
        d = ref["heading"] - anchor["heading"]
        c, s = np.cos(d), np.sin(d)
        cen.append(ref["xy"] + np.array([[c, -s], [s, c]]) @ (h.p[m, :2].mean(0) - np.array(anchor["xy"])))
    c0 = np.mean(cen, 0)
    print(f"{a.hand}: {len(pick)} episodes; task centroid in the reference frame ({c0[0]*1000:.0f}, {c0[1]*1000:.0f}) mm, spread {np.std(cen, 0)*1000} mm", flush=True)

    r, st = a.radius_mm / 1000, a.step_mm / 1000
    xs = np.arange(-r, r + 1e-9, st)
    grid = [(c0[0] + dx, c0[1] + dy, hd) for dx in xs for dy in xs for hd in np.arange(0, 360, 30.0) if 0.12 <= np.hypot(dx, dy) <= r]
    grid = [(x * 1000, y * 1000, hd) for x, y, hd in grid if abs(((np.degrees(np.arctan2(c0[1] - y, c0[0] - x)) - hd + 180) % 360) - 180) < 50]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(demos, a.hand)) as ex:
        if a.also:
            c = ex.submit(_score, tuple(float(v) for v in a.also.split(","))).result()
            print("given:", json.dumps(c), flush=True)
        cands = list(ex.map(_score, grid, chunksize=4))
        print(f"coarse: {len(grid)} candidates in {time.time()-t0:.0f} s", flush=True)
        best = sorted(cands, key=lambda c: c["score"])[:a.keep]
        for level in range(a.refine):
            s, hs = a.step_mm / 2 ** (level + 1), 30.0 / 2 ** (level + 1)
            new = [(b["pl"][0] + dx, b["pl"][1] + dy, b["pl"][2] + dh) for b in best for dx in (-s, 0, s) for dy in (-s, 0, s) for dh in (-hs, 0, hs)
                   if not dx == dy == dh == 0]
            best = sorted(best + list(ex.map(_score, new, chunksize=4)), key=lambda c: c["score"])[:a.keep]
            print(f"refine {level}: best {best[0]}  [{time.time()-t0:.0f} s]", flush=True)
    for b in best:
        print(f"  ({b['pl'][0]:7.1f}, {b['pl'][1]:7.1f}, {b['pl'][2]:6.1f})  score {b['score']:7.3f} worst {b['worst']:7.3f}  pos med {b['pos_med']:5.2f} max {b['pos_max']:5.1f} mm  "
              f"tilt med {b['tilt_med']:5.1f} deg  limit margin {b['limit_min']:5.1f} deg")
    if a.out:
        Path(a.out).write_text(json.dumps(dict(hand=a.hand, episodes=pick, best=best), indent=1))


if __name__ == "__main__":
    main()
