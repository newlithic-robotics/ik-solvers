"""Reading and writing solved trajectories (CSV per hand + JSON summary)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .demo import PHASE_NAME
from .ik_wls import Solution
from .model import JOINTS
from .targets import Targets


def solution_frame(sol: Solution, tg: Targets, cad) -> pd.DataFrame:
    d = {"t": sol.t, "phase": [PHASE_NAME[int(p)] for p in sol.phase], "hold_id": sol.hold_id}
    for k, n in enumerate(JOINTS):
        d[f"{n}_deg"] = np.degrees(sol.q[:, k])
    d["jaw_deg"] = np.degrees(sol.jaw)
    d["gap_mm"] = cad.gap_mm(np.degrees(sol.jaw))
    d["pos_err_mm"], d["tilt_err_deg"], d["roll_err_deg"], d["lost_deg"], d["limit_margin_deg"] = \
        sol.pos_err_mm, sol.tilt_err_deg, sol.roll_err_deg, sol.lost_deg, sol.limit_margin_deg
    for a, ax in enumerate("xyz"):
        d[f"target_{ax}_mm"] = tg.p_b[:, a] * 1000
        d[f"target_approach_{ax}"] = tg.R_b[:, a, 0]
        d[f"target_opening_{ax}"] = tg.R_b[:, a, 1]
    d["height_mm"] = tg.height_m * 1000
    return pd.DataFrame(d)


def write_solution(out_dir: Path, name: str, sol: Solution, tg: Targets, cad, summary: dict | None = None, extra: dict | None = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = solution_frame(sol, tg, cad)
    df.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.5f")
    meta = dict(solver=sol.solver, hand=tg.hand, placement=tg.placement.as_dict(), flip=tg.flip, frames=int(sol.n), summary=summary,
                info={k: v for k, v in sol.info.items() if isinstance(v, (int, float, str, list, dict))})
    if extra:
        meta.update(extra)
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))
    np.savez_compressed(out_dir / f"{name}.npz", q=sol.q, jaw=sol.jaw, t=sol.t, phase=sol.phase, hold_id=sol.hold_id,
                        roll_shift=np.zeros(sol.n) if tg.roll_shift is None else tg.roll_shift)
    return df


def read_solution_joints(path: Path) -> dict:
    z = np.load(Path(path).with_suffix(".npz"))
    return {k: z[k] for k in z.files}
