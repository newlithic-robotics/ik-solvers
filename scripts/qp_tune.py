"""Try QP step settings on one hand and print accuracy vs smoothness."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newlithic.cad import load_rig
from newlithic.demo import Placement, load_demo
from newlithic.ik_qp import QPConfig, solve_qp
from newlithic.ik_wls import free_roll_shift, solve_wls
from newlithic.kin import ArmKinematics
from newlithic.metrics import evaluate_solution, summary
from newlithic.model import ArmConfig, PLATE_THICKNESS, TABLE_Z_BASE, compile_arm
from newlithic.targets import Tolerances, make_targets
hand, pl = sys.argv[1], [float(v) for v in sys.argv[2].split(",")]
demo, cad = load_demo(), load_rig()
model, _, _, _ = compile_arm(ArmConfig(), save=False); kin = ArmKinematics(model, cad=cad)
placement = Placement(pl[0] / 1000, pl[1] / 1000, pl[2], TABLE_Z_BASE + PLATE_THICKNESS)
tg = make_targets(demo, hand, placement, cad, Tolerances())
tgf = make_targets(demo, hand, placement, cad, Tolerances(), free_roll=True)
tg.roll_shift = free_roll_shift(kin, tgf, solve_wls(kin, tgf))
print("%-34s %8s %8s %8s %8s %8s %8s" % ("config", "pos_med", "pos_p95", "tilt", "speed95", "smooth", "time"))
for label, kw in [("it1 gain1", dict(iterations=1, gain=1.0)), ("it3 gain1 (default)", dict(iterations=3, gain=1.0)), ("it3 gain0.5", dict(iterations=3, gain=0.5)),
                  ("it5 gain0.35", dict(iterations=5, gain=0.35)), ("it3 gain1 lm1e-2", dict(iterations=3, gain=1.0, lm_damping=1e-2)),
                  ("it3 gain1 vel90", dict(iterations=3, gain=1.0, vel_limit_rad_s={n: np.radians(90.0) for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")})),
                  ("it1 gain1 damp1e-3", dict(iterations=1, gain=1.0, damping=1e-3))]:
    t0 = time.time()
    sol = solve_qp(model, kin, tg, QPConfig(**kw))
    el = time.time() - t0
    sol = evaluate_solution(kin, tg, sol.q, "qp")
    s = summary(sol)
    task = [p for k, p in s["phases"].items() if k != "transport"]
    print("%-34s %8.2f %8.2f %8.1f %8.0f %8.3f %8.0f" % (label, s["task_pos_mm_median"], max(p["pos_mm_p95"] for p in task), s["task_tilt_deg_median"],
                                                      max(v["p95"] for v in s["joint_speed_deg_s"].values()), s["smoothness_deg"], el))
