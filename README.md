# ik-solvers

Everything needed to turn a UMI-style human demonstration (two Insta360 X3 hand rigs) into a **LeRobot training dataset for
two SO-ARM101 arms carrying the Newlithic gripper**, in one repository:

| package | what it is |
|---|---|
| `newlithic/` | the arm with the Newlithic gripper as a MuJoCo model (lerobot's SO-ARM101 URDF with the stock gripper replaced by the CAD), the IK solvers (per-frame least squares, differential QP, whole-trajectory optimiser, bimanual QP), base-placement search, collision and gravity checks, the cell and its replay |
| `newlithic_lerobot/` | the pipeline: state-estimation results → demonstration → wrist videos → joint trajectories → LeRobotDataset v3.0 |
| `assets/` | the gripper CAD (meshes, measurements, masses), the SO-101 meshes and URDF, the jig and plate, and the Episode-1 demonstration |
| `scripts/` | the controller's own entry points: `place.py` (base placement), `compare.py` (solver comparison), `replay.py` (kinematic / physics video), `layout.py` (cell drawing and 1:1 templates), `build_model.py`, `build_scene.py` |
| `docs/findings.md` | what was learned on Episode-1: the arm's limits, placements, solver comparison, physics replay, open items |

The only external piece is the state-estimation pipeline (`newlithic-robotics/state-estimation`, branch `world-pose`), which
produces the per-frame poses and jaw angles the first stage reads; it is installed as a dependency.

## Install

One Python 3.12 environment runs every stage (lerobot ≥ 0.6 needs 3.12; the solvers and the state-estimation package work
on 3.10 too):

```bash
git clone https://github.com/newlithic-robotics/ik-solvers.git && cd ik-solvers
uv venv .venv --python 3.12
uv pip install -e ".[all]"          # or a subset: .[demo] .[solve] .[export]
```

`.[demo]` pulls the state-estimation package from its private GitHub repository, so git needs credentials for the
`newlithic-robotics` organisation (`gh auth login`, then `gh auth setup-git`; a checkout can also be installed by path:
`uv pip install /path/to/state-estimation`). `.[export]` pulls `lerobot[dataset]`, which brings torch. Use the repository
as an editable install: the solvers find `assets/` next to the packages.

On Windows, lerobot's optional `torchcodec` decoder may fail to load and print a long traceback before lerobot falls back
to PyAV; the dataset is unaffected (`pip uninstall torchcodec` silences it).

## The pipeline

```
state-estimation results ──demo-export──▶ <episode>.npz/.json ──solve──▶ <episode>_retargeted.npz/.json ──export──▶ LeRobotDataset
raw .insv files          ──render-wrist─▶ <episode>_wrist_{left,right}.mp4 ─────────────────────────────────┘
```

One command does all of it for an episode:

```bash
newlithic-lerobot pipeline \
    --results /path/to/state-estimation/results --episode Episode-1 --data /path/to/raw/Episode-1 \
    --left=-40.3,-171.5,37.5 --right=336.5,525.8,225.0 \
    --work out/Episode-1 --dataset datasets/so101_unscrew --repo-id newlithic/so101_unscrew
```

or stage by stage:

```bash
newlithic-lerobot demo-export  --results R --episode Episode-1 --out out/Episode-1/
newlithic-lerobot render-wrist --results R --episode Episode-1 --data D --out out/Episode-1/
newlithic-lerobot solve        --demo out/Episode-1/Episode-1.npz --left=-40.3,-171.5,37.5 --right=336.5,525.8,225.0 --out out/Episode-1/
newlithic-lerobot export       out/Episode-1/Episode-1_retargeted.npz --out datasets/so101_unscrew --repo-id newlithic/so101_unscrew \
    --camera left_wrist=out/Episode-1/Episode-1_wrist_left.mp4 --camera right_wrist=out/Episode-1/Episode-1_wrist_right.mp4 --verify
```

### demo-export: state-estimation results → demonstration

Reads, for one episode, the `pose` and `bimanual` outputs of both clips and the `aperture` output (factory lens model):
`pose/<episode>/<side>/<stem>_pose.csv|_pose_summary.json|_map.npz`, `pose/<episode>/bimanual_summary.json`,
`aperture*/<episode>/<side>/<stem>_aperture.csv|_summary.json`, with `side` = `LHS` (left hand) and `RHS`. Writes, per hand on
one clock (the left rig's pose instants) in one world (the left rig's, gravity aligned, z up): the grip frame (origin =
midpoint of the pad contact centres; approach along the fingers, opening from the R jaw toward the L jaw, normal toward the
camera), the pad gap, the phase (transport / approach / grasp / hold / release, from the gap signal), the hold number, and the
heights over the reference surface (the dominant plane of the map the fingers touched: the plate top in Episode-1).

Two calibrations happen here and are recorded in the file's notes: each hand's aperture is offset so that the docked rest
window reads `--jig-gap-mm` (22.4 mm, the only aperture reference a recording contains), and a short closing of both grippers
at the same moment is relabelled transport (the operator's synchronisation clap).

### render-wrist: raw Insta360 files → wrist-camera videos

Lens 1 of each rig looks along the fingers through a fisheye; the video is a virtual pinhole camera fixed to it (640 × 880,
100° horizontal, the TCP at 70 % of the height, jaws at the bottom): what the X3 on the arm's wrist will see. Both videos are
on the left rig's clock, one frame per demonstration frame, so the export can match frames by index. Needs the raw `.insv`
files (searched recursively under `--data`) and ffmpeg (imageio-ffmpeg's binary is used).

### solve: demonstration → retargeted episode

The offline recipe of the findings, wired as one call:

1. **bimanual QP** (`newlithic.ik_bimanual`): both arms in one QP per frame in the cell model. While the left hand holds the
   screw in the air the right gripper is asked for its pose *relative to the left*. Joint and velocity limits, arm-arm and
   arm-table collisions are hard constraints; the wrist roll at round-object grasps is free.
2. **trajectory optimiser** (`newlithic.ik_traj`), per arm, initialised from the QP result, with the QP's own reached poses as
   targets so the arm-arm coupling survives; the per-phase orientation tolerances come from the demonstration's phases.
3. **evaluation** against the demonstrated targets, gravity torque, collision report.

Placements are `x_mm,y_mm,yaw_deg` of each base in the demonstration's world, as `scripts/place.py` prints them; the values
above are the findings' for Episode-1. `--frames a:b` solves a slice (the CI smoke test); `--no-smooth` keeps the raw QP
result. Episode-1 takes about three minutes.

The retargeted episode (`newlithic_lerobot/episode.py`) holds per hand the joint angles (degrees, lerobot's calibrated zero =
URDF zero), the jaw angle, the pad gap, phases, the demonstrated and the reached grip pose in that arm's base frame, the
residuals, and the placements, solver configuration and provenance.

### export: retargeted episode(s) + videos → LeRobotDataset v3.0

Per output frame, uniform at `--fps` (default 30; the demonstration's 29.97 fps is resampled):

| feature | shape | content |
|---|---|---|
| `observation.state` | (12,) float32 | the solved joints at that instant, `left_shoulder_pan.pos … left_gripper.pos, right_shoulder_pan.pos … right_gripper.pos` (lerobot's `bi_so_follower` names) |
| `action` | (12,) float32 | the state `--action-lead` frames later (default 1): the setpoint the arms move toward, the relation `lerobot-record` stores between leader and follower |
| `observation.images.<camera>` | (H, W, 3) video | the camera frame nearest to that instant; `--camera name=path`, paths may contain `{episode}` / `{stem}` |
| `observation.grip_pose` | (14,) float32 | demonstrated grip pose per hand in its arm's base frame, `x y z qw qx qy qz` (`--no-extras` drops the three extras) |
| `observation.phase` | (2,) float32 | 0 transport, 1 grasp, 2 hold, 3 release, 4 approach |
| `observation.gap_mm` | (2,) float32 | the pad gap the demonstration measured |

Several episodes: pass every `.npz` to one `export`, or append later with `--resume`. `meta/newlithic_export.json` next to
lerobot's metadata records placements, solver configuration, residual summaries and provenance per episode. `--verify`
reloads the dataset with `LeRobotDataset` and decodes a frame per camera.

**Units** are a deployment decision and must match the robot config:

* `--joint-units degrees` (default): the URDF / lerobot angle, zero at the middle of the calibrated travel, i.e.
  `SOFollowerConfig(use_degrees=True)`.
* `--joint-units range_m100_100 --calibration-dir <dir>`: lerobot's default normalisation, each servo's calibrated travel
  mapped to [-100, 100] from that arm's calibration file (`left.json`, `right.json`, lerobot's format).
* The gripper is always a percentage: 0 % = pads touching (jaw angle 0), 100 % = the jaw stop (45.6°, 144 mm gap), linear in
  the jaw angle (`--gripper-open-percent 0` inverts it). Calibrate the real jaw servo with those two ends, or re-map the
  column. The pad gap is 3.3 mm per degree of jaw angle (CAD law), so the demonstrated gap maps 1:1 onto it.

**Time warp** (`--v-max-deg-s` on export, off by default) stretches the clock where a joint would exceed the cap, one warp
for both arms so their relative timing is kept; camera frames repeat in the stretched parts. The trajectory optimiser already
keeps Episode-1 within the servos' capability at 1:1 time.

## Episode-1 numbers

`solve` with the findings' placements (bimanual QP 14 s, optimiser 74 s and 62 s):

| arm | task position error (median) | tilt | speed p95 | accel p95 | smoothness |
|---|---|---|---|---|---|
| left | 0.10 mm | 9.7° | 82 °/s | 202 °/s² | 0.077 °/frame |
| right | 3.32 mm (follows the screw the left holds; the pair is kept relative) | 11.9° | 89 °/s | 237 °/s² | 0.076 °/frame |

The findings' bimanual QP alone: 0.08 mm, 9.9°, accel p95 814 °/s², smoothness 0.31 °/frame. The exported dataset: 1476
frames at 30 fps, two 640 × 880 AV1 videos, 31 MB.

## Conventions and open items (from the findings)

* Joint zero is lerobot's calibrated zero (the URDF zero pose); the grip frame is 168 mm from the wrist flange with this gripper.
* The demonstration's opening axis is from the R jaw toward the L jaw; the sign the other way flips the camera side and nothing fits.
* The reference surface is the plate top; the arm's own table height (`TABLE_Z_BASE`) was measured on the stock arm and must
  be re-measured on the new one before anything is replayed on hardware.
* No robot was involved: `observation.state` is the kinematic solution. The physics replay tracks it to 0.1–0.55° mean per
  joint with gravity feed-forward; the real arm with this gripper has not been built.
* The right wrist camera sees the human's hand holding the left rig, as any UMI recording does.
* The ~10° orientation residual is the 5-DoF arm plus base placement, not the solver; the docked start and end poses are 33°
  off at these placements and a policy needs a designed start pose.

## Development

```bash
uv pip install -e ".[solve,test]"
pytest                                                        # numpy + scipy only: formats, units, clock, frames, phases
python -m newlithic_lerobot solve --demo assets/demo/episode1.npz --left=-40.3,-171.5,37.5 --right=336.5,525.8,225.0 --frames 200:240 --max-nfev 5 --out out/smoke.npz
```

CI runs the unit tests on Python 3.10 and 3.12 and the 40-frame solve smoke test on 3.12. `tools/step_to_meshes.py`
(STEP → named STLs, needs cq-editor's OCP) regenerates the CAD meshes when the gripper changes; `scripts/build_model.py`
rebuilds the MJCF.
