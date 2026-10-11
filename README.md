# ik-solvers

Turns UMI-style human demonstrations (two Insta360 X3 hand rigs, state-estimated grip poses and pad gaps) into **LeRobot
training datasets for two SO-ARM101 arms carrying the Newlithic gripper**, using the IK solvers and the base placements that
the controller work established (`controller - newlithic/docs/findings.md`).

The package here is `newlithic_lerobot`.  It has two stages, because they need different environments:

| stage | input | output | needs |
|---|---|---|---|
| `solve` | the demonstration export (`episode1.npz` + `.json`: grip frame per hand per frame, pad gap, phases) and the two base placements | a **retargeted episode** (`<episode>_retargeted.npz` + `.json`): joint angles per arm, jaw angle, phases, demonstrated and reached grip poses, residuals, provenance | the controller package (`newlithic`: MuJoCo, mink) and its assets |
| `export` | retargeted episode(s) + the wrist-view videos | a **LeRobotDataset v3.0** for `bi_so_follower` | `lerobot >= 0.4.4` |

The retargeted episode is the contract between them (`newlithic_lerobot/episode.py`), so the export never touches MuJoCo
and the solver never touches torch.

## What the solve stage does

Exactly the offline recipe the findings recommend, wired as one command:

1. **bimanual QP** (`newlithic.ik_bimanual`): both arms in one QP per frame in the cell model.  While the left hand holds
   the screw in the air the right gripper is asked for its pose *relative to the left* (the unscrewing is a relation between
   the grippers; the estimator's cross-rig error lives in the absolute poses).  Joint and velocity limits, arm-arm and
   arm-table collisions are hard constraints; the wrist roll at round-object grasps is free.
2. **trajectory optimiser** (`newlithic.ik_traj`), per arm, initialised from the QP result.  Its targets are the grip poses the
   QP solution actually reaches, not the demonstration, so the arm-arm coupling survives; the per-phase orientation
   tolerances still come from the demonstration's phases.  This is the smoothing the findings ask for ("the one to use for a
   recorded dataset / policy training"), which they had not wired yet.
3. **evaluation** against the demonstrated targets (the same numbers the findings tables report), gravity torque, collisions.

```bash
# in the controller venv (MuJoCo + mink); the repo is importable from its root
export NEWLITHIC_CONTROLLER="/path/to/controller - newlithic"
python -m newlithic_lerobot solve \
    --demo "$NEWLITHIC_CONTROLLER/assets/demo/episode1.npz" \
    --left=-40.3,-171.5,37.5 --right=336.5,525.8,225.0 \
    --out out/Episode-1_retargeted.npz
python -m newlithic_lerobot inspect out/Episode-1_retargeted.npz --solver
```

Placements are `x_mm,y_mm,yaw_deg` of each base in the demonstration's world, as `scripts/place.py` prints them.  The ones
above are the findings' values for Episode-1.  `--no-smooth` keeps the raw QP result; `--v-max-deg-s / --a-max-deg-s2 /
--max-nfev` are the optimiser's residual scales and budget.  Episode-1 takes about three minutes.

## What the export stage writes

```bash
# in the lerobot environment
pip install -e .
newlithic-lerobot export out/Episode-1_retargeted.npz \
    --out datasets/so101_unscrew --repo-id newlithic/so101_unscrew \
    --camera left_wrist=/path/to/episode1_wrist_left.mp4 \
    --camera right_wrist=/path/to/episode1_wrist_right.mp4 \
    --verify
```

Per output frame, uniform at `--fps` (default 30; the demonstration's 29.97 fps is resampled):

| feature | shape | content |
|---|---|---|
| `observation.state` | (12,) float32 | the solved joints at that instant, `left_shoulder_pan.pos … left_gripper.pos, right_shoulder_pan.pos … right_gripper.pos` (lerobot's `bi_so_follower` names) |
| `action` | (12,) float32 | the state `--action-lead` frames later (default 1): the setpoint the arms are moving toward, the same relation `lerobot-record` stores between the leader and the follower |
| `observation.images.<camera>` | (H, W, 3) video | the camera frame nearest to that instant; `--camera name=path` per camera, paths may contain `{episode}` / `{stem}` for multi-episode runs |
| `observation.grip_pose` | (14,) float32 | demonstrated grip pose per hand in its arm's base frame, `x y z qw qx qy qz` (optional, `--no-extras` drops the three) |
| `observation.phase` | (2,) float32 | 0 transport, 1 grasp, 2 hold, 3 release, 4 approach |
| `observation.gap_mm` | (2,) float32 | the pad gap the demonstration measured |

Several episodes: pass every `.npz` (in order) to one `export` call, or append later with `--resume`.  `meta/newlithic_export.json`
next to lerobot's own metadata records the placements, solver configuration, residual summaries and provenance of every episode.

**Cameras.**  The videos must be on the demonstration's clock, one video frame per demonstration frame.  For Episode-1 these
are the wrist views rendered by `state-estimation-debugging/scripts/render.py wrist` (640 x 880, a virtual pinhole camera fixed
to each gripper at the centre of its lens 1, the right rig resampled to the left rig's clock): what the X3 on the arm's wrist
will see.

**Units** are a deployment decision and must match the robot config:

* `--joint-units degrees` (default): the URDF / lerobot angle, zero at the middle of the calibrated travel, i.e.
  `SOFollowerConfig(use_degrees=True)`.  The IK produces this directly.
* `--joint-units range_m100_100 --calibration-dir <dir>`: lerobot's default normalisation, each servo's calibrated travel
  mapped to [-100, 100] from that arm's calibration file (`left.json`, `right.json`, the lerobot format).
* The gripper is always a percentage: 0 % = pads touching (jaw angle 0), 100 % = the jaw stop (45.6 deg, 144 mm gap),
  linear in the jaw angle (`--gripper-open-percent 0` inverts it).  Calibrate the real jaw servo with those two ends, or re-map
  this column.  The pad gap is 3.3 mm per degree of jaw angle (CAD law), so the demonstrated gap maps 1:1 onto it.

**Time warp** (`--v-max-deg-s`, off by default): stretches the clock where a joint would exceed the cap, one warp for both arms
so their relative timing is kept.  Camera frames repeat in the stretched parts (a slow-motion demonstration).  The trajectory
optimiser already keeps Episode-1 within the servo's capability at 1:1 time, so this is for faster demonstrations or slower
servos.

## Conventions carried over from the findings

* Joint zero is lerobot's calibrated zero (the URDF zero pose); degrees in the episode file.
* The grip frame is the midpoint of the pad contact-strip centres (168 mm from the wrist flange with this gripper);
  approach along the fingers, opening from the R jaw toward the L jaw, normal toward the camera - the same triad the
  demonstration is expressed in.
* The reference surface is the plate top; the arm's own table height (`TABLE_Z_BASE` in the controller model) was measured on
  the stock arm and must be re-measured on the new one before anything is replayed on hardware.
* The left hand's closing at 42 s of Episode-1 is the docking, not a grasp (the controller relabels it on load).

## Not verified / open

* No robot was involved: `observation.state` is the kinematic solution, not what servos reported.  The physics replay in
  the controller tracks it to 0.1-0.55 deg mean per joint with gravity feed-forward; the real arm with this gripper has not
  been built.
* The right arm's absolute pose error during the in-air holds (about 3 mm) is the correct consequence of following the screw
  the left arm actually holds; the relative pose of the two grippers is what the QP keeps (0.08 mm / 8 deg median).
* The ~10 deg "lost axis" orientation residual is the 5-DoF arm's structure plus the base placement, not the solver; a joint
  two-base search on the relative-pose metric would reduce it.
* The docked start and end poses are 33 deg off in orientation at these placements; a policy needs a designed start pose.

## Development

```bash
uv venv .venv --python 3.12 && uv pip install -e ".[test]"
pytest
```

The tests cover the episode format, feature layout and unit conversions (checked against lerobot's normalisation formula),
the clock (resampling, nearest-frame mapping, time warp) and the frame building / writing with a fake dataset.  They need
only numpy and scipy; the solve stage is exercised by running it on Episode-1.
