# Findings: IK for two SO-ARM101s with the Newlithic gripper, unscrewing an M5 nut (Episode-1)

Date: 2026-10-11.  Everything below was produced by the code in this repository on the exported Episode-1 demonstration
(1475 frames per hand, 49.2 s, left hand holds the screw by its head, right hand ratchets the nut off in 3-4 re-grips per
screw, two screws).  Numbers are model values (MuJoCo model of the arm + CAD gripper); nothing was measured on hardware.

## 1. The arm with this gripper

| quantity | value | source |
|---|---|---|
| grip point (pad-strip midpoint) from the wrist-roll flange | 168.1 mm along -z of `gripper_link` (stock jaws: 98 mm) | CAD YAML, verified on the meshes |
| X3 camera centre | 86 mm off the roll axis, 57 mm beyond the flange | CAD |
| gripper mass incl. camera and jaw servo | 422 g (printed parts assumed 1.0 g/cm3, X3 180 g, STS3215 55 g) | `assets/masses.yaml` |
| whole arm | 955 g | URDF + CAD |
| pad gap law | 3.3 mm per degree of jaw angle; 22.4 mm (jig) at 6.69 deg, 9.5 mm at 2.8 deg, 144 mm at the 45.6 deg stop | CAD pivots + strip polar law |
| gravity torque, Episode-1 trajectory (p95 / max) | shoulder_lift 1.6 / 1.8 N m, elbow 1.3 / 1.3, wrist_flex 0.56 | `gravity.py` on the solved joints |
| STS3215 stall torque | 2.94 N m at 12 V, 1.62 N m at 7.4 V | vendor data |

**Consequences.**  The shoulder runs at 55-60 % of the 12 V stall torque for most of the in-air work and would exceed the
7.4 V stall torque: this gripper needs the 12 V follower motors (and a 12 V supply).  The 168 mm tool changes the reachable
workspace so much that the base placement found for the stock gripper is useless (section 3).  The camera sits 86 mm off
the roll axis: every roll of the wrist swings 180 g around, and the camera is what the gripper can hit the arm's own lower
links with (the QP's collision pairs include it).

## 2. What a 5-DoF arm cannot do, measured

With the grip point held still, the SO-ARM101 (pan + three parallel pitch axes + roll) cannot lean the fingers out of the
vertical plane through its pan axis.  `kin.rotation_mobility` computes the lost axis per pose from the Jacobian; every
solver projects the orientation error onto what is reachable and reports the lost part separately.  On Episode-1, with the
best left placement, the lost component is 9.6 deg median during the in-air holds and 2-4 deg at the table grasps and
releases.  No solver changes this; only the base placement does (and the demonstrator's heading, which a learned policy
would see through the camera).

## 3. Base placement

Searched per hand with `scripts/place.py` (grid of base positions and headings, each candidate scored by the per-frame
solver on 58 key frames: table grasps x3, in-air holds, releases, the docked rest at low weight):

| arm | base in the demo world (mm, deg) | key-frame position error | tilt median | limit margin |
|---|---|---|---|---|
| left (old, stock gripper) | (-34, -107, 101) | up to 69 mm: the docked rest point is ~120 mm from the pan axis with the fingers forward-down, the 168 mm tool would put the flange behind the pan axis | - | 0 |
| **left (new)** | **(-40.3, -171.5, 37.5)** | 0.1 mm max | 10.4 deg | 0 (the dock region) |
| **right (new)** | **(336.5, 525.8, 225.0)** | 0.0 mm max | 14.4 deg | 34 deg |

The two bases are 793 mm apart.  During the in-air unscrewing the left hand's approach is 9.5 deg (median, p95 23) off the
line from its base, the right hand's 10.6 deg (p95 19): the "head-on" rule (both bases on the screw-axis line) holds to
about 10 deg, which is what the solvers then cannot remove.  A joint two-base search on the relative-pose metric would
reduce it; not done.

## 4. Solver comparison (left hand, new placement, same targets and tolerances for all)

Targets: grip position at 0.5 mm, orientation in the demonstrated triad with per-phase tolerances (tilt 3 deg at the grasp,
6 in holds, 25 in transport; roll 8 / 10 / 60), the roll at each grasp of the round screw head left free (one free-roll
pass shared by all solvers).

| solver | what it is | task position median | hold tilt / roll | grasp tilt / roll | speed p95 (deg/s) | accel p95 (deg/s2) | smoothness (deg/frame) | structural collision frames | time |
|---|---|---|---|---|---|---|---|---|---|
| DLS | damped least squares on the 6-D twist, isotropic, no limits (lerobot-style) | 46 mm | - | - | 104 | - | 0.23 | 447 | 6 s |
| WLS | per-frame bounded least squares, lost axis projected, continuity term | 0.00 mm | 9.6 / 0.5 deg | 10.2 / 0.8 | 100 | 435 | 0.146 | 113 | 11 s |
| QP (mink) | velocity-level QP, joint + velocity limits, collision avoidance | 0.09 mm | 9.6 / 0.2 | 10.2 / 9.9 | 120 (cap) | 716 | 0.416 | 52 | 2 s |
| traj | whole-trajectory least squares with speed / acceleration residuals | 0.04 mm | 9.7 / 1.0 | 10.2 / 0.8 | 89 | 271 | 0.093 | 115 | 84 s |
| bimanual QP | both arms in one QP, right gripper relative to the left in the air, arm-arm collisions | left 0.08 mm (hold) | 9.9 / 2.2 | 10.6 | 120 (cap) | 814 | 0.31 | 32 | 15 s |

"Smoothness" is the mean second difference of the joint angles per frame; "structural collisions" are frames with a
convex-hull penetration > 0.5 mm between arm links, or arm vs table/plate, after ignoring the designed overlaps and the
pads touching the object they are grasping (counted separately).  The WLS/traj collisions are the pad tips dipping up to
2.5 mm into the plate hull while the demonstrator's fingertips touched the plate (not an error of the solver), plus a 4 mm
shoulder-vs-forearm fold at 28 s and, before the docking hold was dropped from the task, a 11 mm gripper-vs-shoulder fold
on the way back to the jig at 40.5 s.

What the numbers say:

* **Plain DLS is unusable** for this arm and task: without the lost-axis projection it trades position for an orientation it
  cannot reach, and without limits it runs the wrist into its stops (median 46 mm position error).  This is the family
  lerobot's own kinematics helper belongs to.
* **All three task-aware solvers reach the same orientation residual** (the lost component) and sub-0.1 mm position: the
  orientation quality is decided by placement and the arm's structure, not by the solver.
* **The whole-trajectory optimiser is the smoothest** (0.093 deg/frame, acceleration p95 271 deg/s2 vs 435 for the per-frame
  solve) at the cost of 0.04 mm and 84 s of compute: it is the one to use for a recorded dataset / policy training, where
  the trajectory is known in advance and servo-friendliness matters (the real arm's vibration at the pickup in the earlier
  work was a setpoint problem).
* **The QP is the one for execution and for anything with constraints**: 2 s for the episode, velocity limits and
  collisions as hard constraints (the only solver that removed the shoulder-fold contacts), one QP per 30 Hz frame, and it
  extends to both arms at once.  Its weakness is accuracy at fast transitions when velocity-capped (grasp position 0.7 mm,
  roll 10 deg median at the grasp because the free-roll shift is reached late); use it with the practical 120 deg/s cap, 3
  sub-steps per frame (the step budget shared so a frame never moves more than the cap).
* **For the bimanual task, solve both arms in one QP with the right gripper relative to the left** while the screw is in the
  air (685 of 1475 frames).  The relative pose is then kept to 0.08 mm / 7.9 deg median, whatever the left arm's own
  orientation residual is; the right arm's absolute errors (2.9 mm in holds) are the correct consequence of following the
  screw the left arm actually holds, not an error.  Independent per-arm solves would each follow the demonstrated absolute
  poses and the nut would miss the screw axis by the sum of the two lost components.

**Recommendation.**  Offline (dataset generation): place the bases with `place.py`, solve with the bimanual QP for the
coupling and the collisions, then smooth with the trajectory optimiser initialised from it (not yet wired as one command).
Online (teleoperation or a policy's action post-processing): the bimanual QP as is.  Never a DLS/pseudo-inverse without
the lost-axis projection on this arm.

## 5. Replay in simulation

`scripts/replay.py --mode kinematic` poses the arms at the solved joints (what the solver produced); `--mode physics`
drives torque-limited position servos (kp 70 N m/rad, STS3215 12 V stall limit) with gravity feed-forward computed from
the model, the bolts and nuts as free bodies (the nut on a hinge + slide coupled at the 0.8 mm thread pitch).

* Servo tracking over the whole bimanual episode: 0.1-0.55 deg mean per joint, 2.4 deg worst, with gravity feed-forward.
  Without it the 168 mm tool droops 8 mm at the first grasp (kp 30) and the pads land below the screw head.
* Actuator load (p95): shoulder_lift 1.6-1.8 N m, elbow 1.4-1.5, wrist_flex 0.7.
* The pure-contact grasp of the M5 head with rigid convex pads holds the screw through most of the whip-up and loses it when
  the head slides to the edge of the pad face (`--grasp contact`, see `out/replay/left_pick1_*` diagnostics).  The replay
  therefore emulates the compliant pads with a weld that engages when the measured pad gap has closed onto the object during
  a demonstrated hold and releases when the demonstration leaves the hold (`--grasp weld`, default; the nut welds are soft so
  the two grippers' relative-pose residual is absorbed there, not at the bolt's pads).  With that the episode plays end to
  end: the left picks screw 1 at 7.3 s and holds it to 21.5 s while the right grasps and releases the nut at the four
  ratchet strokes (9.8, 12.5, 14.8, 18.2 s); screw 2 is held 24.8-38.2 s with three of its four strokes caught (the 0.4 s
  stroke at 30.1 s is missed by the capture test).  Both bolts end on the table where the demonstrator put them down.

Videos: `out/replay/bimanual_kinematic.mp4` (two free views + both X3 finger-side cameras, the solver's joints),
`out/replay/bimanual_physics_v3.mp4` (same views, physics), `out/replay/left_pick1_weld.mp4` (close-up of the first pick).

## 6. Conventions that bit, and must be kept

* The demonstration's opening axis is the handheld CAD's +x: from the R jaw toward the L jaw.  The arm rig's document frame
  has L on -x, so the arm's opening axis is pivot_L - pivot_R.  With the sign the other way the camera side flips, every
  solve pins the wrist roll at a stop and nothing fits.
* The grip origin is the midpoint of the pad contact-strip centres, which moves ~0.3 mm along the fingers over the task's
  gap range: the QP's site target is shifted per frame for it.
* The reference surface in the demonstration is the plate top; the bare table is 10 mm lower.  The real arm's table height
  in the base frame (-17.55 mm, measured on the stock arm in September) differs from the CAD base bottom (-2.4 mm) by 15 mm:
  re-measure on the new arm before replaying on hardware.
* The left hand's closing at 42 s is the gripper being docked, not a grasp (`demo.NOT_A_GRASP`).

## 7. Not done / open

* A joint two-base placement search on the relative-pose metric (would cut the ~10 deg head-on error).
* The docked start and end: reachable in position at the new placement but 33 deg off in orientation (the jig faces the
  demonstrator); a learned policy needs a consistent start pose, which should be designed with the jig placed for the arm.
* Trajectory optimiser initialised from the bimanual QP (coupling + smoothness in one result); torque-aware posture cost is
  implemented (`TrajConfig.torque_sigma_Nm`) but not evaluated.
* Hardware: servo gains, the gravity feed-forward (or lerobot's position-mode droop) and the jaw squeeze were tuned in the
  model only; the real pad compliance decides whether the 9.4 mm demonstrated gap grips an 8.5 mm head.
* The physics nut turning uses the emulated grasp; the thread coupling is kinematic (pitch), no thread friction.
