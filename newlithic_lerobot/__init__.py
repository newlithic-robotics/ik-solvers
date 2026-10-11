"""UMI-style demonstrations of two SO-ARM101s with the Newlithic gripper -> LeRobot training datasets.

Two stages, because they need different environments:

  solve    state-estimation export (grip poses, pad gap, phases per hand)  ->  retargeted episode (joint angles per arm)
           needs the `newlithic` controller package (MuJoCo + mink): bimanual QP, then the trajectory optimiser
  export   retargeted episode(s) + wrist videos  ->  LeRobotDataset v3.0 for `bi_so_follower`
           needs `lerobot`

The retargeted episode (`episode.py`) is the contract between the two.
"""
__version__ = "0.1.0"
