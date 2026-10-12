"""UMI-style demonstrations of two SO-ARM101s with the Newlithic gripper -> LeRobot training datasets.

    demo-export   state-estimation results (poses, jaw angles, maps)  ->  demonstration (grip poses, pad gap, phases per hand)
    render-wrist  state-estimation results + raw Insta360 files     ->  wrist-camera videos on the demonstration's clock
    solve         demonstration + base placements                   ->  retargeted episode (joint angles per arm)
                  bimanual QP, then the trajectory optimiser (the `newlithic` package of this repository)
    export        retargeted episode(s) + wrist videos              ->  LeRobotDataset v3.0 for `bi_so_follower`

The retargeted episode (`episode.py`) is the contract between the solvers and the dataset writer.
"""
__version__ = "0.2.0"
