"""UMI-style demonstrations of the Newlithic glove gripper -> the end-effector LeRobot training dataset, and an IK check.

    demo-export   state-estimation results (poses, jaw angles, maps)  ->  demonstration (grip poses, pad gap, phases per hand)
    render-wrist  state-estimation results + raw Insta360 files     ->  wrist-camera videos on the demonstration's clock
    solve         demonstration + base placements                   ->  retargeted episode (joint angles per arm): the IK check,
                  bimanual QP, then the trajectory optimiser (the `newlithic` package of this repository); a QA diagnostic
    collection    sessions -> prepared episodes -> export-ee: the end-effector training dataset (ee_dataset.py: grip poses and
                  openings in each episode's dock frame, wrist views; no joints, they are added at inference)
"""
__version__ = "0.2.0"
