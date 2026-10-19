"""The end-effector training data: frames in the left dock frame, 6D rotation, action one frame ahead."""
import numpy as np

from newlithic_lerobot.ee_dataset import dock_frame, ee_frames, features, names


def _demo(path, n=5):
    t = np.arange(n) / 30.0
    arrays = {}
    for h, x0 in (("left", 1.0), ("right", 1.2)):
        R = np.tile(np.eye(3), (n, 1, 1))
        arrays.update({f"{h}_t": t, f"{h}_R": R, f"{h}_p": np.column_stack([x0 + 0.01 * np.arange(n), np.full(n, 2.0), np.full(n, 0.5)]),
                       f"{h}_gap": np.linspace(0.02, 0.01, n), f"{h}_phase": np.zeros(n, int), f"{h}_valid": np.ones(n, bool)})
    np.savez(path, **arrays)
    return path


def test_poses_are_written_in_the_left_dock_frame(tmp_path):
    anchor = {"xy": [1.0, 2.0], "heading": np.pi / 2, "table_z": 0.3}       # docked at (1, 2), fingers along world +y
    fr = ee_frames(_demo(tmp_path / "d.npz"), anchor)
    n = names()
    i, j = n.index("left_x"), n.index("right_x")
    assert np.allclose(fr["state"][0, i:i + 3], [0.0, 0.0, 0.2])               # the left grip starts at the dock, 0.2 m above the table
    assert np.allclose(fr["state"][0, j:j + 3], [0.0, -0.2, 0.2])             # world +x is the dock frame's -y
    assert np.allclose(fr["state"][0, i + 3:i + 6], [0.0, -1.0, 0.0])         # the grip's approach (world x) seen from the dock
    assert np.allclose(fr["action"][:-1], fr["state"][1:]) and np.allclose(fr["action"][-1], fr["state"][-1])
    assert np.isclose(fr["state"][0, i + 9], 0.02)


def test_dock_frame_round_trip():
    Rz, o = dock_frame({"xy": [0.3, -0.1], "heading": 0.7, "table_z": -0.15})
    p = np.array([0.5, 0.2, 0.0])
    assert np.allclose(Rz.T @ (Rz @ (p - o)) + o, p) and np.allclose(Rz @ Rz.T, np.eye(3))


def test_features_name_twenty_numbers_and_two_cameras():
    f = features({"left_wrist": (880, 640), "right_wrist": (880, 640)})
    assert f["observation.state"]["shape"] == (20,) and f["action"]["names"] == names()
    assert f["observation.images.right_wrist"]["shape"] == (880, 640, 3)
