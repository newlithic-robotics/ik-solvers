import json

import numpy as np
import pytest

from newlithic_lerobot.features import Units, dataset_features, load_calibration, motor_names, state_matrix


def test_motor_names_follow_bi_so_follower():
    names = motor_names(["left", "right"])
    assert names[0] == "left_shoulder_pan.pos" and names[5] == "left_gripper.pos"
    assert names[6] == "right_shoulder_pan.pos" and names[-1] == "right_gripper.pos"
    assert len(names) == 12


def test_dataset_features_shapes():
    f = dataset_features(["left", "right"], {"left_wrist": (880, 640)}, extras=True)
    assert f["action"]["shape"] == (12,) and f["observation.state"]["names"] == motor_names(["left", "right"])
    assert f["observation.images.left_wrist"] == {"dtype": "video", "shape": (880, 640, 3), "names": ["height", "width", "channels"]}
    assert f["observation.grip_pose"]["shape"] == (14,) and f["observation.phase"]["shape"] == (2,)
    assert "observation.grip_pose" not in dataset_features(["left"], extras=False)


def test_gripper_percent_both_conventions():
    u = Units()
    np.testing.assert_allclose(u.gripper(np.array([0.0, 22.8, 45.6, 60.0]), 45.6), [0, 50, 100, 100])
    inv = Units(gripper_open_percent=0.0)
    np.testing.assert_allclose(inv.gripper(np.array([0.0, 45.6]), 45.6), [100, 0])
    np.testing.assert_allclose(inv.gripper_to_jaw_deg(inv.gripper(np.array([10.0]), 45.6), 45.6), [10.0])


def test_degrees_pass_through(episode):
    S = state_matrix(episode, Units())
    assert S.shape == (episode.n, 12)
    np.testing.assert_allclose(S[:, :5], episode.hands["left"].q_deg)
    np.testing.assert_allclose(S[:, 11], 100 * episode.hands["right"].jaw_deg / 45.6)


def test_range_m100_matches_lerobot_formula(tmp_path):
    cal = {m: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 3000}
           for i, m in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"])}
    p = tmp_path / "left.json"
    p.write_text(json.dumps(cal))
    c = load_calibration(p)
    assert set(c) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"}
    u = Units(joints="range_m100_100", calibration={"left": c})
    deg = np.array([[10.0, -20.0, 0.0, 45.0, 90.0]])
    v = u.arm(deg, "left")
    # lerobot: raw = mid + deg * 4095 / 360 ; m100 = (raw - min) / (max - min) * 200 - 100
    raw = 2000 + deg * 4095 / 360
    np.testing.assert_allclose(v, (raw - 1000) / 2000 * 200 - 100)
    np.testing.assert_allclose(u.arm_to_degrees(v, "left"), deg)


def test_units_validation():
    with pytest.raises(ValueError):
        Units(joints="range_m100_100")
    with pytest.raises(ValueError):
        Units(joints="radians")
    with pytest.raises(ValueError):
        Units(gripper_open_percent=50.0)
