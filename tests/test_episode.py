import numpy as np
import pytest

from newlithic_lerobot.episode import HandTrajectory, RetargetedEpisode, pose7


def test_round_trip(episode, tmp_path):
    p = episode.save(tmp_path / "ep")
    assert p.exists() and p.with_suffix(".json").exists()
    back = RetargetedEpisode.load(p)
    assert back.episode == episode.episode and back.n == episode.n and list(back.hands) == ["left", "right"]
    np.testing.assert_allclose(back.t, episode.t)
    for h in episode.hands:
        np.testing.assert_allclose(back.hands[h].q_deg, episode.hands[h].q_deg)
        assert back.hands[h].phase.dtype == np.int64
    assert back.placements["right"]["yaw_deg"] == 225.0
    assert back.task == "test task"
    assert "frames" in back.describe()


def test_validation_catches_shape_mismatch(episode):
    bad = episode.hands["left"]
    bad.q_deg = bad.q_deg[:-1]
    with pytest.raises(ValueError):
        episode.validate()


def test_validation_rejects_non_monotone_time(episode):
    episode.t[10] = episode.t[9]
    with pytest.raises(ValueError):
        episode.validate()


def test_load_rejects_foreign_json(tmp_path):
    (tmp_path / "x.json").write_text('{"format": "other"}')
    (tmp_path / "x.npz").write_bytes(b"")
    with pytest.raises(ValueError):
        RetargetedEpisode.load(tmp_path / "x.npz")


def test_pose7_is_continuous():
    from scipy.spatial.transform import Rotation as Rot
    ang = np.linspace(0, 2 * np.pi, 50)                    # a full turn crosses the quaternion sign flip
    R = Rot.from_euler("z", ang[:, None]).as_matrix()
    P = pose7(R, np.zeros((50, 3)))
    assert P.shape == (50, 7)
    assert np.all(np.sum(P[1:, 3:] * P[:-1, 3:], axis=1) > 0)
    np.testing.assert_allclose(np.linalg.norm(P[:, 3:], axis=1), 1.0)
