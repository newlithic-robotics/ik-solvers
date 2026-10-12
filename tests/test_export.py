import numpy as np
import pytest

from conftest import FakeDataset, FakeVideo
from newlithic_lerobot.export import ExportConfig, build_frames, camera_paths, write_episode
from newlithic_lerobot.features import Units


def test_build_frames_state_and_action(episode):
    f = build_frames(episode, ExportConfig(fps=30, action_lead=1))
    assert f.state.shape == (f.m, 12) and f.action.shape == f.state.shape
    assert abs(f.clock.t_out[-1] - episode.duration_s) < 1 / 30
    np.testing.assert_allclose(f.action[:-1], f.state[1:])
    np.testing.assert_allclose(f.action[-1], f.state[-1])
    # the first output frame is the first demonstration frame
    np.testing.assert_allclose(f.state[0, :5], episode.hands["left"].q_deg[0])
    np.testing.assert_allclose(f.state[0, 5], 100 * episode.hands["left"].jaw_deg[0] / 45.6)
    assert set(f.extras) == {"observation.grip_pose", "observation.phase", "observation.gap_mm"}
    assert f.extras["observation.grip_pose"].shape == (f.m, 14) and f.extras["observation.phase"].shape == (f.m, 2)
    assert f.names[6] == "right_shoulder_pan.pos"


def test_action_lead_zero_and_no_extras(episode):
    f = build_frames(episode, ExportConfig(action_lead=0, extras=False))
    np.testing.assert_allclose(f.action, f.state)
    assert f.extras == {}


def test_window_trims_and_keeps_video_indices(episode):
    f = build_frames(episode, ExportConfig(start_s=2.0, end_s=5.0))
    assert f.clock.src_index[0] >= 59 and f.clock.src_index[-1] <= 151
    assert abs(f.clock.t_out[-1] - 3.0) < 2 / 30


def test_warp_in_export(episode):
    plain = build_frames(episode, ExportConfig())
    warped = build_frames(episode, ExportConfig(v_max_deg_s=5.0))
    assert warped.m > plain.m
    assert len(np.unique(warped.clock.src_index)) == episode.n


def test_write_episode_uses_nearest_video_frame(episode):
    cfg = ExportConfig(fps=30, action_lead=2)
    f = build_frames(episode, cfg)
    ds, vid = FakeDataset(), {"left_wrist": FakeVideo(episode.n), "right_wrist": FakeVideo(episode.n)}
    n = write_episode(ds, f, vid, "do the thing")
    assert n == f.m and ds.episodes == [f.m]
    fr = ds.frames[0]
    assert fr["task"] == "do the thing"
    assert fr["observation.state"].dtype == np.float32 and fr["action"].shape == (12,)
    assert fr["observation.images.left_wrist"].shape == (8, 6, 3) and fr["observation.images.left_wrist"].dtype == np.uint8
    assert vid["left_wrist"].reads == list(f.clock.src_index)
    assert fr["observation.grip_pose"].dtype == np.float32
    assert all(k in {"observation.state", "action", "task", "observation.grip_pose", "observation.phase", "observation.gap_mm",
                     "observation.images.left_wrist", "observation.images.right_wrist"} for k in fr)


def test_camera_paths_patterns(episode, tmp_path):
    p = camera_paths({"left_wrist": "v/{episode}_L.mp4", "right_wrist": "v/{stem}_R.mp4"}, episode, tmp_path / "Episode-T_retargeted.npz")
    assert p["left_wrist"].name == "Episode-T_L.mp4" and p["right_wrist"].name == "Episode-T_retargeted_R.mp4"


def test_config_validation():
    with pytest.raises(ValueError):
        ExportConfig(fps=29.97)
    with pytest.raises(ValueError):
        ExportConfig(vcodec="vp9")
    with pytest.raises(ValueError):
        ExportConfig(action_lead=-1)
    ExportConfig(units=Units(gripper_open_percent=0.0))
