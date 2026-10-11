import numpy as np

from newlithic_lerobot.timing import interp, interp_pose7, make_clock, nearest_index, time_warp


def test_uniform_clock_without_warp():
    t = np.arange(100) / 29.97
    c = make_clock(t, 30)
    assert c.t_out[0] == 0.0 and np.allclose(np.diff(c.t_out), 1 / 30)
    assert c.t_out[-1] <= t[-1] - t[0] + 1e-9
    assert c.m == 100                                  # 99 / 29.97 s holds 100 frames at 30 fps (the last at 3.300 s <= 3.303 s)
    assert np.all(np.diff(c.src_index) >= 0) and c.src_index[0] == 0 and c.src_index[-1] == 99


def test_nearest_index():
    t = np.array([0.0, 1.0, 2.0])
    np.testing.assert_array_equal(nearest_index(t, np.array([-1, 0.4, 0.6, 1.5, 1.6, 9])), [0, 0, 1, 1, 2, 2])   # a tie goes to the earlier frame


def test_warp_caps_speed_and_keeps_slow_parts():
    t = np.arange(200) / 30
    Q = np.zeros((200, 5))
    Q[80:120, 2] = np.linspace(0, 80, 40)              # one joint does 80 deg in 40 frames: 60 deg/s
    Q[120:, 2] = 80.0
    te = time_warp(t, [Q], v_max_deg_s=30.0)
    assert te[0] == 0.0 and np.all(np.diff(te) >= np.diff(t) - 1e-12)
    v = np.abs(np.diff(Q, axis=0)).max(1) / np.diff(te)
    assert v.max() <= 30.0 + 1e-9
    assert np.allclose(np.diff(te)[:60], np.diff(t)[:60])        # far from the fast part nothing is stretched
    te2 = time_warp(t, [Q], v_max_deg_s=1000.0)
    np.testing.assert_allclose(te2, t)


def test_warped_clock_repeats_source_frames():
    t = np.arange(60) / 30
    Q = np.zeros((60, 5))
    Q[20:40, 0] = np.linspace(0, 100, 20)
    c = make_clock(t, 30, Qs=[Q], v_max_deg_s=30.0)
    assert c.m > 60
    assert np.all(np.diff(c.src_index) >= 0) and len(np.unique(c.src_index)) == 60


def test_interp_shapes():
    t = np.array([0.0, 1.0])
    X = np.array([[0.0, 10.0], [2.0, 20.0]])
    np.testing.assert_allclose(interp(t, X, np.array([0.5])), [[1.0, 15.0]])
    np.testing.assert_allclose(interp(t, X[:, 0], np.array([0.25])), [0.5])
    P = np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]], float)
    out = interp_pose7(t, P, np.array([0.5]))
    np.testing.assert_allclose(np.linalg.norm(out[:, 3:], axis=1), 1.0)
