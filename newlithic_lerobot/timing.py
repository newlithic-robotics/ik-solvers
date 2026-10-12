"""From the demonstration's clock to the dataset's clock.

LeRobot wants uniform frames at an integer fps; the demonstration runs at 29.97 fps on the left rig's pose instants.  The
joints are interpolated onto the dataset clock and every output frame remembers the nearest demonstration frame, which is
the camera frame it gets.

Optionally the clock is also stretched (`time_warp`) where the person moved faster than the servos can follow: every interval
gets the time its fastest joint needs under a speed cap, the slow-downs start a few frames early so the acceleration stays
sane, and the stretch is smoothed.  One warp serves both arms, so their relative timing - the point of a bimanual recording -
is kept exactly.  Where the clock is stretched the camera frames repeat; the policy then sees a slow-motion demonstration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def time_warp(t: np.ndarray, Qs, v_max_deg_s: float, lead_frames: int = 6, smooth_frames: int = 9) -> np.ndarray:
    """Qs: one or more (n,k) joint paths [deg] on the clock t.  Returns the stretched clock (n,), starting at 0."""
    t = np.asarray(t, float)
    dt = np.diff(t)
    need = np.ones(len(dt))
    for Q in Qs:
        v = np.abs(np.diff(np.asarray(Q, float), axis=0)).max(1) / dt
        need = np.maximum(need, v / v_max_deg_s)
    k = max(1, int(lead_frames))
    pad = np.pad(need, (k, k), mode="edge")
    need = np.array([pad[i:i + 2 * k + 1].max() for i in range(len(need))])
    if smooth_frames >= 3:
        m = int(smooth_frames) | 1
        pad = np.pad(need, (m // 2, m // 2), mode="edge")
        need = np.convolve(pad, np.ones(m) / m, mode="valid")
    need = np.maximum(need, 1.0)                                   # the smoothing must never speed anything up
    return np.concatenate([[0.0], np.cumsum(dt * need)])


@dataclass
class Clock:
    t_out: np.ndarray          # (m,) dataset timestamps, k / fps
    t_demo: np.ndarray         # (m,) the demonstration instant each output frame samples
    src_index: np.ndarray      # (m,) nearest demonstration frame (the camera frame to use)
    t_exec: np.ndarray         # (n,) the demonstration frames on the dataset clock

    @property
    def m(self) -> int:
        return len(self.t_out)


def make_clock(t: np.ndarray, fps: float, Qs=None, v_max_deg_s: float = 0.0) -> Clock:
    """t: demonstration instants (n,).  With v_max_deg_s > 0 the clock is warped first (Qs required)."""
    t = np.asarray(t, float)
    t = t - t[0]
    t_exec = time_warp(t, Qs, v_max_deg_s) if v_max_deg_s and v_max_deg_s > 0 else t
    t_out = np.arange(0.0, t_exec[-1] + 1e-9, 1.0 / fps)
    t_demo = np.interp(t_out, t_exec, t)
    src = nearest_index(t, t_demo)
    return Clock(t_out=t_out, t_demo=t_demo, src_index=src, t_exec=t_exec)


def nearest_index(t_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    i = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    left_closer = np.abs(t_src[i - 1] - t_query) <= np.abs(t_src[i] - t_query)
    return np.where(left_closer, i - 1, i).astype(np.int64)


def interp(t_src: np.ndarray, X: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation of (n,) or (n,k) values."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        return np.interp(t_dst, t_src, X)
    return np.stack([np.interp(t_dst, t_src, X[:, j]) for j in range(X.shape[1])], axis=1)


def interp_pose7(t_src: np.ndarray, P: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """(n,7) [xyz, qw qx qy qz] with a continuous quaternion sign: linear, then the quaternion renormalised."""
    out = interp(t_src, P, t_dst)
    q = out[:, 3:7]
    out[:, 3:7] = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return out
