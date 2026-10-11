import numpy as np
import pytest

from newlithic_lerobot.episode import HandTrajectory, RetargetedEpisode


def synthetic_hand(t: np.ndarray, seed: int) -> HandTrajectory:
    rng = np.random.default_rng(seed)
    n = len(t)
    q = 30 * np.sin(np.outer(t, [0.5, 0.7, 0.9, 1.1, 1.3]) + rng.uniform(0, 6, 5))
    jaw = 20 + 15 * np.sin(0.4 * t)
    phase = np.zeros(n, np.int64)
    phase[n // 3: 2 * n // 3] = 2
    hold = np.where(phase == 2, 0, -1).astype(np.int64)
    pose = np.zeros((n, 7))
    pose[:, :3] = 0.1 * np.stack([np.cos(t), np.sin(t), 0.5 + 0 * t], 1)
    pose[:, 3] = 1.0
    z = np.zeros(n)
    return HandTrajectory(q_deg=q, jaw_deg=jaw, gap_mm=3.3 * jaw + 0.18, phase=phase, hold_id=hold, pose_demo=pose, pose_fk=pose,
                          pos_err_mm=z + 0.1, tilt_err_deg=z + 1.0, roll_err_deg=z + 0.5, lost_deg=z + 2.0)


@pytest.fixture
def episode() -> RetargetedEpisode:
    fps = 30000 / 1001
    t = np.arange(300) / fps - 0.009
    return RetargetedEpisode(episode="Episode-T", fps=fps, t=t, hands={"left": synthetic_hand(t, 1), "right": synthetic_hand(t, 2)},
                             placements={"left": dict(x_m=-0.04, y_m=-0.17, yaw_deg=37.5, surface_z_base=-0.00755),
                                         "right": dict(x_m=0.34, y_m=0.53, yaw_deg=225.0, surface_z_base=-0.00755)},
                             gripper=dict(open_stop_deg=45.6, gap_mm_at_0=0.18, gap_mm_at_stop=144.0), solver={"stages": ["test"]},
                             provenance={"demo": "synthetic"}, task="test task")


class FakeDataset:
    """Records what a LeRobotDataset would receive."""

    def __init__(self):
        self.frames, self.episodes = [], []

    def add_frame(self, frame: dict):
        self.frames.append(frame)

    def save_episode(self):
        self.episodes.append(len(self.frames))


class FakeVideo:
    def __init__(self, n: int, h: int = 8, w: int = 6):
        self.n, self.height, self.width = n, h, w
        self.reads = []

    def read(self, i: int):
        self.reads.append(i)
        img = np.zeros((self.height, self.width, 3), np.uint8)
        img[..., 0] = i % 256
        return img
