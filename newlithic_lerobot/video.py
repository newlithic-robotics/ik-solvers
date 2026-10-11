"""Camera frames for the dataset, read from a video that is already on the demonstration's clock (one video frame per
demonstration frame): the rendered wrist views of the two rigs (`state-estimation-debugging/scripts/render.py wrist`,
a virtual pinhole camera fixed to each gripper, the right rig resampled to the left rig's clock).

Reads are sequential-friendly: frames are decoded in order and a repeated index returns the cached frame, so a nearest-frame
mapping that only moves forward never seeks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class VideoFrames:
    def __init__(self, path: Path):
        import cv2
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video {self.path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self._next = 0
        self._last_index = -1
        self._last = None
        self._cv2 = cv2

    @property
    def shape(self):
        return (self.height, self.width, 3)

    def read(self, index: int) -> np.ndarray:
        """Frame `index` as RGB uint8 (h, w, 3)."""
        index = int(index)
        if index == self._last_index:
            return self._last
        if index < 0 or index >= self.n:
            raise IndexError(f"{self.path}: frame {index} of {self.n}")
        if index != self._next:
            self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, index)
            self._next = index
        ok, bgr = self.cap.read()
        if not ok:
            raise IOError(f"{self.path}: could not decode frame {index}")
        self._next = index + 1
        self._last_index, self._last = index, self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        return self._last

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
