"""Stage 0b: the wrist-camera videos the dataset uses as `observation.images.*`.

Lens 1 of each rig sits behind the jaws and looks along the fingers, so it already is a wrist camera, only through a ~200 deg
fisheye.  The view rendered here is a virtual pinhole camera at the centre of lens 1, rigidly attached to it (it turns with
the hand, no stabilisation): x along the jaw opening (left jaw on the left), y down, the optical axis pitched so that the
TCP lands at `tcp_row` of the image height, the jaws at the bottom and the scene ahead above them.  The camera never moves
with respect to the lens, so the fisheye -> pinhole map is one fixed remap table per rig.

Both videos are written on the LEFT rig's clock, one frame per demonstration frame (the right rig's nearest frame), so the
export stage can match frames by index.  Frames are decoded straight from the Insta360 `.insv` files (an MP4 container),
rendered in parallel chunks, each piped into its own ffmpeg (imageio-ffmpeg's binary), then concatenated.
"""
from __future__ import annotations

import subprocess
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np

from .demo_export import Rig, load_registration, load_rig, skeleton_tcp

FPS = "30000/1001"


def lens1_window(path) -> tuple:
    """(x0, width, height) of lens 1 inside the frames of `path`: the whole frame of a 5.7K lens file, the right half of a
    3K dual-fisheye file (2944 x 1472: lens 0 left, lens 1 right)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not h:
        raise FileNotFoundError(f"cannot open {path}")
    return (w // 2, w // 2, h) if w == 2 * h else (0, w, h)


class WristView:
    def __init__(self, rig: Rig, W: int = 640, H: int = 880, hfov_deg: float = 100.0, tcp_row: float = 0.70, src_scale: Optional[int] = None):
        import cv2
        from state_estimation.gripper import jaw_points, tips
        from state_estimation.mei import MeiCamera
        from state_estimation.telemetry import read_telemetry
        if not rig.files.get(1):
            raise FileNotFoundError(f"{rig.name}: the lens-1 .insv of clip {rig.stem} was not found under the data directory")
        tm = read_telemetry(rig.files[0] or rig.files[1])
        self.x0, lens_w, lens_h = lens1_window(rig.files[1])
        cam = MeiCamera.from_offset_v3(tm.offset_v3, 1, tm.crop, video_size=(lens_w, lens_h))
        if src_scale is None:
            src_scale = max(1, round(lens_w / 1440))               # remap from ~1440 px: half of a 5.7K lens, all of a 3K one
        self.lens_w = lens_w
        self.rig, self.W, self.H, self.src_scale = rig, int(W), int(H), int(src_scale)
        T = rig.T_lens_base
        qL, qR = np.median(rig.q, 0)
        to_lens = lambda P: np.atleast_2d(P) @ T[:3, :3].T + T[:3, 3]
        tL, tR = tips(rig.geom, qL, qR)
        tcp = to_lens(0.5 * (np.asarray(tL).reshape(3) + np.asarray(tR).reshape(3)))[0]
        tL, tR = to_lens(np.asarray(tL).reshape(3))[0], to_lens(np.asarray(tR).reshape(3))[0]
        self.f = 0.5 * self.W / np.tan(np.radians(hfov_deg) / 2)
        # x: the jaw opening, z: the optical axis, pitched about x until the TCP sits at tcp_row of the height
        x = tR - tL
        x[1] = 0.0
        x /= np.linalg.norm(x)
        t = tcp / np.linalg.norm(tcp)
        want = np.arctan((tcp_row - 0.5) * self.H / self.f)                     # angle below the axis the TCP should be at
        y0 = np.array([0.0, 1.0, 0.0])                                         # the lens's own "down", already orthogonal to x
        z0 = np.cross(x, y0)
        have = np.arctan2(t @ y0, t @ z0)
        a = have - want                                                          # turn the axis down by a
        z = np.cos(a) * z0 + np.sin(a) * y0
        y = np.cross(z, x)
        self.R_lens_view = np.stack([x, y, z], 1)                                # view -> lens
        u, v = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)
        rays = np.stack([(u - self.W / 2) / self.f, (v - self.H / 2) / self.f, np.ones_like(u)], -1)
        uv, valid = cam.project(rays @ self.R_lens_view.T)
        uv = (uv + 0.5) / self.src_scale - 0.5
        uv[~valid] = -1e4
        self.mx, self.my = uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)
        self._cv2 = cv2

    def render(self, lens1_img: np.ndarray) -> np.ndarray:
        """lens1_img: the lens-1 frame (BGR), already shrunk by src_scale."""
        return self._cv2.remap(lens1_img, self.mx, self.my, self._cv2.INTER_LINEAR, borderMode=self._cv2.BORDER_CONSTANT)


class Reader:
    """Sequential frames of one lens file from a starting frame; get(i) with non-decreasing i (repeats allowed)."""

    def __init__(self, path, start, scale, x0: int = 0, width: Optional[int] = None):
        import cv2
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open {path}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        self.next, self.img, self.scale = int(start), None, scale
        self.x0, self.width = int(x0), width

    def get(self, i):
        i = int(i)
        if i < self.next - 1:
            self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, i)
            self.next = i
        while self.next <= i:
            ok, im = self.cap.read()
            self.next += 1
            if ok and self.width:
                im = im[:, self.x0:self.x0 + self.width]
            if ok:
                self.img = (self._cv2.resize(im, (im.shape[1] // self.scale, im.shape[0] // self.scale), interpolation=self._cv2.INTER_AREA)
                            if self.scale > 1 else im)
        if self.img is None:
            raise IOError(f"could not decode frame {i}")
        return self.img


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def writer(path, W, H, crf=18, preset="medium", fps: str = FPS):
    return subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                             "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", "4",
                             "-x264-params", "keyint=60:min-keyint=60:scenecut=0", str(path)], stdin=subprocess.PIPE)


def concat(parts, out: Path):
    lst = Path(out).with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in parts))
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    lst.unlink()


def _chunks(a, b, n):
    edges = np.linspace(a, b, n + 1).round().astype(int)
    return [(int(x), int(y)) for x, y in zip(edges[:-1], edges[1:]) if y > x]


def _work(job):
    results, episode, data, start, end, paths, view_kw, crf = job
    reg = load_registration(results, episode)
    rigs = {n: load_rig(results, episode, n, data, reg) for n in ("left", "right")}
    L, R = rigs["left"], rigs["right"]
    vw = {n: WristView(r, **view_kw) for n, r in rigs.items()}
    m_of = R.frame_at(L.t)
    rd = {"left": Reader(L.files[1], start, vw["left"].src_scale, vw["left"].x0, vw["left"].lens_w),
          "right": Reader(R.files[1], m_of[start], vw["right"].src_scale, vw["right"].x0, vw["right"].lens_w)}
    ff = {n: writer(paths[n], vw[n].W, vw[n].H, crf=crf) for n in rigs}
    t0 = time.time()
    for i in range(start, end):
        ff["left"].stdin.write(vw["left"].render(rd["left"].get(i)).tobytes())
        ff["right"].stdin.write(vw["right"].render(rd["right"].get(m_of[i])).tobytes())
    for f in ff.values():
        f.stdin.close()
        f.wait()
    return paths, end - start, time.time() - t0


def render_wrist_videos(results: Path, episode: str, data: Path, out_dir: Path, workers: int = 4, frames: Optional[tuple] = None,
                        W: int = 640, H: int = 880, hfov_deg: float = 100.0, tcp_row: float = 0.70, src_scale: Optional[int] = None, crf: int = 18,
                        log=print) -> dict:
    """Writes <out_dir>/<episode>_wrist_left.mp4 and _right.mp4 (640 x 880 by default, 29.97 fps, one frame per demonstration frame).
    frames: (a, b) sub-range of the left clip's frames, for tests.  Returns name -> path."""
    results, data, out_dir = Path(results), Path(data), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = load_registration(results, episode)
    L = load_rig(results, episode, "left", data, reg)
    a, b = (0, L.n) if frames is None else (max(0, int(frames[0])), min(L.n, int(frames[1])))
    view_kw = dict(W=W, H=H, hfov_deg=hfov_deg, tcp_row=tcp_row, src_scale=src_scale)
    parts = out_dir / "parts"
    parts.mkdir(exist_ok=True)
    jobs = [(str(results), episode, str(data), s, e, {n: str(parts / f"wrist_{n}_{s:05d}.mp4") for n in ("left", "right")}, view_kw, crf)
            for s, e in _chunks(a, b, max(1, workers))]
    t0 = time.time()
    if len(jobs) == 1:
        done = [_work(jobs[0])]
    else:
        with Pool(len(jobs)) as pool:
            done = pool.map(_work, jobs)
    tag = "" if frames is None else f"_{a}-{b}"
    out = {}
    for n in ("left", "right"):
        out[n] = out_dir / f"{episode}_wrist_{n}{tag}.mp4"
        concat([d[0][n] for d in done], out[n])
        for d in done:
            Path(d[0][n]).unlink()
    if not any(parts.iterdir()):
        parts.rmdir()
    log(f"wrist videos: {b - a} frames in {time.time() - t0:.0f} s -> {out['left']}, {out['right']}")
    return out
