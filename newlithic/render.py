"""Offscreen renders (PNG) and videos (MP4) of a MuJoCo model."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


def render(model, data, lookat, distance, azimuth, elevation, width=1280, height=960, camera: str | None = None,
           renderer: "mujoco.Renderer | None" = None, group_sites: bool = True):
    r = renderer or mujoco.Renderer(model, height, width)
    opt = mujoco.MjvOption()
    opt.sitegroup[1] = group_sites
    if camera:
        cam = camera
    else:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    r.update_scene(data, cam, opt)
    img = r.render()
    if renderer is None:
        r.close()
    return img


VIEWS = {"iso": (135, -25), "side": (90, -5), "front": (180, -10), "top": (90, -89)}


def render_views(model, data, out_dir: Path, stem: str, lookat, distance: float, views: dict = VIEWS, width=1280, height=960):
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(model, height, width)
    paths = []
    for name, (az, el) in views.items():
        img = render(model, data, lookat, distance, az, el, width, height, renderer=r)
        p = out_dir / f"{stem}_{name}.png"
        Image.fromarray(img).save(p)
        paths.append(p)
    r.close()
    return paths


class VideoWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int):
        import imageio
        self.w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=8)
        self.width, self.height = width, height

    def add(self, img: np.ndarray):
        self.w.append_data(img)

    def close(self):
        self.w.close()
