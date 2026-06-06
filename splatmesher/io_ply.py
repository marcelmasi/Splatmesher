"""Reading 3D Gaussian Splatting PLY files into :class:`Gaussians`."""

from __future__ import annotations

import numpy as np
from plyfile import PlyData

from .gaussian import SH_C0, Gaussians, sigmoid


def load_gaussians(path: str) -> Gaussians:
    """Load a 3D Gaussian Splatting PLY file into a :class:`Gaussians` object.

    The function expects the standard INRIA 3DGS vertex properties:
    ``x, y, z``, ``f_dc_0..2``, ``opacity``, ``scale_0..2`` and ``rot_0..3``.
    Raw values are activated (exp on scales, sigmoid on opacity, SH-DC to RGB)
    so the returned Gaussians are ready to render.

    Args:
        path: Filesystem path to the ``.ply`` file to read.

    Returns:
        A :class:`Gaussians` instance with N primitives in world space.

    Raises:
        KeyError: If a required property is missing from the PLY vertex element.
    """
    ply = PlyData.read(path)
    v = ply["vertex"].data

    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    scales = np.exp(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float64)
    )
    quats = np.stack(
        [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1
    ).astype(np.float64)
    opacities = sigmoid(np.asarray(v["opacity"], dtype=np.float64))

    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float64)
    colors = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)

    return Gaussians(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        colors=colors,
    )
