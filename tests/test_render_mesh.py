"""Render a simple cube mesh from the 5 fixed viewpoints (debug + sanity)."""

from __future__ import annotations

import os

import numpy as np
import trimesh

from splatmesher.camera import fixed_viewpoints
from splatmesher.imageio import save_image
from splatmesher.render_mesh import render_mesh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "debug_renders", "mesh")


def test_render_cube_to_files() -> None:
    """Build a unit cube, render all 5 views, and write debug PNGs."""
    os.makedirs(OUT_DIR, exist_ok=True)

    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = np.asarray(cube.vertices, dtype=np.float64)
    faces = np.asarray(cube.faces, dtype=np.int64)

    center = vertices.mean(axis=0)
    radius = 0.5 * float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    cams = fixed_viewpoints(center, radius, width=512, height=512)

    for name, cam in cams.items():
        img = render_mesh(vertices, faces, cam, base_color=(0.2, 0.55, 0.9))
        assert img.shape == (512, 512, 3)
        assert np.isfinite(img).all()
        # The cube should cover a meaningful chunk of the frame.
        covered = np.mean(np.any(img < 0.95, axis=2))
        assert covered > 0.05, f"view {name} looks empty ({covered:.4f})"
        save_image(img, os.path.join(OUT_DIR, f"{name}.png"))
