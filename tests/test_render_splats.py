"""Render a real example splat from the 5 fixed viewpoints (debug + sanity)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from splatmesher.camera import fixed_viewpoints
from splatmesher.imageio import save_image
from splatmesher.io_ply import load_gaussians
from splatmesher.render_splats import render_gaussians

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_PLY = os.path.join(REPO_ROOT, "Examples", "FrangipaniCropped.ply")
OUT_DIR = os.path.join(REPO_ROOT, "debug_renders", "splats")


def _robust_center_radius(means: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate object center and radius, ignoring far-away floaters.

    Args:
        means: Array (N, 3) of Gaussian centers in world units.

    Returns:
        Tuple ``(center, radius)`` where ``center`` is shape (3,) and ``radius``
        is the bounding-sphere radius (world units) of the robust bounding box.
    """
    lo = np.percentile(means, 1.0, axis=0)
    hi = np.percentile(means, 99.0, axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))
    return center, radius


@pytest.mark.skipif(not os.path.exists(EXAMPLE_PLY), reason="example PLY missing")
def test_render_splats_to_files() -> None:
    """Load an example splat, render all 5 views, and write debug PNGs."""
    os.makedirs(OUT_DIR, exist_ok=True)
    gaussians = load_gaussians(EXAMPLE_PLY)
    assert len(gaussians) > 0

    center, radius = _robust_center_radius(gaussians.means)
    cams = fixed_viewpoints(center, radius, width=512, height=512)

    for name, cam in cams.items():
        img = render_gaussians(gaussians, cam)
        assert img.shape == (512, 512, 3)
        assert np.isfinite(img).all()
        # The object must actually paint something (not a blank background).
        non_white = np.mean(np.any(img < 0.95, axis=2))
        assert non_white > 0.01, f"view {name} looks empty ({non_white:.4f})"
        save_image(img, os.path.join(OUT_DIR, f"{name}.png"))
