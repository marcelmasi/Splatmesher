"""Iso-surface extraction from a density grid via Marching Cubes."""

from __future__ import annotations

import numpy as np
from skimage import measure

from .field import DensityGrid


def extract_surface(
    grid: DensityGrid,
    iso_relative: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a triangle mesh at a relative iso-level of the density field.

    Args:
        grid: The sampled :class:`DensityGrid`.
        iso_relative: Iso-level as a fraction of the field maximum, in (0, 1).
            Lower values yield a larger / outer surface, higher values a tighter
            one. This normalization keeps the level scene-independent.

    Returns:
        Tuple ``(vertices, faces)`` where ``vertices`` is an (V, 3) float array in
        world coordinates and ``faces`` is an (F, 3) int array of triangle
        indices.

    Raises:
        ValueError: If the chosen iso-level does not intersect the field (e.g.
            an empty grid), so no surface can be extracted.
    """
    values = grid.values
    max_val = float(values.max())
    if max_val <= 0.0:
        raise ValueError("Density field is empty; cannot extract a surface.")

    level = iso_relative * max_val
    verts, faces, _normals, _vals = measure.marching_cubes(
        values.astype(np.float32),
        level=level,
        spacing=(grid.voxel_size, grid.voxel_size, grid.voxel_size),
    )
    # marching_cubes returns coordinates in (index * spacing); shift to world.
    verts = verts + grid.origin[None, :]
    return verts.astype(np.float64), faces.astype(np.int64)
