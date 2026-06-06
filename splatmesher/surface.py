"""Solid-shell surface extraction from a density grid via Marching Cubes."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure
from skimage import measure

from .field import DensityGrid


def extract_surface(
    grid: DensityGrid,
    iso_relative: float = 0.04,
    pre_blur_sigma: float = 0.0,
    close_iters: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a watertight outer shell from the sampled density field.

    Instead of marching cubes on a scalar iso-level (which follows individual
    Gaussian blobs and produces spongy, perforated surfaces), this builds a
    solid occupancy volume: voxels above a low density threshold are treated as
    inside the object, small gaps are closed morphologically, interior voids are
    filled, and the outer boundary is extracted.

    Args:
        grid: The sampled :class:`DensityGrid`.
        iso_relative: Occupancy threshold as a fraction of the field maximum,
            in (0, 1). Lower values include more outer splat material; higher
            values yield a tighter surface.
        pre_blur_sigma: Optional Gaussian blur sigma (in voxels) applied to the
            field before binarization to smooth the occupancy envelope (0 disables).
        close_iters: Morphological closing iterations on the occupancy mask to
            bridge narrow gaps between splats (0 disables).

    Returns:
        Tuple ``(vertices, faces)`` where ``vertices`` is an (V, 3) float array in
        world coordinates and ``faces`` is an (F, 3) int array of triangle
        indices.

    Raises:
        ValueError: If the field is empty or no solid volume can be formed.
    """
    values = grid.values.astype(np.float32)
    max_val = float(values.max())
    if max_val <= 0.0:
        raise ValueError("Density field is empty; cannot extract a surface.")

    if pre_blur_sigma > 0.0:
        from scipy.ndimage import gaussian_filter

        values = gaussian_filter(values, sigma=pre_blur_sigma).astype(np.float32)
        max_val = float(values.max())
        if max_val <= 0.0:
            raise ValueError("Density field is empty after blur; cannot extract a surface.")

    threshold = iso_relative * max_val
    mask = values > threshold
    if not np.any(mask):
        raise ValueError(
            "Density threshold does not intersect the field; try lowering --iso."
        )

    if close_iters > 0:
        struct = generate_binary_structure(3, 2)
        mask = binary_closing(mask, structure=struct, iterations=close_iters)

    solid = binary_fill_holes(mask)
    if not np.any(solid):
        raise ValueError("Solid occupancy volume is empty; cannot extract a surface.")

    spacing = (grid.voxel_size, grid.voxel_size, grid.voxel_size)
    verts, faces, _normals, _vals = measure.marching_cubes(
        solid.astype(np.float32),
        level=0.5,
        spacing=spacing,
    )
    verts = verts + grid.origin[None, :]
    return verts.astype(np.float64), faces.astype(np.int64)
