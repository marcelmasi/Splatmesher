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
    shell_smooth_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a watertight outer shell from the sampled density field.

    Instead of marching cubes on a scalar iso-level (which follows individual
    Gaussian blobs and produces spongy, perforated surfaces), this builds a
    solid occupancy volume: voxels above a low density threshold are treated as
    inside the object, small gaps are closed morphologically, interior voids are
    filled, and the outer boundary is extracted.

    Marching cubes on a hard binary 0/1 volume produces terraced ("rippled")
    surfaces several triangles wide, because vertices snap to voxel midplanes
    with no gradient to interpolate along. To avoid this, the binary solid is
    blurred into a smooth scalar field before extraction so marching cubes can
    place vertices continuously.

    Args:
        grid: The sampled :class:`DensityGrid`.
        iso_relative: Occupancy threshold as a fraction of the field maximum,
            in (0, 1). Lower values include more outer splat material; higher
            values yield a tighter surface.
        pre_blur_sigma: Optional Gaussian blur sigma (in voxels) applied to the
            field before binarization to smooth the occupancy envelope (0 disables).
        close_iters: Morphological closing iterations on the occupancy mask to
            bridge narrow gaps between splats (0 disables).
        shell_smooth_sigma: Gaussian blur sigma (in voxels) applied to the binary
            solid volume before marching cubes; removes terracing ripples. Set to
            0 to extract directly from the hard binary volume (terraced).

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

    # Convert the binary solid into a smooth scalar field so marching cubes can
    # interpolate vertex positions and avoid stair-step terracing artifacts.
    field = solid.astype(np.float32)
    if shell_smooth_sigma > 0.0:
        from scipy.ndimage import gaussian_filter

        # Pad by one voxel so the blur does not clip the surface at array edges.
        field = np.pad(field, 1, mode="constant", constant_values=0.0)
        field = gaussian_filter(field, sigma=shell_smooth_sigma).astype(np.float32)
        origin_shift = grid.origin - grid.voxel_size
    else:
        origin_shift = grid.origin

    spacing = (grid.voxel_size, grid.voxel_size, grid.voxel_size)
    verts, faces, _normals, _vals = measure.marching_cubes(
        field,
        level=0.5,
        spacing=spacing,
    )
    verts = verts + origin_shift[None, :]
    return verts.astype(np.float64), faces.astype(np.int64)
