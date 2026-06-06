"""Voxelization of the Gaussian density field.

The object surface is defined as a level set of the continuous density field::

    f(x) = sum_i  opacity_i * exp(-0.5 * (x - mean_i)^T Sigma_i^-1 (x - mean_i))

This module samples ``f`` on a regular voxel grid. To stay tractable, each
Gaussian is only "splatted" into the local block of voxels within ``k`` standard
deviations of its center instead of being evaluated over the whole grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gaussian import Gaussians


@dataclass
class DensityGrid:
    """A sampled scalar density field on a regular axis-aligned voxel grid.

    Attributes:
        values: Array (Nx, Ny, Nz) of accumulated density values.
        origin: World coordinate of voxel (0, 0, 0) center, shape (3,).
        voxel_size: Edge length of a (cubic) voxel in world units.
    """

    values: np.ndarray
    origin: np.ndarray
    voxel_size: float

    def voxel_to_world(self, ijk: np.ndarray) -> np.ndarray:
        """Convert fractional voxel indices to world coordinates.

        Args:
            ijk: Array (..., 3) of (i, j, k) voxel indices (may be fractional).

        Returns:
            Array (..., 3) of world coordinates.
        """
        return self.origin[None, :] + ijk * self.voxel_size


def build_density_grid(
    gaussians: Gaussians,
    resolution: int = 256,
    sigma_cutoff: float = 3.0,
    padding_voxels: int = 3,
) -> DensityGrid:
    """Sample the Gaussian density field onto a voxel grid.

    Args:
        gaussians: The Gaussians defining the field (world space, activated).
        resolution: Number of voxels along the longest bounding-box axis; the
            voxel size is derived from this and reused for all three axes.
        sigma_cutoff: Number of standard deviations of each Gaussian to splat;
            larger captures more of each tail at higher cost.
        padding_voxels: Extra voxels of empty margin added on every side so the
            extracted surface is not clipped at the grid boundary.

    Returns:
        A :class:`DensityGrid` holding the accumulated field.
    """
    lo, hi = gaussians.bounds()
    extent = hi - lo
    longest = float(np.max(extent))
    voxel_size = longest / max(resolution - 1, 1)
    if voxel_size <= 0:
        voxel_size = 1.0

    origin = lo - padding_voxels * voxel_size
    dims = np.ceil((hi - origin) / voxel_size).astype(int) + padding_voxels + 1
    dims = np.maximum(dims, 1)
    grid = np.zeros(tuple(int(d) for d in dims), dtype=np.float32)

    covariances = gaussians.covariances()
    # Per-Gaussian splat radius (world units) from the largest principal axis.
    radii = sigma_cutoff * gaussians.scales.max(axis=1)

    means = gaussians.means
    opacities = gaussians.opacities

    for i in range(len(gaussians)):
        cov = covariances[i]
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue

        center = means[i]
        radius = radii[i]
        lo_idx = np.floor((center - radius - origin) / voxel_size).astype(int)
        hi_idx = np.ceil((center + radius - origin) / voxel_size).astype(int)
        lo_idx = np.maximum(lo_idx, 0)
        hi_idx = np.minimum(hi_idx, np.array(grid.shape) - 1)
        if np.any(lo_idx > hi_idx):
            continue

        xs = np.arange(lo_idx[0], hi_idx[0] + 1)
        ys = np.arange(lo_idx[1], hi_idx[1] + 1)
        zs = np.arange(lo_idx[2], hi_idx[2] + 1)
        wx = origin[0] + xs * voxel_size - center[0]
        wy = origin[1] + ys * voxel_size - center[1]
        wz = origin[2] + zs * voxel_size - center[2]

        gx, gy, gz = np.meshgrid(wx, wy, wz, indexing="ij")
        d = np.stack([gx, gy, gz], axis=-1)  # (nx, ny, nz, 3)

        # quad = d^T inv d, evaluated for every voxel in the block.
        quad = np.einsum("...a,ab,...b->...", d, inv, d)
        contrib = opacities[i] * np.exp(-0.5 * quad)

        grid[
            lo_idx[0] : hi_idx[0] + 1,
            lo_idx[1] : hi_idx[1] + 1,
            lo_idx[2] : hi_idx[2] + 1,
        ] += contrib.astype(np.float32)

    return DensityGrid(values=grid, origin=origin, voxel_size=float(voxel_size))
