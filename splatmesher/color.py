"""Transfer Gaussian colors onto mesh vertices."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .gaussian import Gaussians


def transfer_colors(
    vertices: np.ndarray,
    gaussians: Gaussians,
    k: int = 8,
) -> np.ndarray:
    """Assign an RGB color to each mesh vertex from nearby Gaussians.

    For every vertex the ``k`` nearest Gaussians are found and their colors are
    blended using each Gaussian's actual density (opacity times anisotropic
    Gaussian weight) at the vertex, so closer and more opaque Gaussians dominate.

    Args:
        vertices: Array (V, 3) of mesh vertex positions in world units.
        gaussians: The source Gaussians providing positions and colors.
        k: Number of nearest Gaussians to consider per vertex.

    Returns:
        Array (V, 4) of uint8 RGBA vertex colors (alpha fixed at 255).
    """
    means = gaussians.means
    tree = cKDTree(means)
    k_eff = min(k, len(gaussians))
    _dist, idx = tree.query(vertices, k=k_eff)
    if k_eff == 1:
        idx = idx[:, None]

    covariances = gaussians.covariances()
    colors = gaussians.colors
    opacities = gaussians.opacities

    out = np.zeros((vertices.shape[0], 3), dtype=np.float64)
    for n in range(vertices.shape[0]):
        neigh = idx[n]
        d = vertices[n][None, :] - means[neigh]  # (k, 3)
        weights = np.empty(k_eff, dtype=np.float64)
        for j, g in enumerate(neigh):
            try:
                inv = np.linalg.inv(covariances[g])
            except np.linalg.LinAlgError:
                weights[j] = 0.0
                continue
            quad = float(d[j] @ inv @ d[j])
            weights[j] = opacities[g] * np.exp(-0.5 * quad)
        total = weights.sum()
        if total <= 1e-12:
            out[n] = colors[neigh[0]]
        else:
            out[n] = (weights[:, None] * colors[neigh]).sum(axis=0) / total

    rgb = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    alpha = np.full((vertices.shape[0], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=1)
