"""Transfer Gaussian colors onto mesh vertices."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .gaussian import Gaussians


def _inverse_covariances(covariances: np.ndarray) -> np.ndarray:
    """Invert a batch of 3x3 covariance matrices.

    Args:
        covariances: Array (N, 3, 3) of covariance matrices.

    Returns:
        Array (N, 3, 3) of inverses; singular matrices are replaced with zeros.
    """
    inv = np.empty_like(covariances)
    for i, cov in enumerate(covariances):
        try:
            inv[i] = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv[i] = 0.0
    return inv


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

    inv_covs = _inverse_covariances(gaussians.covariances())
    neigh_means = means[idx]
    d = vertices[:, None, :] - neigh_means
    inv = inv_covs[idx]
    quad = np.einsum("vki,vkij,vkj->vk", d, inv, d)
    weights = gaussians.opacities[idx] * np.exp(-0.5 * quad)
    neigh_colors = gaussians.colors[idx]

    total = weights.sum(axis=1, keepdims=True)
    fallback = neigh_colors[:, 0]
    weighted = (weights[..., None] * neigh_colors).sum(axis=1)
    out = np.where(total > 1e-12, weighted / np.maximum(total, 1e-12), fallback)

    rgb = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    alpha = np.full((vertices.shape[0], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=1)
