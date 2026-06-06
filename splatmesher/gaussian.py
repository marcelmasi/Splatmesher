"""Data structures and math for 3D Gaussian Splatting primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# DC component factor of the real spherical harmonics basis (band 0).
SH_C0: float = 0.28209479177387814


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid.

    Args:
        x: Input array of arbitrary shape (raw logit values).

    Returns:
        Array of the same shape with values in the open interval (0, 1).
    """
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def quaternions_to_rotation_matrices(quats: np.ndarray) -> np.ndarray:
    """Convert a batch of quaternions (w, x, y, z) to rotation matrices.

    Args:
        quats: Array of shape (N, 4) holding quaternions in (w, x, y, z) order,
            as stored by the INRIA 3DGS PLY format. They need not be normalized.

    Returns:
        Array of shape (N, 3, 3) with one rotation matrix per quaternion.
    """
    q = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    n = q.shape[0]
    r = np.empty((n, 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


@dataclass
class Gaussians:
    """A collection of 3D Gaussians in world space (already activated).

    Attributes:
        means: Array (N, 3) of Gaussian centers in world units (e.g. meters).
        scales: Array (N, 3) of per-axis standard deviations (world units),
            i.e. exp() of the raw log-scales stored in the PLY.
        quats: Array (N, 4) of orientation quaternions in (w, x, y, z) order.
        opacities: Array (N,) of opacities in [0, 1] (sigmoid of raw values).
        colors: Array (N, 3) of base RGB colors in [0, 1] (SH DC term).
    """

    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray

    def __len__(self) -> int:
        """Return the number of Gaussians in the collection."""
        return int(self.means.shape[0])

    def rotation_matrices(self) -> np.ndarray:
        """Return rotation matrices for all Gaussians.

        Returns:
            Array (N, 3, 3) of rotation matrices built from the quaternions.
        """
        return quaternions_to_rotation_matrices(self.quats)

    def covariances(self) -> np.ndarray:
        """Compute the 3x3 world-space covariance matrices.

        Returns:
            Array (N, 3, 3) where each matrix equals R diag(scale^2) R^T.
        """
        r = self.rotation_matrices()
        s2 = self.scales**2
        # M = R * S (scale columns), then Sigma = M M^T.
        m = r * s2[:, None, :]
        return m @ np.transpose(r, (0, 2, 1))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute an axis-aligned bounding box of the Gaussian centers.

        Returns:
            Tuple (min_xyz, max_xyz), each an array of shape (3,) in world units.
        """
        return self.means.min(axis=0), self.means.max(axis=0)

    def select(self, mask: np.ndarray) -> "Gaussians":
        """Return a subset of the Gaussians selected by a boolean/index mask.

        Args:
            mask: Boolean array (N,) or an integer index array selecting which
                Gaussians to keep.

        Returns:
            A new :class:`Gaussians` containing only the selected primitives.
        """
        return Gaussians(
            means=self.means[mask],
            scales=self.scales[mask],
            quats=self.quats[mask],
            opacities=self.opacities[mask],
            colors=self.colors[mask],
        )


def filter_gaussians(
    gaussians: Gaussians,
    min_opacity: float = 0.1,
    outlier_neighbors: int = 16,
    outlier_std_ratio: float = 2.0,
) -> Gaussians:
    """Remove transparent Gaussians and spatial floaters/outliers.

    The cleanup first drops low-opacity Gaussians, then performs statistical
    outlier removal: a Gaussian is discarded if its mean distance to its
    nearest neighbors is far above the global average (a floater in empty space).

    Args:
        gaussians: The input Gaussians to filter.
        min_opacity: Gaussians with opacity below this are removed; set to 0 to
            disable opacity filtering.
        outlier_neighbors: Number of nearest neighbors used to estimate local
            density for outlier removal.
        outlier_std_ratio: A Gaussian is an outlier if its mean neighbor
            distance exceeds ``global_mean + ratio * global_std``. Set to 0 to
            disable outlier removal.

    Returns:
        A new :class:`Gaussians` with the surviving primitives.
    """
    from scipy.spatial import cKDTree

    g = gaussians
    if min_opacity > 0.0:
        g = g.select(g.opacities >= min_opacity)
    if len(g) == 0:
        return g

    if outlier_std_ratio > 0.0 and len(g) > outlier_neighbors + 1:
        tree = cKDTree(g.means)
        k = outlier_neighbors + 1  # include the point itself
        dists, _ = tree.query(g.means, k=k)
        mean_dist = dists[:, 1:].mean(axis=1)
        threshold = mean_dist.mean() + outlier_std_ratio * mean_dist.std()
        g = g.select(mean_dist <= threshold)

    return g
