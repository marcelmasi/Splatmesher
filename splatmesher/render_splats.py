"""CPU EWA-style rasterizer for 3D Gaussian Splats.

This is not meant to be fast; it is a dependency-light, headless renderer used
for evaluation and debugging. Each Gaussian is projected to a 2D Gaussian
(EWA splatting) and composited front-to-back with alpha blending.
"""

from __future__ import annotations

import numpy as np

from .camera import Camera
from .gaussian import Gaussians


def render_gaussians(
    gaussians: Gaussians,
    camera: Camera,
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigma_cutoff: float = 3.0,
    min_opacity: float = 1.0 / 255.0,
) -> np.ndarray:
    """Render Gaussians to an RGB image using front-to-back alpha blending.

    Args:
        gaussians: The Gaussians to render (world space, activated values).
        camera: The camera defining the viewpoint and intrinsics.
        background: Background RGB color in [0, 1] shown where nothing covers.
        sigma_cutoff: How many standard deviations of each splat to rasterize;
            larger is more accurate but slower.
        min_opacity: Gaussians with opacity below this are skipped.

    Returns:
        Array (H, W, 3) of float32 RGB values in [0, 1].
    """
    h, w = camera.height, camera.width
    color = np.zeros((h, w, 3), dtype=np.float64)
    transmittance = np.ones((h, w), dtype=np.float64)

    means_cam = camera.world_to_camera(gaussians.means)
    uv, depth = camera.project(means_cam)

    cov3d = gaussians.covariances()
    rot = camera.rotation

    # Visibility prefilter: in front of the camera and bright enough.
    visible = (depth > 1e-4) & (gaussians.opacities > min_opacity)
    idx = np.nonzero(visible)[0]
    # Composite near-to-far so transmittance accumulates correctly.
    idx = idx[np.argsort(depth[idx])]

    for i in idx:
        z = depth[i]
        cx, cy = means_cam[i, 0], means_cam[i, 1]

        # Jacobian of the perspective projection (pixels per camera unit).
        j = np.array(
            [
                [camera.fx / z, 0.0, -camera.fx * cx / (z * z)],
                [0.0, -camera.fy / z, camera.fy * cy / (z * z)],
            ]
        )
        t = j @ rot  # 2x3: world -> image-plane derivative
        cov2d = t @ cov3d[i] @ t.T
        cov2d[0, 0] += 0.3  # low-pass filter so tiny splats stay >= 1 px
        cov2d[1, 1] += 0.3

        det = cov2d[0, 0] * cov2d[1, 1] - cov2d[0, 1] * cov2d[1, 0]
        if det <= 1e-12:
            continue
        inv = np.array(
            [[cov2d[1, 1], -cov2d[0, 1]], [-cov2d[1, 0], cov2d[0, 0]]]
        ) / det

        # Pixel bounding box from the 2D covariance eigenvalues.
        mid = 0.5 * (cov2d[0, 0] + cov2d[1, 1])
        disc = max(mid * mid - det, 0.0)
        lambda_max = mid + np.sqrt(disc)
        radius = int(np.ceil(sigma_cutoff * np.sqrt(max(lambda_max, 1e-6))))
        if radius < 1:
            radius = 1

        u0, v0 = uv[i, 0], uv[i, 1]
        x_min = max(int(np.floor(u0 - radius)), 0)
        x_max = min(int(np.ceil(u0 + radius)), w - 1)
        y_min = max(int(np.floor(v0 - radius)), 0)
        y_max = min(int(np.ceil(v0 + radius)), h - 1)
        if x_min > x_max or y_min > y_max:
            continue

        xs = np.arange(x_min, x_max + 1)
        ys = np.arange(y_min, y_max + 1)
        dx = xs[None, :] - u0
        dy = ys[:, None] - v0
        power = -0.5 * (
            inv[0, 0] * dx * dx
            + (inv[0, 1] + inv[1, 0]) * dx * dy
            + inv[1, 1] * dy * dy
        )
        weight = np.exp(np.minimum(power, 0.0))
        alpha = np.clip(gaussians.opacities[i] * weight, 0.0, 0.99)

        patch_t = transmittance[y_min : y_max + 1, x_min : x_max + 1]
        contrib = patch_t * alpha
        color[y_min : y_max + 1, x_min : x_max + 1] += (
            contrib[:, :, None] * gaussians.colors[i][None, None, :]
        )
        transmittance[y_min : y_max + 1, x_min : x_max + 1] = patch_t * (1.0 - alpha)

    bg = np.asarray(background, dtype=np.float64)
    color += transmittance[:, :, None] * bg[None, None, :]
    return np.clip(color, 0.0, 1.0).astype(np.float32)
