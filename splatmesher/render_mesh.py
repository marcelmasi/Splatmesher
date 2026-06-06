"""CPU software rasterizer for triangle meshes.

Headless, dependency-light z-buffer rasterizer with simple Lambertian shading
from a camera-attached headlight. Shares the :class:`Camera` model with the
Gaussian splat renderer so the two can be compared pixel-for-pixel.
"""

from __future__ import annotations

import numpy as np

from .camera import Camera


def render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: Camera,
    base_color: tuple[float, float, float] = (0.7, 0.7, 0.7),
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ambient: float = 0.25,
) -> np.ndarray:
    """Render a triangle mesh to an RGB image with a z-buffer.

    Args:
        vertices: Array (V, 3) of vertex positions in world units.
        faces: Array (F, 3) of vertex indices (one triangle per row).
        camera: The camera defining the viewpoint and intrinsics.
        base_color: Diffuse RGB albedo in [0, 1] applied to all faces.
        background: Background RGB color in [0, 1] where no triangle is hit.
        ambient: Ambient light term in [0, 1] added to the diffuse shading.

    Returns:
        Array (H, W, 3) of float32 RGB values in [0, 1].
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    h, w = camera.height, camera.width

    image = np.tile(np.asarray(background, dtype=np.float64), (h, w, 1))
    zbuffer = np.full((h, w), np.inf, dtype=np.float64)

    verts_cam = camera.world_to_camera(vertices)
    uv, depth = camera.project(verts_cam)

    base = np.asarray(base_color, dtype=np.float64)

    for tri in faces:
        cam_pts = verts_cam[tri]
        # Skip triangles with any vertex behind the camera.
        if np.any(cam_pts[:, 2] <= 1e-4):
            continue

        p = uv[tri]
        z = depth[tri]

        x_min = int(np.floor(p[:, 0].min()))
        x_max = int(np.ceil(p[:, 0].max()))
        y_min = int(np.floor(p[:, 1].min()))
        y_max = int(np.ceil(p[:, 1].max()))
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, w - 1)
        y_max = min(y_max, h - 1)
        if x_min > x_max or y_min > y_max:
            continue

        # Edge-function / barycentric setup in pixel space.
        x0, y0 = p[0]
        x1, y1 = p[1]
        x2, y2 = p[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue

        xs = np.arange(x_min, x_max + 1) + 0.5
        ys = np.arange(y_min, y_max + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)

        l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        l2 = 1.0 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not np.any(inside):
            continue

        frag_z = l0 * z[0] + l1 * z[1] + l2 * z[2]

        # Flat shading: world-space face normal lit by a headlight at the eye.
        v0, v1, v2 = vertices[tri]
        normal = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            continue
        normal /= norm_len
        centroid = (v0 + v1 + v2) / 3.0
        light_dir = camera.eye - centroid
        light_dir /= np.linalg.norm(light_dir) + 1e-12
        intensity = ambient + (1.0 - ambient) * abs(float(np.dot(normal, light_dir)))
        shade = np.clip(base * intensity, 0.0, 1.0)

        sub_z = zbuffer[y_min : y_max + 1, x_min : x_max + 1]
        update = inside & (frag_z < sub_z)
        sub_z[update] = frag_z[update]
        sub_img = image[y_min : y_max + 1, x_min : x_max + 1, :]
        sub_img[update] = shade

    return np.clip(image, 0.0, 1.0).astype(np.float32)
