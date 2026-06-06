"""Pinhole camera model and the fixed evaluation viewpoints.

Camera-space convention (right-handed, computer-vision style):
    +x points right, +y points up, +z points forward (into the scene).
A world point P projects to pixel (u, v) via::

    P_cam = R @ (P_world - eye)
    u = cx + fx * (x_cam / z_cam)
    v = cy - fy * (y_cam / z_cam)

with ``z_cam > 0`` for points in front of the camera. The minus sign on ``v``
maps world-up to the top of the image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector along ``v``.

    Args:
        v: A 1-D vector.

    Returns:
        The normalized vector (unchanged direction, length 1).
    """
    return v / (np.linalg.norm(v) + 1e-12)


@dataclass
class Camera:
    """A pinhole camera with intrinsics and a world-to-camera transform.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Horizontal focal length in pixels.
        fy: Vertical focal length in pixels.
        cx: Principal point x in pixels.
        cy: Principal point y in pixels.
        rotation: World-to-camera rotation matrix of shape (3, 3).
        eye: Camera position in world coordinates, shape (3,).
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: np.ndarray
    eye: np.ndarray

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        """Transform world points into camera space.

        Args:
            points: Array (N, 3) of world-space coordinates.

        Returns:
            Array (N, 3) of camera-space coordinates (z > 0 is in front).
        """
        return (points - self.eye[None, :]) @ self.rotation.T

    def project(self, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project camera-space points to pixel coordinates.

        Args:
            points_cam: Array (N, 3) of camera-space coordinates.

        Returns:
            Tuple ``(uv, depth)`` where ``uv`` is an (N, 2) array of pixel
            coordinates and ``depth`` is an (N,) array equal to ``z_cam``
            (positive in front of the camera).
        """
        z = points_cam[:, 2]
        z_safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.cx + self.fx * (points_cam[:, 0] / z_safe)
        v = self.cy - self.fy * (points_cam[:, 1] / z_safe)
        return np.stack([u, v], axis=1), z


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    fov_y_deg: float = 45.0,
) -> Camera:
    """Build a :class:`Camera` looking from ``eye`` toward ``target``.

    Args:
        eye: Camera position in world coordinates, shape (3,).
        target: Point the camera looks at, shape (3,).
        up: Approximate world up direction, shape (3,).
        width: Image width in pixels.
        height: Image height in pixels.
        fov_y_deg: Vertical field of view in degrees.

    Returns:
        A configured :class:`Camera`.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = _normalize(target - eye)
    right = _normalize(np.cross(up, forward))
    cam_up = np.cross(forward, right)
    rotation = np.stack([right, cam_up, forward], axis=0)

    fy = 0.5 * height / np.tan(np.radians(fov_y_deg) * 0.5)
    fx = fy
    cx = width * 0.5
    cy = height * 0.5
    return Camera(width, height, fx, fy, cx, cy, rotation, eye)


def fixed_viewpoints(
    center: np.ndarray,
    radius: float,
    width: int = 512,
    height: int = 512,
    fov_y_deg: float = 45.0,
    margin: float = 1.3,
) -> dict[str, Camera]:
    """Create the five fixed evaluation cameras around an object.

    Cameras are placed on the +/- world axes looking at ``center`` from a
    distance chosen so the bounding sphere of radius ``radius`` fits in view.

    Args:
        center: Object center in world coordinates, shape (3,).
        radius: Radius of the object's bounding sphere in world units.
        width: Image width in pixels.
        height: Image height in pixels.
        fov_y_deg: Vertical field of view in degrees.
        margin: Multiplicative slack so the object does not touch the borders.

    Returns:
        Dict mapping view name (``top``, ``left``, ``right``, ``front``,
        ``back``) to a :class:`Camera`.
    """
    center = np.asarray(center, dtype=np.float64)
    radius = max(float(radius), 1e-6)
    distance = margin * radius / np.tan(np.radians(fov_y_deg) * 0.5)

    y_up = np.array([0.0, 1.0, 0.0])
    z_up = np.array([0.0, 0.0, -1.0])
    specs = {
        "front": (np.array([0.0, 0.0, -1.0]), y_up),
        "back": (np.array([0.0, 0.0, 1.0]), y_up),
        "left": (np.array([-1.0, 0.0, 0.0]), y_up),
        "right": (np.array([1.0, 0.0, 0.0]), y_up),
        "top": (np.array([0.0, 1.0, 0.0]), z_up),
    }

    cams: dict[str, Camera] = {}
    for name, (direction, up) in specs.items():
        eye = center + direction * distance
        cams[name] = look_at(eye, center, up, width, height, fov_y_deg)
    return cams
