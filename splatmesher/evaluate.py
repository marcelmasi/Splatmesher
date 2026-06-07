"""Evaluation: compare mesh renders against Gaussian Splat reference renders."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import trimesh

from .camera import Camera, fixed_viewpoints
from .config import ConvertConfig
from .io_ply import load_gaussians
from .pipeline import convert_stats, convert_to_mesh
from .render_mesh import render_mesh
from .render_splats import render_gaussians

VIEW_NAMES = ("top", "bottom", "left", "right", "front", "back")


@dataclass
class EvalResult:
    """Outcome of evaluating one mesh against a splat reference.

    Attributes:
        l1_error: Mean L1 RGB difference averaged over all five views.
        per_view_l1: L1 error per view name.
        stats: Mesh quality metrics (components, watertight, etc.).
        passed_integrity: True if the mesh is not overly fragmented.
    """

    l1_error: float
    per_view_l1: dict[str, float]
    stats: dict[str, float | int | bool]
    passed_integrity: bool


def robust_center_radius(means: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate object center and radius, ignoring far-away floaters.

    Args:
        means: Array (N, 3) of Gaussian centers in world units.

    Returns:
        Tuple ``(center, radius)`` where ``center`` is shape (3,) and ``radius``
        is the bounding-sphere radius (world units) of the robust bounding box.
    """
    lo = np.percentile(means, 1.0, axis=0)
    hi = np.percentile(means, 99.0, axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))
    return center, radius


def render_splat_references(
    ply_path: str,
    width: int = 256,
    height: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, Camera]]:
    """Render reference images from the five fixed viewpoints.

    Args:
        ply_path: Path to the Gaussian Splatting ``.ply`` file.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Tuple of (images dict, cameras dict) keyed by view name.
    """
    gaussians = load_gaussians(ply_path)
    center, radius = robust_center_radius(gaussians.means)
    cameras = fixed_viewpoints(center, radius, width=width, height=height)
    images = {name: render_gaussians(gaussians, cam) for name, cam in cameras.items()}
    return images, cameras


def l1_image_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean L1 RGB difference between two float images in [0, 1].

    Only pixels where at least one image differs from white background are
    included, so empty background does not dominate the score.

    Args:
        a: Reference image (H, W, 3) float32 in [0, 1].
        b: Comparison image (H, W, 3) float32 in [0, 1].

    Returns:
        Mean L1 distance in [0, 1] over the masked pixels.
    """
    bg = 0.98
    mask = np.any(a < bg, axis=2) | np.any(b < bg, axis=2)
    if not np.any(mask):
        return 1.0
    diff = np.abs(a - b)
    return float(diff[mask].mean())


def mesh_passes_integrity(
    stats: dict[str, float | int | bool],
    max_components: int = 3,
    min_main_fraction: float = 0.92,
) -> bool:
    """Check that a mesh is not broken into too many disconnected parts.

    Args:
        stats: Output of :func:`convert_stats`.
        max_components: Maximum allowed connected components after post-processing.
        min_main_fraction: The largest component must hold at least this fraction
            of all faces.

    Returns:
        True if the mesh passes the integrity constraints.
    """
    num = int(stats["num_components"])
    main_frac = float(stats["main_component_fraction"])
    return num <= max_components and main_frac >= min_main_fraction


def evaluate_mesh(
    mesh: trimesh.Trimesh,
    reference_images: dict[str, np.ndarray],
    cameras: dict[str, Camera],
    max_components: int = 3,
    min_main_fraction: float = 0.92,
) -> EvalResult:
    """Score a mesh against pre-rendered splat reference images.

    Args:
        mesh: The mesh to evaluate (with vertex colors for fair comparison).
        reference_images: Splat reference renders keyed by view name.
        cameras: Cameras used for the reference renders.
        max_components: Maximum allowed connected components.
        min_main_fraction: Minimum fraction of faces in the largest component.

    Returns:
        :class:`EvalResult` with L1 error and integrity flags.
    """
    stats = convert_stats(mesh)
    passed = mesh_passes_integrity(stats, max_components, min_main_fraction)

    vcols = None
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vcols = np.asarray(mesh.visual.vertex_colors)

    per_view: dict[str, float] = {}
    for name in VIEW_NAMES:
        mesh_img = render_mesh(
            np.asarray(mesh.vertices),
            np.asarray(mesh.faces),
            cameras[name],
            vertex_colors=vcols,
        )
        per_view[name] = l1_image_error(reference_images[name], mesh_img)

    l1 = float(np.mean(list(per_view.values())))
    return EvalResult(
        l1_error=l1,
        per_view_l1=per_view,
        stats=stats,
        passed_integrity=passed,
    )


def evaluate_config(
    ply_path: str,
    config: ConvertConfig,
    reference_images: dict[str, np.ndarray] | None = None,
    cameras: dict[str, Camera] | None = None,
    image_size: int = 256,
    max_components: int = 3,
    min_main_fraction: float = 0.92,
) -> EvalResult:
    """Convert a splat with ``config`` and evaluate the resulting mesh.

    Args:
        ply_path: Input Gaussian Splatting ``.ply`` file.
        config: Conversion parameters to test.
        reference_images: Optional cached splat renders; computed if ``None``.
        cameras: Optional cached cameras; computed if ``None``.
        image_size: Render resolution (square) when references are built.
        max_components: Maximum allowed connected components.
        min_main_fraction: Minimum fraction of faces in the largest component.

    Returns:
        :class:`EvalResult` for this configuration.

    Raises:
        ValueError: If conversion or surface extraction fails.
    """
    if reference_images is None or cameras is None:
        reference_images, cameras = render_splat_references(
            ply_path, width=image_size, height=image_size
        )
    mesh = convert_to_mesh(ply_path, config=config, with_colors=True)
    return evaluate_mesh(
        mesh,
        reference_images,
        cameras,
        max_components=max_components,
        min_main_fraction=min_main_fraction,
    )


def save_baseline(path: str, config: ConvertConfig, result: EvalResult) -> None:
    """Persist the current best configuration and score to a JSON file.

    Args:
        path: Output JSON file path.
        config: The winning :class:`ConvertConfig`.
        result: Its evaluation result.

    Returns:
        None. Writes ``path``.
    """
    payload = {
        "config": config.to_dict(),
        "l1_error": result.l1_error,
        "per_view_l1": result.per_view_l1,
        "stats": result.stats,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_baseline(path: str) -> tuple[ConvertConfig, float]:
    """Load a saved baseline configuration.

    Args:
        path: JSON file written by :func:`save_baseline`.

    Returns:
        Tuple of (config, l1_error).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return ConvertConfig.from_dict(data["config"]), float(data["l1_error"])
