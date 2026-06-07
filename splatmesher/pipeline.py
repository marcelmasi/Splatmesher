"""Programmatic splat-to-mesh conversion for evaluation and optimization."""

from __future__ import annotations

import numpy as np
import trimesh

from .color import transfer_colors
from .config import ConvertConfig
from .field import build_density_grid
from .gaussian import (
    Gaussians,
    filter_gaussians,
    filter_haze,
    filter_support_surface,
    should_filter_support_surface,
)
from .io_ply import load_gaussians
from .postprocess import mesh_component_stats, postprocess
from .surface import extract_surface


def _filter_extreme_scales(gaussians: Gaussians, max_scale_ratio: float) -> Gaussians:
    """Drop Gaussians with abnormally large axis scales.

    Args:
        gaussians: Input Gaussians in world space.
        max_scale_ratio: Keep Gaussians whose largest axis is at most this
            multiple of the median largest-axis scale; 0 disables filtering.

    Returns:
        Filtered :class:`Gaussians`.
    """
    if max_scale_ratio <= 0.0 or len(gaussians) == 0:
        return gaussians
    max_scales = gaussians.scales.max(axis=1)
    median = float(np.median(max_scales))
    if median <= 0.0:
        return gaussians
    limit = max_scale_ratio * median
    return gaussians.select(max_scales <= limit)


def convert_to_mesh(
    ply_path: str,
    config: ConvertConfig | None = None,
    with_colors: bool = True,
) -> trimesh.Trimesh:
    """Convert a Gaussian Splatting PLY file to a processed mesh.

    Args:
        ply_path: Path to the input ``.ply`` file.
        config: Conversion parameters; defaults are used when ``None``.
        with_colors: If True, attach per-vertex RGBA colors from the splat.

    Returns:
        A post-processed :class:`trimesh.Trimesh`, optionally with vertex colors.

    Raises:
        ValueError: If the density field is empty or surface extraction fails.
    """
    cfg = config or ConvertConfig()
    gaussians = load_gaussians(ply_path)
    gaussians = filter_gaussians(
        gaussians,
        min_opacity=cfg.min_opacity,
        outlier_std_ratio=cfg.outlier_std,
    )
    if cfg.filter_support and should_filter_support_surface(gaussians):
        gaussians = filter_support_surface(gaussians)
    if cfg.filter_haze:
        gaussians = filter_haze(gaussians)
    gaussians = _filter_extreme_scales(gaussians, cfg.max_scale_ratio)

    grid = build_density_grid(
        gaussians,
        resolution=cfg.resolution,
        sigma_cutoff=cfg.sigma_cutoff,
        robust_bounds=cfg.robust_bounds,
    )
    verts, faces = extract_surface(
        grid,
        iso_relative=cfg.iso,
        pre_blur_sigma=cfg.field_blur_sigma,
        close_iters=cfg.morph_close_iters,
        shell_smooth_sigma=cfg.shell_smooth_sigma,
    )
    mesh = postprocess(
        verts,
        faces,
        smooth_iterations=cfg.smooth,
        target_faces=cfg.target_faces,
        min_face_fraction=cfg.min_face_fraction,
        keep_largest_only=cfg.keep_largest_only,
    )
    if with_colors:
        mesh.visual.vertex_colors = transfer_colors(mesh.vertices, gaussians)
    return mesh


def convert_stats(mesh: trimesh.Trimesh) -> dict[str, float | int | bool]:
    """Summarize mesh quality metrics used by the optimizer.

    Args:
        mesh: The mesh to inspect.

    Returns:
        Dict with component count, main-component face fraction, watertight flag,
        and vertex/face counts.
    """
    stats = mesh_component_stats(mesh)
    return {
        "num_components": stats.num_components,
        "main_component_fraction": stats.main_component_fraction,
        "watertight": bool(mesh.is_watertight),
        "num_vertices": len(mesh.vertices),
        "num_faces": len(mesh.faces),
    }
