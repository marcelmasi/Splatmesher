"""Mesh clean-up: component selection, smoothing, decimation and repair."""

from __future__ import annotations

import numpy as np
import trimesh


def keep_largest_components(
    mesh: trimesh.Trimesh,
    min_face_fraction: float = 0.02,
) -> trimesh.Trimesh:
    """Drop small disconnected islands, keeping the significant components.

    Args:
        mesh: Input mesh, possibly with many disconnected pieces (e.g. floaters).
        min_face_fraction: Components with fewer than this fraction of the total
            face count are discarded. The largest component is always kept.

    Returns:
        A mesh containing only the retained components (concatenated).
    """
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh

    total_faces = len(mesh.faces)
    threshold = max(1, int(min_face_fraction * total_faces))
    kept = [c for c in components if len(c.faces) >= threshold]
    if not kept:
        kept = [max(components, key=lambda c: len(c.faces))]
    return trimesh.util.concatenate(kept)


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 10) -> trimesh.Trimesh:
    """Apply Taubin smoothing to reduce voxel staircase artefacts.

    Taubin smoothing alternates positive and negative Laplacian steps so the
    mesh is smoothed without the strong shrinkage of plain Laplacian smoothing.

    Args:
        mesh: Input mesh (modified in place by the smoothing filter).
        iterations: Number of Taubin iterations; more is smoother but blurs
            detail.

    Returns:
        The smoothed mesh.
    """
    if iterations <= 0:
        return mesh
    trimesh.smoothing.filter_taubin(mesh, iterations=iterations)
    return mesh


def decimate_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce the triangle count using quadric edge-collapse decimation.

    Args:
        mesh: Input mesh.
        target_faces: Desired number of faces after decimation; if the mesh
            already has fewer faces, it is returned unchanged.

    Returns:
        The decimated mesh (or the original if decimation is unnecessary or the
        backend is unavailable).
    """
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except (ValueError, ImportError, AttributeError):
        return mesh


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Make the mesh as watertight and consistently oriented as possible.

    Args:
        mesh: Input mesh.

    Returns:
        The repaired mesh with degenerate/duplicate faces removed, holes filled
        where possible, and outward-consistent winding/normals.
    """
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes()
    mesh.fix_normals()
    return mesh


def postprocess(
    vertices: np.ndarray,
    faces: np.ndarray,
    smooth_iterations: int = 10,
    target_faces: int = 0,
    min_face_fraction: float = 0.02,
) -> trimesh.Trimesh:
    """Run the full post-processing chain on a raw marching-cubes mesh.

    Args:
        vertices: Array (V, 3) of raw surface vertices in world units.
        faces: Array (F, 3) of triangle vertex indices.
        smooth_iterations: Taubin smoothing iterations (0 disables smoothing).
        target_faces: Target face count for decimation (0 disables decimation).
        min_face_fraction: Island removal threshold (see
            :func:`keep_largest_components`).

    Returns:
        A cleaned, smoothed, optionally decimated and repaired
        :class:`trimesh.Trimesh`.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh = keep_largest_components(mesh, min_face_fraction=min_face_fraction)
    mesh = smooth_mesh(mesh, iterations=smooth_iterations)
    mesh = decimate_mesh(mesh, target_faces=target_faces)
    mesh = repair_mesh(mesh)
    return mesh
