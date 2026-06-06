"""Export a (optionally colored) mesh to OBJ."""

from __future__ import annotations

import numpy as np
import trimesh


def export_obj(
    mesh: trimesh.Trimesh,
    path: str,
    vertex_colors: np.ndarray | None = None,
) -> None:
    """Write a mesh to an OBJ file, embedding per-vertex colors if given.

    Vertex colors are written using the widely supported ``v x y z r g b`` OBJ
    extension (read by e.g. MeshLab and Blender).

    Args:
        mesh: The mesh to export.
        path: Output ``.obj`` file path.
        vertex_colors: Optional array (V, 3) or (V, 4) of uint8 colors to attach
            to the mesh vertices before export. If ``None``, any existing colors
            on the mesh are used.

    Returns:
        None. The mesh is written to ``path``.
    """
    if vertex_colors is not None:
        mesh.visual.vertex_colors = vertex_colors
    mesh.export(path, include_color=True)
