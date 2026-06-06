"""Splatmesher CLI implementation: convert a Gaussian Splatting PLY to a mesh.

Usage:
    python splatmesher.py myscan.ply output.obj [options]

The pipeline is:
    1. load + clean the Gaussians,
    2. sample the Gaussian density field on a voxel grid,
    3. extract an iso-surface with marching cubes,
    4. post-process the mesh (islands, smoothing, decimation, repair),
    5. transfer colors onto the vertices,
    6. export to OBJ.
"""

from __future__ import annotations

import argparse
import time

from splatmesher.color import transfer_colors
from splatmesher.export import export_obj
from splatmesher.field import build_density_grid
from splatmesher.gaussian import (
    filter_gaussians,
    filter_support_surface,
    should_filter_support_surface,
)
from splatmesher.io_ply import load_gaussians
from splatmesher.postprocess import postprocess
from splatmesher.surface import extract_surface


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Splatmesher CLI.

    Args:
        argv: Optional explicit argument list (defaults to ``sys.argv``).

    Returns:
        Parsed arguments namespace with input/output paths and tuning knobs.
    """
    p = argparse.ArgumentParser(description="Convert a Gaussian Splat to a mesh.")
    p.add_argument("input", help="Input Gaussian Splatting .ply file")
    p.add_argument("output", help="Output mesh .obj file")
    p.add_argument(
        "--resolution",
        type=int,
        default=160,
        help="Voxels along the longest axis (mesh detail vs. time/memory).",
    )
    p.add_argument(
        "--iso",
        type=float,
        default=0.10,
        help="Iso-level as a fraction of the field maximum, in (0, 1).",
    )
    p.add_argument(
        "--sigma-cutoff",
        type=float,
        default=3.0,
        help="Standard deviations of each Gaussian to splat into the grid.",
    )
    p.add_argument(
        "--min-opacity",
        type=float,
        default=0.1,
        help="Drop Gaussians below this opacity (0 disables).",
    )
    p.add_argument(
        "--outlier-std",
        type=float,
        default=0.0,
        help="Floater removal aggressiveness (0 disables; 2+ can break some scans).",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="Taubin smoothing iterations (0 disables).",
    )
    p.add_argument(
        "--target-faces",
        type=int,
        default=0,
        help="Decimate to this many faces (0 disables).",
    )
    p.add_argument(
        "--no-support-filter",
        action="store_true",
        help="Skip automatic removal of flat table/support surface splats.",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Skip per-vertex color transfer.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return p.parse_args(argv)


def convert(args: argparse.Namespace) -> None:
    """Run the full splat-to-mesh conversion described by ``args``.

    Args:
        args: Parsed CLI arguments (see :func:`parse_args`).

    Returns:
        None. Writes the resulting mesh to ``args.output``.
    """

    def log(msg: str) -> None:
        """Print a timestamped progress message unless running quietly.

        Args:
            msg: The message to print.

        Returns:
            None.
        """
        if not args.quiet:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Loading {args.input}")
    gaussians = load_gaussians(args.input)
    log(f"  {len(gaussians)} Gaussians loaded")

    gaussians = filter_gaussians(
        gaussians,
        min_opacity=args.min_opacity,
        outlier_std_ratio=args.outlier_std,
    )
    if not args.no_support_filter and should_filter_support_surface(gaussians):
        before = len(gaussians)
        gaussians = filter_support_surface(gaussians)
        log(f"  removed {before - len(gaussians)} support-surface Gaussians")
    log(f"  {len(gaussians)} Gaussians after cleanup")

    log(f"Building density grid (resolution={args.resolution})")
    grid = build_density_grid(
        gaussians,
        resolution=args.resolution,
        sigma_cutoff=args.sigma_cutoff,
    )
    log(f"  grid {grid.values.shape}, voxel size {grid.voxel_size:.5f}")

    log(f"Extracting surface (iso={args.iso})")
    verts, faces = extract_surface(grid, iso_relative=args.iso)
    log(f"  raw mesh: {len(verts)} verts, {len(faces)} faces")

    log("Post-processing mesh")
    mesh = postprocess(
        verts,
        faces,
        smooth_iterations=args.smooth,
        target_faces=args.target_faces,
    )
    log(f"  final mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    colors = None
    if not args.no_color:
        log("Transferring colors")
        colors = transfer_colors(mesh.vertices, gaussians)

    log(f"Writing {args.output}")
    export_obj(mesh, args.output, vertex_colors=colors)
    log("Done.")


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse arguments and run the conversion.

    Args:
        argv: Optional explicit argument list (defaults to ``sys.argv``).

    Returns:
        None.
    """
    convert(parse_args(argv))


if __name__ == "__main__":
    main()
