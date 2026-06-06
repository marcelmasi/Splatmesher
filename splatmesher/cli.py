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

from splatmesher.config import ConvertConfig
from splatmesher.export import export_obj
from splatmesher.pipeline import convert_to_mesh


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
        default=0.04,
        help="Solid-shell density threshold as a fraction of the field max.",
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
        "--min-face-fraction",
        type=float,
        default=0.02,
        help="Discard connected components smaller than this face fraction.",
    )
    p.add_argument(
        "--keep-largest-only",
        action="store_true",
        help="Keep only the largest connected mesh component.",
    )
    p.add_argument(
        "--robust-bounds",
        action="store_true",
        help="Use percentile bounds for the voxel grid (ignores floaters).",
    )
    p.add_argument(
        "--field-blur",
        type=float,
        default=1.5,
        help="Blur sigma (voxels) on the density field before solid extraction.",
    )
    p.add_argument(
        "--max-scale-ratio",
        type=float,
        default=0.0,
        help="Drop Gaussians with axis scale above median * ratio (0 disables).",
    )
    p.add_argument(
        "--morph-close",
        type=int,
        default=2,
        help="Closing iterations on the occupancy mask before hole fill.",
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


def args_to_config(args: argparse.Namespace) -> ConvertConfig:
    """Build a :class:`ConvertConfig` from parsed CLI arguments.

    Args:
        args: Namespace returned by :func:`parse_args`.

    Returns:
        Equivalent :class:`ConvertConfig`.
    """
    return ConvertConfig(
        resolution=args.resolution,
        iso=args.iso,
        sigma_cutoff=args.sigma_cutoff,
        min_opacity=args.min_opacity,
        outlier_std=args.outlier_std,
        smooth=args.smooth,
        target_faces=args.target_faces,
        min_face_fraction=args.min_face_fraction,
        keep_largest_only=args.keep_largest_only,
        robust_bounds=args.robust_bounds,
        field_blur_sigma=args.field_blur,
        max_scale_ratio=args.max_scale_ratio,
        morph_close_iters=args.morph_close,
        filter_support=not args.no_support_filter,
    )


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

    cfg = args_to_config(args)
    log(f"Converting {args.input}")
    mesh = convert_to_mesh(args.input, config=cfg, with_colors=not args.no_color)
    log(f"  mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    log(f"Writing {args.output}")
    export_obj(mesh, args.output)
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
