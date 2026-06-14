"""Conversion configuration shared by the CLI and the optimizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class ConvertConfig:
    """All tunable knobs for the splat-to-mesh pipeline.

    Attributes:
        resolution: Voxels along the longest bounding-box axis.
        iso: Occupancy threshold as a fraction of the field maximum, in (0, 1).
            Voxels above this density are treated as inside the object before
            the solid shell is extracted.
        sigma_cutoff: Standard deviations of each Gaussian to splat into the grid.
        min_opacity: Drop Gaussians below this opacity (0 disables).
        outlier_std: Floater removal aggressiveness (0 disables).
        smooth: Taubin smoothing iterations (0 disables).
        target_faces: Decimate to this many faces (0 disables).
        min_face_fraction: Island removal threshold for connected components.
        keep_largest_only: If True, keep only the largest connected component.
        robust_bounds: Use percentile bounds instead of min/max for the grid.
        field_blur_sigma: Gaussian blur sigma (in voxels) applied to the density
            field before building the solid occupancy mask (0 disables).
        max_scale_ratio: Drop Gaussians whose largest axis exceeds this multiple
            of the median scale (0 disables).
        morph_close_iters: Morphological closing iterations on the occupancy mask
            before hole filling; bridges narrow gaps between splats (0 disables).
        shell_smooth_sigma: Gaussian blur sigma (in voxels) applied to the binary
            solid volume before marching cubes; removes terracing ripples on the
            extracted shell (0 disables, leaving a terraced surface).
        filter_support: If True, automatically remove flat table/support surface
            splats when detected in the scan.
        filter_haze: If True, remove desaturated, spatially sparse "haze" splats
            (floating cloud-like background noise) before meshing.
        filter_density: If True, remove splats in low summed-overlap-density
            regions (fog/floaters) before meshing, keeping high-overlap surface.
        density_min_ratio: Drop splats whose local opacity-weighted overlap
            density is below this fraction of the median (0 disables).
    """

    resolution: int = 160
    iso: float = 0.04
    sigma_cutoff: float = 3.0
    min_opacity: float = 0.1
    outlier_std: float = 0.0
    smooth: int = 10
    target_faces: int = 0
    min_face_fraction: float = 0.02
    keep_largest_only: bool = False
    robust_bounds: bool = False
    field_blur_sigma: float = 1.3
    max_scale_ratio: float = 0.0
    morph_close_iters: int = 4
    shell_smooth_sigma: float = 1.0
    filter_support: bool = True
    filter_haze: bool = True
    filter_density: bool = True
    density_min_ratio: float = 0.12

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a plain dictionary.

        Returns:
            Dict of field names to values.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConvertConfig":
        """Build a config from a dictionary, ignoring unknown keys.

        Args:
            data: Mapping of field names to values.

        Returns:
            A :class:`ConvertConfig` instance.
        """
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})
