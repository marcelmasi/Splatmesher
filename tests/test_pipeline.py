"""End-to-end test of the splat-to-mesh conversion pipeline."""

from __future__ import annotations

import os

import pytest
import trimesh

from splatmesher import cli

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_PLY = os.path.join(REPO_ROOT, "Examples", "FrangipaniCropped.ply")


@pytest.mark.skipif(not os.path.exists(EXAMPLE_PLY), reason="example PLY missing")
def test_convert_produces_valid_obj(tmp_path) -> None:
    """Convert an example splat at low resolution and validate the OBJ output."""
    out_path = str(tmp_path / "out.obj")
    args = cli.parse_args(
        [EXAMPLE_PLY, out_path, "--resolution", "80", "--quiet"]
    )
    cli.convert(args)

    assert os.path.exists(out_path)
    mesh = trimesh.load(out_path, process=False)
    assert len(mesh.vertices) > 100
    assert len(mesh.faces) > 100
    # The mesh should be a single closed shell suitable for printing.
    assert mesh.is_watertight
