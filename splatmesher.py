#!/usr/bin/env python3
"""Command-line entry point for Splatmesher.

This thin wrapper lets you run ``python splatmesher.py input.ply output.obj``.
The actual implementation lives in :mod:`splatmesher.cli`.
"""

from splatmesher.cli import main

if __name__ == "__main__":
    main()
