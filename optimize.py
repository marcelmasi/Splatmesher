#!/usr/bin/env python3
"""Continuous optimizer for the Splatmesher algorithm.

Tries parameter mutations and algorithm tweaks, evaluates L1 render error
against the Gaussian Splat reference, and commits improvements to git.

Run until stopped (Ctrl+C):

    python optimize.py
"""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import fields
from typing import Any

from splatmesher.config import ConvertConfig
from splatmesher.evaluate import (
    EvalResult,
    evaluate_config,
    load_baseline,
    render_splat_references,
    save_baseline,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(REPO_ROOT, "optimization", "baseline.json")
LOG_PATH = os.path.join(REPO_ROOT, "optimization", "log.jsonl")

EXAMPLES = [
    os.path.join(REPO_ROOT, "Examples", "FrangipaniCropped.ply"),
    os.path.join(REPO_ROOT, "Examples", "Pear.ply"),
]
EXAMPLES = [p for p in EXAMPLES if os.path.exists(p)]

# Search ranges for random mutations around the current best config.
MUTATION_RANGES: dict[str, tuple[Any, ...]] = {
    "resolution": (96, 128, 160, 192, 224, 256),
    "iso": (0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35),
    "sigma_cutoff": (2.5, 3.0, 3.5, 4.0, 4.5),
    "min_opacity": (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25),
    "outlier_std": (0.0, 1.0, 1.5, 2.0, 2.5, 3.0),
    "smooth": (0, 3, 5, 8, 10, 15, 20),
    "min_face_fraction": (0.01, 0.02, 0.05, 0.10),
    "field_blur_sigma": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    "max_scale_ratio": (0.0, 3.0, 5.0, 8.0, 12.0),
    "morph_close_iters": (0, 1, 2, 3),
    "shell_smooth_sigma": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
}

BOOL_FIELDS = ("keep_largest_only", "robust_bounds")


def log_event(event: dict[str, Any]) -> None:
    """Append one JSON line to the optimization log.

    Args:
        event: Serializable event dict (iteration, score, config, etc.).

    Returns:
        None. Appends to ``LOG_PATH``.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    event = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)


def git_commit(message: str) -> bool:
    """Stage all changes and create a git commit.

    Args:
        message: Commit message summarizing the improvement.

    Returns:
        True if a commit was created, False if there was nothing to commit.
    """
    subprocess.run(["git", "add", "-A"], cwd=REPO_ROOT, check=True)
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if status.returncode == 0:
        return False
    subprocess.run(["git", "commit", "-m", message], cwd=REPO_ROOT, check=True)
    return True


def mutate_config(base: ConvertConfig, n_changes: int = 2) -> ConvertConfig:
    """Randomly mutate a few fields of a configuration.

    Args:
        base: Starting configuration.
        n_changes: How many scalar/bool fields to change.

    Returns:
        A new mutated :class:`ConvertConfig`.
    """
    cfg = copy.deepcopy(base)
    names = [f.name for f in fields(ConvertConfig)]
    chosen = random.sample(names, k=min(n_changes, len(names)))
    for name in chosen:
        if name in MUTATION_RANGES:
            setattr(cfg, name, random.choice(MUTATION_RANGES[name]))
        elif name in BOOL_FIELDS:
            setattr(cfg, name, random.choice([True, False]))
        elif name == "target_faces":
            setattr(cfg, name, random.choice([0, 0, 0, 50000, 80000]))
    return cfg


def grid_neighbor_configs(base: ConvertConfig) -> list[ConvertConfig]:
    """Generate structured single-step neighbors for systematic exploration.

    Args:
        base: Current best configuration.

    Returns:
        List of nearby configs worth trying next.
    """
    neighbors: list[ConvertConfig] = []
    d = base.to_dict()

    def tweak(**kwargs: Any) -> ConvertConfig:
        """Copy base and override given fields."""
        data = dict(d)
        data.update(kwargs)
        return ConvertConfig.from_dict(data)

    # Iso sweep (most impactful for surface placement).
    for iso in (0.12, 0.15, 0.18, 0.22, 0.28):
        if abs(iso - base.iso) > 1e-6:
            neighbors.append(tweak(iso=iso))

    # Resolution steps.
    for res in (128, 160, 192):
        if res != base.resolution:
            neighbors.append(tweak(resolution=res))

    # Integrity-preserving ideas.
    for keep in (True, False):
        if keep != base.keep_largest_only:
            neighbors.append(tweak(keep_largest_only=keep))
    for robust in (True, False):
        if robust != base.robust_bounds:
            neighbors.append(tweak(robust_bounds=robust))
    for blur in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        if blur != base.field_blur_sigma:
            neighbors.append(tweak(field_blur_sigma=blur))
    for close in (0, 1, 2, 3, 4):
        if close != base.morph_close_iters:
            neighbors.append(tweak(morph_close_iters=close))
    for shell in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        if shell != base.shell_smooth_sigma:
            neighbors.append(tweak(shell_smooth_sigma=shell))

    return neighbors


def score_config(
    ply_path: str,
    config: ConvertConfig,
    refs: dict[str, Any],
    cams: dict[str, Any],
) -> tuple[float | None, dict[str, Any] | None]:
    """Evaluate one config; return None score on failure or integrity violation.

    Args:
        ply_path: Example splat file.
        config: Parameters to test.
        refs: Cached reference splat images for this example.
        cams: Cached cameras for this example.

    Returns:
        Tuple ``(l1_error, info_dict)`` or ``(None, None)`` if rejected.
    """
    try:
        result = evaluate_config(
            ply_path,
            config,
            reference_images=refs,
            cameras=cams,
            image_size=256,
        )
    except Exception as exc:
        log_event(
            {
                "action": "reject",
                "reason": "exception",
                "error": str(exc),
                "config": config.to_dict(),
            }
        )
        return None, None

    if not result.passed_integrity:
        log_event(
            {
                "action": "reject",
                "reason": "integrity",
                "stats": result.stats,
                "config": config.to_dict(),
            }
        )
        return None, None

    info = {
        "l1_error": result.l1_error,
        "per_view_l1": result.per_view_l1,
        "stats": result.stats,
        "config": config.to_dict(),
    }
    return result.l1_error, info


def average_score(
    config: ConvertConfig,
    example_data: list[tuple[str, dict, dict]],
) -> tuple[float | None, list[dict[str, Any]]]:
    """Evaluate a config on all examples and return the mean L1 error.

    Args:
        config: Configuration to score.
        example_data: List of (ply_path, refs, cams) tuples.

    Returns:
        Mean L1 across examples, or ``None`` if any example fails integrity.
    """
    details: list[dict[str, Any]] = []
    scores: list[float] = []
    for ply_path, refs, cams in example_data:
        score, info = score_config(ply_path, config, refs, cams)
        if score is None or info is None:
            return None, details
        info["example"] = os.path.basename(ply_path)
        details.append(info)
        scores.append(score)
    return float(np_mean(scores)), details


def np_mean(xs: list[float]) -> float:
    """Compute the arithmetic mean of a list of floats.

    Args:
        xs: Non-empty list of scores.

    Returns:
        The average value.
    """
    return sum(xs) / len(xs)


def establish_baseline(
    example_data: list[tuple[str, dict, dict]],
) -> tuple[ConvertConfig, float]:
    """Create or load the starting baseline configuration.

    Args:
        example_data: Cached example render data.

    Returns:
        Tuple of (baseline config, baseline mean L1 error).
    """
    if os.path.exists(BASELINE_PATH):
        cfg, score = load_baseline(BASELINE_PATH)
        log_event({"action": "loaded_baseline", "l1_error": score, "config": cfg.to_dict()})
        return cfg, score

    cfg = ConvertConfig()
    score, details = average_score(cfg, example_data)
    if score is None:
        # Fall back to keep_largest_only if default fragments.
        cfg = ConvertConfig(keep_largest_only=True, robust_bounds=True)
        score, details = average_score(cfg, example_data)
    assert score is not None
    save_baseline(
        BASELINE_PATH,
        cfg,
        EvalResult(
            l1_error=score,
            per_view_l1=details[0]["per_view_l1"],
            stats=details[0]["stats"],
            passed_integrity=True,
        ),
    )
    log_event({"action": "established_baseline", "l1_error": score, "config": cfg.to_dict()})
    return cfg, score


def try_candidate(
    label: str,
    config: ConvertConfig,
    best_cfg: ConvertConfig,
    best_score: float,
    example_data: list[tuple[str, dict, dict]],
) -> tuple[ConvertConfig, float, bool]:
    """Evaluate a candidate config and commit if it improves the baseline.

    Args:
        label: Human-readable description of what is being tried.
        config: Candidate configuration.
        best_cfg: Current best configuration.
        best_score: Current best mean L1 error.
        example_data: Cached example data.

    Returns:
        Updated (best_cfg, best_score, improved_flag).
    """
    score, details = average_score(config, example_data)
    if score is None:
        return best_cfg, best_score, False

    log_event(
        {
            "action": "try",
            "label": label,
            "l1_error": score,
            "best": best_score,
            "delta": score - best_score,
            "config": config.to_dict(),
            "details": details,
        }
    )

    if score < best_score - 1e-5:
        save_baseline(
            BASELINE_PATH,
            config,
            EvalResult(
                l1_error=score,
                per_view_l1=details[0]["per_view_l1"],
                stats=details[0]["stats"],
                passed_integrity=True,
            ),
        )
        msg = (
            f"Improve L1 {best_score:.5f} -> {score:.5f}: {label}\n\n"
            f"Config: {json.dumps(config.to_dict(), indent=2)}"
        )
        git_commit(msg)
        log_event({"action": "accept", "label": label, "l1_error": score})
        return config, score, True

    return best_cfg, best_score, False


def optimization_loop() -> None:
    """Run the continuous optimization loop until interrupted.

    Returns:
        None. Runs until ``KeyboardInterrupt``.
    """
    if not EXAMPLES:
        print("No example PLY files found in Examples/", file=sys.stderr)
        sys.exit(1)

    print("Caching reference splat renders...", flush=True)
    example_data: list[tuple[str, dict, dict]] = []
    for path in EXAMPLES:
        refs, cams = render_splat_references(path, width=256, height=256)
        example_data.append((path, refs, cams))
        print(f"  {os.path.basename(path)}", flush=True)

    best_cfg, best_score = establish_baseline(example_data)
    print(f"Baseline L1 = {best_score:.5f}", flush=True)

    iteration = 0
    neighbor_queue = grid_neighbor_configs(best_cfg)
    random.seed()

    while True:
        iteration += 1
        try:
            # Alternate structured neighbors and random mutations.
            if neighbor_queue:
                candidate = neighbor_queue.pop(0)
                label = f"neighbor iter {iteration}"
            else:
                candidate = mutate_config(best_cfg, n_changes=random.randint(1, 3))
                label = f"random iter {iteration}"
                if iteration % 5 == 0:
                    neighbor_queue = grid_neighbor_configs(best_cfg)

            best_cfg, best_score, improved = try_candidate(
                label, candidate, best_cfg, best_score, example_data
            )
            if improved:
                neighbor_queue = grid_neighbor_configs(best_cfg)
                print(f"*** New best L1 = {best_score:.5f} ***", flush=True)

        except KeyboardInterrupt:
            print("\nStopped by user.", flush=True)
            break
        except Exception:
            traceback.print_exc()
            time.sleep(1.0)


if __name__ == "__main__":
    optimization_loop()
