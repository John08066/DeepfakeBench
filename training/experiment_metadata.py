"""Generic reproducibility metadata writer for training and evaluation runs."""

import json
import subprocess
from pathlib import Path


def write_run_metadata(output_dir, config, extra=None):
    """Persist config, Git revision, seed, datasets, and PRD component choices."""
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    metadata = {
        "git_commit": git_commit,
        "random_seed": config.get("manualSeed"),
        "train_datasets": config.get("train_dataset"),
        "test_datasets": config.get("test_dataset"),
        "probe": config.get("probe", {"type": "sd15_vae"}),
        "residual": config.get("residual", {"type": "signed_diff"}),
        "encoder": config.get("encoder", {"type": "clip"}),
        "feature_mode": config.get("feature_mode", "concat"),
        "config": config,
    }
    metadata.update(extra or {})
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, default=str)
