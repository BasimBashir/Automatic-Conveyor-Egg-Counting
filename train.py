"""
YOLO Training Script for Egg Counting
1-class detection: egg

Self-contained to THIS repository: every path it reads or writes is anchored to
the directory containing this file, so `python train.py` works from any working
directory and nothing leaks to (or depends on) machine-specific locations or the
global Ultralytics settings (datasets_dir / weights_dir / runs_dir).

Reads (all inside the repo):
    - dataset:      <repo>/dataset/data.yaml  (+ train/ valid/ [test/] image+label dirs)
    - base weights: <repo>/yolo26s.pt         (pretrained checkpoint that ships in the repo)

Writes (all inside the repo):
    - run outputs:  <repo>/runs/egg_counter/...      (weights, plots, results.csv)
    - validation:   <repo>/runs/egg_counter_val/...
    - deployed model: <repo>/best.pt                 (copied from the run's best.pt)

Dataset setup:
    Unzip the Roboflow export so that <repo>/dataset/ contains:
        data.yaml
        train/images/  train/labels/
        valid/images/  valid/labels/
        test/images/   test/labels/   (optional)

After training, the best weights are copied to <repo>/best.pt automatically, which
is the model the counting app loads (see app/config.py: model_path = "best.pt").
"""

import os
import shutil
from pathlib import Path

from ultralytics import YOLO

# Repository root = directory that contains this script. EVERYTHING resolves from
# here, so the script is independent of the caller's current working directory.
REPO_ROOT = Path(__file__).resolve().parent


def _resolve_base_weights() -> str:
    """Use the pretrained checkpoint that ships in the repo so training relies on
    repo assets and avoids a network download. If it is missing, fall back to the
    bare model name; because we chdir into the repo first, Ultralytics downloads
    it INTO the repo root rather than the caller's working directory."""
    repo_weights = REPO_ROOT / "yolo26s.pt"
    if repo_weights.exists():
        return str(repo_weights)
    return "yolo26s.pt"


def main():
    # Run from the repo root so any incidental Ultralytics downloads (pretrained
    # weights, assets) land inside the repository, not in the caller's CWD.
    os.chdir(REPO_ROOT)

    # Absolute, repo-anchored path so Ultralytics resolves the relative train/val
    # paths inside data.yaml against the repo regardless of working directory.
    data_path = REPO_ROOT / "dataset" / "data.yaml"
    if not data_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found at '{data_path}'.\n"
            "Unzip the Roboflow egg dataset into <repo>/dataset/ first.\n"
            "Expected structure:\n"
            "  dataset/\n"
            "    data.yaml\n"
            "    train/images/  train/labels/\n"
            "    valid/images/  valid/labels/"
        )

    runs_dir = REPO_ROOT / "runs"

    # Load YOLO26s pretrained on COCO (transfer learning), preferring the repo's
    # own checkpoint.
    model = YOLO(_resolve_base_weights())

    model.train(
        data=str(data_path),
        epochs=20,
        imgsz=1280,
        multi_scale=0.3,
        batch=-1,
        patience=3,
        device=0,               # GPU 0; change to "cpu" if no GPU

        # Optimizer
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Augmentation — keep REALISTIC to a fixed camera over a one-way conveyor.
        # The big recall lever is the DATASET (see docs/DATASET.md): train on frames
        # from the ACTUAL deployed 1280x720 stream covering the hard cases — belt
        # stopped, slow, fast/motion-blurred, dense/occluded. Add motion-blur aug in
        # Roboflow (fast-belt blur is a likely cause of missed birds).
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,              # plant lighting drifts — a bit more brightness aug
        degrees=3.0,            # fixed camera; minimal rotation
        translate=0.1,
        scale=0.5,              # pairs with multi_scale for size robustness
        fliplr=0.5,
        flipud=0.0,             # conveyor moves horizontally — no vertical flip
        mosaic=1.0,             # KEEP: helps dense/small-object detection
        mixup=0.1,
        copy_paste=0.1,
        erasing=0.4,            # random erasing → robustness to occlusion in the chain

        # Scheduler / precision
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        cache="disk",
        workers=8,
        seed=42,
        verbose=True,
        plots=True,

        # Output — repo-relative so all artifacts stay inside the repository.
        project=str(runs_dir),
        name="egg_counter",
        exist_ok=True,
    )

    # Validate the trained model, routing its output into the repo's runs dir too
    # (otherwise val falls back to the global Ultralytics runs_dir, outside the repo).
    metrics = model.val(
        data=str(data_path),
        project=str(runs_dir),
        name="egg_counter_val",
        exist_ok=True,
    )
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # Deploy: copy the run's best weights to <repo>/best.pt so the counting app
    # (app/config.py model_path = "best.pt") picks them up. Stays inside the repo.
    best_src = runs_dir / "egg_counter" / "weights" / "best.pt"
    best_dst = REPO_ROOT / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\nBest model saved to:  {best_src}")
        print(f"Deployed to:          {best_dst}")
    else:
        print(f"\nWARNING: expected best weights not found at {best_src}")


if __name__ == "__main__":
    main()
