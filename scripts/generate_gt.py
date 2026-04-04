#!/usr/bin/env python
"""Regenerate ground-truth fixtures used by test_output_consistency.py.

Requires:
  - HF_TOKEN env var set (for PRISM weight download)
  - test-wsi.tif and test-mask.tif present in tests/fixtures/input/

Usage:
  HF_TOKEN=<token> python scripts/generate_gt.py
  HF_TOKEN=<token> python scripts/generate_gt.py --output-dir tests/fixtures/gt
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "tests"
INPUT_DIR = TEST_DIR / "fixtures" / "input"
DEFAULT_GT_DIR = TEST_DIR / "fixtures" / "gt"

# Must stay in sync with test_output_consistency.py
TILING_PARAMS = dict(
    target_spacing_um=0.5,
    tolerance=0.07,
    target_tile_size_px=224,
    overlap=0.0,
    tissue_threshold=0.1,
    drop_holes=False,
    use_padding=True,
)
TILING_SEG_PARAMS = dict(
    downsample=64,
    sthresh=8,
    sthresh_up=255,
    mthresh=7,
    close=4,
    use_otsu=False,
    use_hsv=True,
)
TILING_FILTER_PARAMS = dict(
    ref_tile_size=224,
    a_t=4,
    a_h=2,
    max_n_holes=8,
    filter_white=False,
    filter_black=False,
    white_threshold=220,
    black_threshold=25,
    fraction_threshold=0.9,
)
MODEL_PARAMS = dict(
    level="slide",
    name="prism",
    mode="cls",
    arch=None,
    pretrained_weights=None,
    batch_size=8,
    input_size=224,
    patch_size=256,
    token_size=16,
    save_tile_embeddings=True,
    save_latents=False,
)
SPEED_PARAMS = dict(
    precision="fp16",
    num_workers=4,
    num_workers_embedding=4,
)


def main():
    parser = argparse.ArgumentParser(description="Regenerate slide2vec GT fixtures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GT_DIR,
        help=f"Directory to write GT fixtures into (default: {DEFAULT_GT_DIR})",
    )
    args = parser.parse_args()

    wsi_path = INPUT_DIR / "test-wsi.tif"
    mask_path = INPUT_DIR / "test-mask.tif"
    for p in (wsi_path, mask_path):
        if not p.is_file():
            sys.exit(f"Missing fixture: {p}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tmp_csv = tmp_path / "test.csv"
        tmp_csv.write_text(
            f"sample_id,image_path,mask_path\ntest-wsi,{wsi_path},{mask_path}\n"
        )
        cfg = OmegaConf.create({
            "csv": str(tmp_csv),
            "output_dir": str(tmp_path),
            "resume": False,
            "resume_dirname": None,
            "seed": 0,
            "tiling": {
                "read_coordinates_from": None,
                "read_tiles_from": None,
                "on_the_fly": True,
                "backend": "asap",
                "params": TILING_PARAMS,
                "seg_params": TILING_SEG_PARAMS,
                "filter_params": TILING_FILTER_PARAMS,
                "preview": {"save": False, "downsample": 32},
            },
            "model": MODEL_PARAMS,
            "speed": SPEED_PARAMS,
            "wandb": {"enable": False},
        })
        cfg_path = tmp_path / "config.yaml"
        OmegaConf.save(cfg, cfg_path)

        print("Running pipeline...")
        subprocess.run(
            [
                sys.executable, "-m", "slide2vec",
                "--config-file", str(cfg_path),
                "--skip-datetime",
                "--run-on-cpu",
            ],
            cwd=REPO_ROOT,
            check=True,
        )

        # Copy slide embedding
        slide_emb_src = tmp_path / "slide_embeddings" / "test-wsi.pt"
        shutil.copyfile(slide_emb_src, output_dir / "test-wsi.pt")
        print(f"Saved slide embedding: {output_dir / 'test-wsi.pt'}")

        # Copy tile embeddings
        tile_emb_src = tmp_path / "tile_embeddings" / "test-wsi.pt"
        tile_emb = torch.load(tile_emb_src, map_location="cpu", weights_only=True)
        torch.save(tile_emb, output_dir / "test-wsi.tiles.pt")
        print(f"Saved tile embeddings: {output_dir / 'test-wsi.tiles.pt'} — shape {tuple(tile_emb.shape)}")

        # Copy coordinates
        import numpy as np
        coords_src = tmp_path / "tiles" / "test-wsi.coordinates.npz"
        shutil.copyfile(coords_src, output_dir / "test-wsi.coordinates.npz")
        print(f"Saved coordinates: {output_dir / 'test-wsi.coordinates.npz'}")

        meta_src = tmp_path / "tiles" / "test-wsi.coordinates.meta.json"
        shutil.copyfile(meta_src, output_dir / "test-wsi.coordinates.meta.json")
        print(f"Saved coordinates meta: {output_dir / 'test-wsi.coordinates.meta.json'}")

    print("\nDone. GT fixtures updated.")


if __name__ == "__main__":
    main()
