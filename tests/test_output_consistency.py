import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf

# ---------------------------------------------------------------------------
# Hardcoded pipeline parameters
# ---------------------------------------------------------------------------

# -- tiling.params --
TILING_PARAMS = dict(
    requested_spacing_um=0.5,
    tolerance=0.07,           # override (default: 0.05)
    requested_tile_size_px=224,  # override (default: 256)
    overlap=0.0,
    tissue_threshold=0.1,     # override (default: 0.01)
)

# -- tiling.seg_params --
TILING_SEG_PARAMS = dict(
    downsample=64,          # override (default: 16)
    sthresh=8,
    sthresh_up=255,
    mthresh=7,
    close=4,
    use_otsu=False,
    use_hsv=True,
)

# -- tiling.filter_params --
TILING_FILTER_PARAMS = dict(
    ref_tile_size=224,  # override (default: 16)
    a_t=4,
    a_h=2,
    filter_white=False,
    filter_black=False,
    white_threshold=220,
    black_threshold=25,
    fraction_threshold=0.9,
)

# -- tiling.preview --
TILING_PREVIEW = dict(save=False, downsample=32)

# -- model --
MODEL_PARAMS = dict(
    name="prism",            # override (default: null)
    batch_size=8,            # override (default: 256)
    save_tile_embeddings=True,
    save_slide_embeddings=False,
    save_latents=False,
)

# -- speed --
SPEED_PARAMS = dict(
    precision="fp16",       # override (default: fp32)
    num_dataloader_workers=0,  # keep the Prism subprocess path single-process to avoid worker SHM pressure
)

# ---------------------------------------------------------------------------
# Paths relative to this test file
# ---------------------------------------------------------------------------
TEST_DIR = Path(__file__).parent
INPUT_DIR = TEST_DIR / "fixtures" / "input"
GT_DIR = TEST_DIR / "fixtures" / "gt"
REPO_ROOT = TEST_DIR.parent


@pytest.fixture(scope="module")
def wsi_path() -> Path:
    p = INPUT_DIR / "test-wsi.tif"
    if not p.is_file():
        pytest.skip(f"Test fixture missing: {p}")
    return p


@pytest.fixture(scope="module")
def mask_path() -> Path:
    p = INPUT_DIR / "test-mask.tif"
    if not p.is_file():
        pytest.skip(f"Test fixture missing: {p}")
    return p


@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN required for model weight download",
)
def test_output_consistency(wsi_path, mask_path, tmp_path):
    """Running the full pipeline with hardcoded params produces x/y coordinates and
    embeddings that match the ground truth fixtures in test/gt/."""

    pytest.importorskip("transformers")
    pytest.importorskip("wholeslidedata")

    # 1. Build a temporary CSV with resolved absolute paths
    tmp_csv = tmp_path / "test.csv"
    tmp_csv.write_text(
        f"sample_id,image_path,mask_path\ntest-wsi,{wsi_path},{mask_path}\n"
    )

    # 2. Build config from hardcoded constants (no dependency on test/input/config.yaml)
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
            "preview": TILING_PREVIEW,
        },
        "model": MODEL_PARAMS,
        "speed": SPEED_PARAMS,
        "wandb": {"enable": False},
    })
    cfg_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, cfg_path)

    # 3. Run the pipeline
    subprocess.run(
        [
            "slide2vec",
            str(cfg_path),
            "--skip-datetime",
            "--run-on-cpu",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    # 4. Assert coordinates match exactly (tiling is deterministic)
    gt_coords = np.load(GT_DIR / "test-wsi.coordinates.npz", allow_pickle=False)
    coords = np.load(tmp_path / "tiles" / "test-wsi.coordinates.npz", allow_pickle=False)
    np.testing.assert_array_equal(coords, gt_coords)

    meta = json.loads((tmp_path / "tiles" / "test-wsi.coordinates.meta.json").read_text())
    assert meta["provenance"]["sample_id"] == "test-wsi"
    assert meta["provenance"]["backend"] == "asap"
    assert meta["tiling"]["requested_spacing_um"] == pytest.approx(0.5)
    assert meta["tiling"]["requested_tile_size_px"] == 224

    # 5. Assert slide embeddings are within tolerance
    gt_emb = torch.load(GT_DIR / "test-wsi.pt", map_location="cpu", weights_only=True)
    emb = torch.load(tmp_path / "slide_embeddings" / "test-wsi.pt", map_location="cpu", weights_only=True)
    assert emb.shape == gt_emb.shape, f"Shape mismatch: {emb.shape} vs {gt_emb.shape}"

    cos = torch.nn.functional.cosine_similarity(emb, gt_emb, dim=-1)
    mean_cos = float(cos.mean())
    atol, rtol = 1e-2, 1e-3
    if not torch.allclose(emb, gt_emb, atol=atol, rtol=rtol):
        assert mean_cos >= 0.99, (
            f"Embedding mismatch: mean cosine similarity={mean_cos:.4f} "
            f"(atol={atol}, rtol={rtol})"
        )
    else:
        print(f"OK: slide embeddings within tolerance; mean cosine similarity={mean_cos:.4f}")

    # 6. Assert tile-level embeddings match ground truth (verifies tile ordering)
    gt_tile_emb = torch.load(GT_DIR / "test-wsi.tiles.pt", map_location="cpu", weights_only=True)
    tile_emb = torch.load(tmp_path / "tile_embeddings" / "test-wsi.pt", map_location="cpu", weights_only=True)
    assert tile_emb.shape == gt_tile_emb.shape, (
        f"Tile embedding shape mismatch: {tile_emb.shape} vs {gt_tile_emb.shape}"
    )
    tile_cos = torch.nn.functional.cosine_similarity(tile_emb, gt_tile_emb, dim=-1)
    mean_tile_cos = float(tile_cos.mean())
    atol, rtol = 1e-2, 1e-3
    if not torch.allclose(tile_emb, gt_tile_emb, atol=atol, rtol=rtol):
        assert mean_tile_cos >= 0.99, (
            f"Tile embedding mismatch: mean cosine similarity={mean_tile_cos:.4f} "
            f"(atol={atol}, rtol={rtol})"
        )
    else:
        print(f"OK: tile embeddings within tolerance; mean cosine similarity={mean_tile_cos:.4f}")
