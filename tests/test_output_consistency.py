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
    target_spacing_um=0.5,
    tolerance=0.07,           # override (default: 0.05)
    target_tile_size_px=224,  # override (default: 256)
    overlap=0.0,
    tissue_threshold=0.1,     # override (default: 0.01)
    drop_holes=False,
    use_padding=True,
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
    max_n_holes=8,
    filter_white=False,
    filter_black=False,
    white_threshold=220,
    black_threshold=25,
    fraction_threshold=0.9,
)

# -- tiling.preview --
TILING_PREVIEW = dict(downsample=32)

# -- model --
MODEL_PARAMS = dict(
    level="slide",           # override (default: "tile")
    name="prism",            # override (default: null)
    mode="cls",
    arch=None,
    pretrained_weights=None,
    batch_size=8,            # override (default: 256)
    input_size=224,          # resolved from ${tiling.params.target_tile_size_px}
    patch_size=256,
    token_size=16,
    save_tile_embeddings=False,
    save_latents=False,
)

# -- speed --
SPEED_PARAMS = dict(
    fp16=True,               # override (default: false)
    num_workers=4,           # override (default: 8)
    num_workers_embedding=4, # override (default: 8)
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
    """Running the full pipeline with hardcoded params produces coordinates and
    embeddings that match the ground truth fixtures in test/gt/."""

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
        "save_previews": False,  # override (default: true)
        "seed": 0,
        "tiling": {
            "read_tiles_from": None,
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
            sys.executable, "-m", "slide2vec",
            "--config-file", str(cfg_path),
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
    assert meta["sample_id"] == "test-wsi"
    assert meta["target_spacing_um"] == pytest.approx(0.5)
    assert meta["target_tile_size_px"] == 224

    # 5. Assert embeddings are within tolerance
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
        print(f"OK: embeddings within tolerance; mean cosine similarity={mean_cos:.4f}")
