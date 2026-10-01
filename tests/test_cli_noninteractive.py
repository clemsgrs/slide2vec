"""The CLI runs without a terminal and without ``HF_TOKEN``.

Ways this can fail:

- the CLI prompts for a Hugging Face token when ``HF_TOKEN`` is unset. Without a TTY the
  prompt raises ``EOFError``, including for ``--tiling-only`` runs, which load no model,
  and for runs that rely on a cached Hub login.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

REPO_ROOT = Path(__file__).parent.parent


def test_tiling_only_cli_run_needs_no_token_and_no_terminal(tmp_path):
    image_path = tmp_path / "slide.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (448, 448), (160, 70, 110)).save(image_path)
    Image.fromarray(np.ones((112, 112), dtype=np.uint8)).save(mask_path)
    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,mask_path,spacing_at_level_0\n"
        f"slide,{image_path},{mask_path},0.5\n"
    )
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "csv": str(manifest),
                "output_dir": str(tmp_path / "out"),
                "model": {"name": "virchow2"},
                "tiling": {
                    "params": {"requested_spacing_um": 0.5, "requested_tile_size_px": 224},
                    "seg_params": {"downsample": 1},
                    "filter_params": {"a_t": 0, "a_h": 0},
                    "preview": {"save_mask_preview": False, "save_tiling_preview": False},
                },
                "speed": {"num_preprocessing_workers": 1},
            }
        ),
        config_path,
    )
    env = {key: value for key, value in os.environ.items() if key != "HF_TOKEN"}
    env["HF_HUB_OFFLINE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "slide2vec",
            str(config_path),
            "--tiling-only",
            "--skip-datetime",
            "--run-on-cpu",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,  # no terminal: a prompt raises EOFError
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    rows = pd.read_csv(tmp_path / "out" / "process_list.csv")
    assert rows["tiling_status"].tolist() == ["success"]
    assert rows["num_tiles"].tolist() == [4]
