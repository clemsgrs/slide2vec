"""Real-weight forward-shape contracts for GPFM and GenBio-PathFM.

These load multi-GB foundation-model weights on CPU (minutes each), so they are
marked ``heavy`` and excluded from the PR suite (``-m 'not heavy'``); they run on
the scheduled/manual heavy workflow. Each skips cleanly when weights / optional
deps / network are unavailable so a developer run never hard-fails.

They pin the two registration choices that can only be confirmed against the real
checkpoints:

* GPFM: ``GPFM.pth`` loads into the timm DINOv2 ViT-L/14 (``strict=True``) and the
  pooled forward returns the registered 1024-d embedding.
* GenBio-PathFM: the single-channel backbone's canonical forward returns the
  registered 4608-d (= embed_dim*3) per-channel-CLS concatenation.
"""

from __future__ import annotations

import pytest
import torch

from slide2vec.encoders import encoder_registry


@pytest.mark.heavy
def test_gpfm_loads_strict_and_forward_is_1024_d():
    try:
        encoder = encoder_registry.require("gpfm")(output_variant="default")
    except Exception as exc:  # network / weights / optional dep unavailable
        pytest.skip(f"gpfm weights unavailable: {type(exc).__name__}: {exc}")
    encoder = encoder.to("cpu")
    assert encoder.encode_dim == 1024
    assert encoder.patch_size == (14, 14)
    batch = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        out = encoder.encode_tiles(batch)
    assert out.shape == (1, 1024)


@pytest.mark.heavy
def test_genbio_pathfm_forward_is_4608_d():
    try:
        encoder = encoder_registry.require("genbio-pathfm")(output_variant="default")
    except Exception as exc:  # network / weights / optional dep unavailable
        pytest.skip(f"genbio-pathfm weights unavailable: {type(exc).__name__}: {exc}")
    encoder = encoder.to("cpu")
    assert encoder.encode_dim == 4608
    batch = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        out = encoder.encode_tiles(batch)
    # Single-channel ViT (in_chans=1): forward concatenates the 3 per-channel CLS
    # tokens (embed_dim 1536) -> 4608.
    assert out.shape == (1, 4608)
