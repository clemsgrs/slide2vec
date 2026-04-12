"""Checkpoint loading and assembly for vendored MOOZY inference."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .case import CaseAggregator
from .slide import MOOZYSlideEncoder
from .types import MoozyInferenceComponents


_HF_REPO_ID = "AtlasAnalyticsLab/MOOZY"
_HF_CHECKPOINT = "moozy.pt"


def _build_slide_encoder(state: Mapping[str, torch.Tensor], config: Mapping[str, Any]) -> MOOZYSlideEncoder:
    try:
        feat_dim = int(config["feat_dim"])
        d_model = int(config["d_model"])
        n_layers = int(config["n_layers"])
        n_heads = int(config["n_heads"])
        dim_feedforward = int(config["dim_feedforward"])
    except KeyError as exc:
        raise ValueError(f"MOOZY checkpoint missing required slide config key: {exc.args[0]}") from exc

    num_registers = int(config.get("num_registers", 0))
    learnable_alibi = bool(config.get("learnable_alibi", False))
    qk_norm = bool(config.get("qk_norm", False))

    has_layerscale = any(key.endswith(".ls_attn.gamma") or key.endswith(".ls_mlp.gamma") for key in state)
    layerscale_enabled = bool(config.get("layerscale_enabled", has_layerscale))
    layerscale_init = float(config.get("layerscale_init", 0.0)) if layerscale_enabled else 0.0

    slide_encoder = MOOZYSlideEncoder(
        feat_dim=feat_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        num_registers=num_registers,
        dropout=float(config.get("dropout", 0.0)),
        attn_dropout=float(config.get("attn_dropout", 0.0)),
        layer_drop=float(config.get("layer_drop", 0.0)),
        qk_norm=qk_norm,
        layerscale_init=layerscale_init,
        learnable_alibi=learnable_alibi,
    )
    slide_encoder.load_state_dict(state, strict=True)
    return slide_encoder


def _build_case_transformer(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    *,
    d_model: int,
) -> CaseAggregator:
    case_model = CaseAggregator(
        d_model=d_model,
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config.get("dropout", 0.0)),
        layerscale_init=float(config.get("layerscale_init", 0.0)),
        layer_drop=float(config.get("layer_drop", 0.0)),
        qk_norm=bool(config.get("qk_norm", False)),
        num_registers=int(config.get("num_registers", 0)),
    )
    case_model.load_state_dict(state, strict=True)
    return case_model


def load_moozy_inference_components(device: torch.device) -> MoozyInferenceComponents:
    """Download MOOZY checkpoint and build inference modules."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for bundled MOOZY inference support. "
            "Install it with: pip install 'slide2vec[moozy]'"
        ) from exc

    checkpoint_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_HF_CHECKPOINT)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Invalid MOOZY checkpoint payload: expected a dictionary")

    slide_state = payload.get("teacher_slide_encoder")
    slide_cfg = payload.get("slide_encoder_config") or payload.get("meta")
    case_state = payload.get("case_transformer")
    case_cfg = payload.get("case_transformer_config")

    if slide_state is None or slide_cfg is None or case_state is None or case_cfg is None:
        raise ValueError("MOOZY checkpoint payload missing required keys for inference components")

    slide_encoder = _build_slide_encoder(slide_state, slide_cfg)
    case_transformer = _build_case_transformer(case_state, case_cfg, d_model=slide_encoder.d_model)

    slide_encoder = slide_encoder.to(device).eval()
    case_transformer = case_transformer.to(device).eval()
    return MoozyInferenceComponents(slide_encoder=slide_encoder, case_transformer=case_transformer)
