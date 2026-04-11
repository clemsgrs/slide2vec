"""MOOZY slide and patient encoder implementations."""

import torch

from slide2vec.encoders.base import PatientEncoder, SlideEncoder, preferred_default_device, resolve_requested_output_variant
from slide2vec.encoders.registry import register_encoder


@register_encoder(
    "moozy-slide",
    level="slide",
    tile_encoder="lunit",
    tile_encoder_output_variant="default",
    output_variants={"default": {"encode_dim": 768}},
    default_output_variant="default",
    supported_spacing_um=0.5,
    precision="fp32",
    source="AtlasAnalyticsLab/MOOZY",
)
class MOOZYSlideEncoder(SlideEncoder):
    def __init__(self, *, output_variant: str | None = None):
        from moozy.hf_hub import ensure_checkpoint
        from moozy.models.factory import load_stage2_inference_model

        ckpt_path = ensure_checkpoint()
        full_model = load_stage2_inference_model(ckpt_path, device=torch.device("cpu"))
        self._model = full_model.slide_encoder.eval()
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(output_variant)

    @property
    def encode_dim(self) -> int:
        return 768

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "MOOZYSlideEncoder":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self

    def encode_slide(
        self,
        tile_features: torch.Tensor,
        coordinates: torch.Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> torch.Tensor:
        if coordinates is None or tile_size_lv0 is None:
            raise ValueError("MOOZY slide encoding requires coordinates and tile_size_lv0")
        # MOOZYSlideEncoder expects [B, crop_h, crop_w, feat_dim]; use [1, 1, N, D]
        x = tile_features.unsqueeze(0).unsqueeze(0)
        coords = coordinates.unsqueeze(0).to(torch.float32)
        patch_sizes = torch.tensor([tile_size_lv0], dtype=torch.float32, device=tile_features.device)
        cls, _, _ = self._model(x, coords_xy=coords, patch_sizes=patch_sizes)
        return cls.squeeze(0)


@register_encoder(
    "moozy",
    level="patient",
    tile_encoder="lunit",
    tile_encoder_output_variant="default",
    output_variants={"default": {"encode_dim": 768}},
    default_output_variant="default",
    supported_spacing_um=0.5,
    precision="fp32",
    source="AtlasAnalyticsLab/MOOZY",
)
class MOOZYPatientEncoder(PatientEncoder):
    def __init__(self, *, output_variant: str | None = None):
        from moozy.hf_hub import ensure_checkpoint
        from moozy.models.factory import load_stage2_inference_model

        ckpt_path = ensure_checkpoint()
        full_model = load_stage2_inference_model(ckpt_path, device=torch.device("cpu"))
        self._slide_model = full_model.slide_encoder.eval()
        self._case_transformer = full_model.case_transformer.eval()
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(output_variant)

    @property
    def encode_dim(self) -> int:
        return 768

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "MOOZYPatientEncoder":
        self._device = torch.device(device)
        self._slide_model = self._slide_model.to(self._device)
        self._case_transformer = self._case_transformer.to(self._device)
        return self

    def encode_slide(
        self,
        tile_features: torch.Tensor,
        coordinates: torch.Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> torch.Tensor:
        if coordinates is None or tile_size_lv0 is None:
            raise ValueError("MOOZY patient encoding requires coordinates and tile_size_lv0")
        x = tile_features.unsqueeze(0).unsqueeze(0)
        coords = coordinates.unsqueeze(0).to(torch.float32)
        patch_sizes = torch.tensor([tile_size_lv0], dtype=torch.float32, device=tile_features.device)
        cls, _, _ = self._slide_model(x, coords_xy=coords, patch_sizes=patch_sizes)
        return cls.squeeze(0)

    def encode_patient(self, slide_embeddings: torch.Tensor) -> torch.Tensor:
        return self._case_transformer(slide_embeddings)
