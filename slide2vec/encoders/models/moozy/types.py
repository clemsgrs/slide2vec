"""Shared types for vendored MOOZY inference components."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class MoozyInferenceComponents:
    slide_encoder: nn.Module
    case_transformer: nn.Module
