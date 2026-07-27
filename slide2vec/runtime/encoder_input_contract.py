"""Name the two regimes that can own an encoder run's input geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from slide2vec.runtime.dense_encoder_input import DenseEncoderInputPlan
from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

#: Closed vocabulary for who owns the geometry handed to the encoder.
EncoderInputRegime = Literal["declared", "given"]

#: A resolved declared geometry: pooled or dense, both stated over the *effective encoder
#: input* — the geometry of the tensor immediately before ``encode_tiles`` /
#: ``encode_tiles_dense`` — and both validated by the same capability check.
EncoderInputPlan = PooledEncoderInputPlan | DenseEncoderInputPlan


@dataclass(frozen=True, kw_only=True)
class EncoderInputContract:
    """Which side of the model-load seam owns the encoder input geometry.

    ``"declared"`` — the caller states the encoder input it wants and slide2vec honors
    it exactly or raises: the encoder must be variable-input capable and the size must be
    a multiple of the patch geometry (and, for a pooled off-preset request,
    ``allow_non_recommended_settings`` must be set). ``plan`` carries the resolved
    contract — :class:`PooledEncoderInputPlan` for a pooled run,
    :class:`DenseEncoderInputPlan` for a dense one. The two differ only in how the
    effective encoder input is *derived*; what is checked about it is shared.

    ``"given"`` — the caller supplies pixels it never requested (a pre-cropped tile
    dataset, arbitrary and often non-square). The encoder's shipped transform *is* the
    contract; slide2vec records the observed encoder input size (see
    ``batching._record_encoder_input_size``) rather than vetoing it.

    There is deliberately no default and no "unset" state at the model-load seam:
    ``load_model`` takes the contract as a required argument, so "the caller never
    requested this geometry" can never be confused with "the caller forgot". That
    confusion is not hypothetical — a downstream consumer that routed by GPU count
    built the plan on one route and not the other, and the same config silently
    produced two different transforms.
    """

    regime: EncoderInputRegime
    #: Resolved declared geometry; ``None`` in the Given regime, which has none.
    plan: EncoderInputPlan | None

    def __post_init__(self) -> None:
        if self.regime == "declared" and self.plan is None:
            raise ValueError(
                "A declared encoder-input contract requires a resolved encoder-input "
                "plan; use EncoderInputContract.declared_pooled(...) or "
                "EncoderInputContract.declared_dense(...)."
            )
        if self.regime == "given" and self.plan is not None:
            raise ValueError(
                "A given encoder-input contract carries no plan: the caller never "
                "requested the geometry it supplied."
            )

    @classmethod
    def declared_pooled(
        cls,
        encoder_name: str,
        *,
        requested_tile_size_px: int,
        allow_non_recommended_settings: bool,
    ) -> "EncoderInputContract":
        """Resolve the pooled tile geometry the caller asked for, or raise."""
        return cls(
            regime="declared",
            plan=PooledEncoderInputPlan.resolve(
                encoder_name,
                requested_tile_size_px=requested_tile_size_px,
                allow_non_recommended_settings=allow_non_recommended_settings,
            ),
        )

    @classmethod
    def declared_dense(
        cls,
        encoder_name: str,
        *,
        target_size_px: int,
        window_size: int | None,
    ) -> "EncoderInputContract":
        """Resolve the dense ROI geometry the caller asked for, or raise.

        Dense states a supervision geometry rather than an encoder input, so the effective
        encoder input is derived from it (padded tile, or one patch-aligned window) before
        the shared capability check runs.
        """
        return cls(
            regime="declared",
            plan=DenseEncoderInputPlan.resolve(
                encoder_name,
                target_size_px=target_size_px,
                window_size=window_size,
            ),
        )

    @classmethod
    def given(cls) -> "EncoderInputContract":
        """Accept whatever pixels the caller supplies under the shipped recipe."""
        return cls(regime="given", plan=None)

    def get_transform(self, encoder) -> Callable:
        """Return the encoder-owned transform this contract selects."""
        if self.plan is None:
            return encoder.get_transform()
        return self.plan.get_transform(encoder)

    def construction_kwargs_for(self, encoder_name: str) -> dict[str, bool]:
        """Return the constructor settings this contract imposes on *encoder_name*.

        Only the plan's own tile encoder is affected: a slide/patient encoder resolves
        its geometry through the tile encoder it declares as its dependency.
        """
        if self.plan is None or self.plan.tile_encoder_name != encoder_name:
            return {}
        return dict(self.plan.model_construction_kwargs)
