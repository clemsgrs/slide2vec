"""Name the two regimes that can own an encoder run's input geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

#: Closed vocabulary for who owns the geometry handed to the encoder.
EncoderInputRegime = Literal["declared", "given"]


@dataclass(frozen=True, kw_only=True)
class EncoderInputContract:
    """Which side of the model-load seam owns the encoder input geometry.

    ``"declared"`` — the caller states the encoder input it wants and slide2vec honors
    it exactly or raises: an off-preset request needs
    ``allow_non_recommended_settings``, the encoder must be variable-input capable, and
    the size must be a multiple of the patch geometry. ``plan`` carries the resolved
    contract (see :class:`PooledEncoderInputPlan`).

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
    plan: PooledEncoderInputPlan | None

    def __post_init__(self) -> None:
        if self.regime == "declared" and self.plan is None:
            raise ValueError(
                "A declared encoder-input contract requires a resolved "
                "PooledEncoderInputPlan; use EncoderInputContract.declared(...)."
            )
        if self.regime == "given" and self.plan is not None:
            raise ValueError(
                "A given encoder-input contract carries no plan: the caller never "
                "requested the geometry it supplied."
            )

    @classmethod
    def declared(
        cls,
        encoder_name: str,
        *,
        requested_tile_size_px: int,
        allow_non_recommended_settings: bool,
    ) -> "EncoderInputContract":
        """Resolve the geometry the caller asked for, or raise."""
        return cls(
            regime="declared",
            plan=PooledEncoderInputPlan.resolve(
                encoder_name,
                requested_tile_size_px=requested_tile_size_px,
                allow_non_recommended_settings=allow_non_recommended_settings,
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
