"""Encoder abstractions for tile-level and slide-level feature extraction."""

from abc import ABC, abstractmethod
from typing import Callable

import timm
import torch
from timm.data import create_transform, resolve_data_config
from torch import Tensor
from torchvision.transforms import v2


def preferred_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_requested_output_variant(
    output_variant: str | None,
    *,
    default: str = "default",
    allowed: tuple[str, ...] = ("default",),
) -> str:
    """Normalize and validate a requested encoder output variant."""
    resolved = output_variant or default
    if resolved not in allowed:
        available = ", ".join(allowed)
        raise ValueError(
            f"Unsupported output_variant '{resolved}'. Available: {available}"
        )
    return resolved


def resolve_recommended_dynamic_img_size(
    *,
    requested: bool | None,
    recommended: bool,
    allow_non_recommended: bool,
    encoder_name: str,
) -> bool:
    """Resolve ``dynamic_img_size`` against an encoder's card-recommended value.

    ``None`` uses the recommended value. A value that differs from the
    recommendation requires ``allow_non_recommended_settings=True`` (e.g. dense
    feature extraction deliberately enabling variable input size, justified by
    the registration / native-size-no-op tests); otherwise it raises, so a
    pipeline never silently runs an encoder outside its documented config.
    """
    if requested is None:
        return recommended
    if requested != recommended and not allow_non_recommended:
        raise ValueError(
            f"Encoder '{encoder_name}' recommends dynamic_img_size={recommended} "
            f"(per its model card); got dynamic_img_size={requested}, which deviates "
            "from the recommended setting. Pass allow_non_recommended_settings=True "
            "to override it deliberately (e.g. dense extraction needs variable input "
            "size; this is a native-size no-op, verified in the encoder tests)."
        )
    return requested


def reshape_tokens_to_grid(
    tokens: Tensor,
    *,
    grid_h: int,
    grid_w: int,
    num_prefix_tokens: int,
    encoder_name: str,
) -> Tensor:
    """Fold a ViT ``(B, T, d)`` token sequence into a dense ``(B, d, h, w)`` grid.

    Strips the leading ``num_prefix_tokens`` (CLS + register tokens) and reshapes
    the remaining patch tokens back into their row-major spatial grid. ViT patch
    tokens are emitted in row-major order ``[(0,0), (0,1), ..., (h-1, w-1)]`` after
    the prefix tokens, so ``transpose(1, 2).reshape(B, d, h, w)`` recovers the
    spatial layout. Verified bit-for-bit against timm's
    ``get_intermediate_layers(..., reshape=True)`` in the encoder tests.

    Fails loudly if the post-strip token count does not match ``grid_h * grid_w``:
    a silent reshape would train a decoder on spatially corrupted features, which
    is worse than a hard failure.
    """
    if tokens.ndim != 3:
        raise ValueError(
            f"Dense extraction for '{encoder_name}' expected a (B, T, d) token "
            f"sequence from the backbone, got shape {tuple(tokens.shape)}. This "
            "encoder may not expose a recoverable patch-token grid."
        )
    patch_tokens = tokens[:, num_prefix_tokens:, :]
    batch_size, num_tokens, dim = patch_tokens.shape
    expected = grid_h * grid_w
    if num_tokens != expected:
        raise ValueError(
            f"Dense token accounting mismatch for '{encoder_name}': backbone "
            f"returned {tokens.shape[1]} tokens; after stripping "
            f"{num_prefix_tokens} prefix token(s), {num_tokens} remain, but the "
            f"{grid_h}x{grid_w} grid expects {expected}. Check the prefix-token "
            "count and the input-size / patch-size / grid geometry."
        )
    return patch_tokens.transpose(1, 2).reshape(batch_size, dim, grid_h, grid_w)


def prefix_attention_to_grid(
    attn_weights: Tensor,
    *,
    num_prefix_tokens: int,
    include_registers: bool,
    grid_h: int,
    grid_w: int,
    encoder_name: str,
) -> Tensor:
    """Fold one block's self-attention into per-prefix-token spatial maps.

    Given the full attention weights ``(B, nh, N, N)`` of one transformer block
    (rows = query tokens, columns = key tokens, each row a softmax over keys),
    select the **prefix-token query rows** — the CLS token always, the ``M``
    register tokens too when ``include_registers`` — slice the **patch key columns**,
    and reshape each selected row back into its ``(grid_h, grid_w)`` spatial layout.

    The output is ``(B, K, grid_h, grid_w)`` with ``K = num_query * nh`` channels in
    the deterministic order ``[cls, reg…][head]`` (query-token outer, head inner):
    channel ``q * nh + head`` is prefix-query ``q``'s attention from head ``head``.
    ``num_query = num_prefix_tokens`` when ``include_registers`` else ``1`` (CLS only).

    This is the attention analog of :func:`reshape_tokens_to_grid` (one folds patch
    *tokens* into a grid, the other folds prefix-token *attention* into grids); it
    reuses the same ``num_prefix_tokens`` split. Per-head is preserved on purpose —
    head specialization is the signal the downstream pixel-classifier exploits, and
    reducing it would be lossy and irreversible in the cache. Fails loud on a token
    accounting mismatch rather than silently mis-reshaping.
    """
    if attn_weights.ndim != 4:
        raise ValueError(
            f"Attention extraction for '{encoder_name}' expected (B, nh, N, N) "
            f"attention weights, got shape {tuple(attn_weights.shape)}."
        )
    batch_size, num_heads, num_query_tokens, num_key_tokens = attn_weights.shape
    if num_query_tokens != num_key_tokens:
        raise ValueError(
            f"Attention extraction for '{encoder_name}' expected square attention "
            f"(query==key tokens), got {num_query_tokens}x{num_key_tokens}."
        )
    if num_prefix_tokens < 1:
        raise ValueError(
            f"Attention extraction for '{encoder_name}' needs at least one prefix "
            f"token (the CLS query row), got num_prefix_tokens={num_prefix_tokens}."
        )
    num_patches = grid_h * grid_w
    expected = num_prefix_tokens + num_patches
    if num_key_tokens != expected:
        raise ValueError(
            f"Attention token accounting mismatch for '{encoder_name}': block "
            f"returned {num_key_tokens} tokens; with {num_prefix_tokens} prefix "
            f"token(s) the {grid_h}x{grid_w} grid expects {expected}. Check the "
            "prefix-token count and the input-size / patch-size / grid geometry."
        )
    num_query = num_prefix_tokens if include_registers else 1
    # rows: prefix query tokens [0:num_query]; columns: patch keys [num_prefix:].
    patch_rows = attn_weights[:, :, :num_query, num_prefix_tokens:]  # (B, nh, q, P)
    # (B, nh, q, P) -> (B, q, nh, P) so reshape yields the [query][head] channel order.
    maps = patch_rows.permute(0, 2, 1, 3).reshape(batch_size, num_query * num_heads, num_patches)
    return maps.reshape(batch_size, num_query * num_heads, grid_h, grid_w)


def timm_self_attention_weights(attn_module, x: Tensor) -> Tensor:
    """Recompute a timm ``Attention`` block's softmax weights ``(B, nh, N, N)``.

    timm's attention runs a *fused* SDPA kernel by default, which never
    materializes the attention matrix. To recover it we re-run the projection from
    the module's own input ``x`` (the post-``norm1`` residual-branch input, captured
    via a forward-pre-hook) using the module's own ``qkv`` / ``q_norm`` / ``k_norm``
    / ``num_heads`` / ``head_dim`` / ``scale`` — i.e. exactly the non-fused branch of
    ``Attention.forward``, so the result is bit-equivalent to the weights the fused
    kernel applies internally. Dropout is omitted (extraction runs under ``eval``).
    """
    if not hasattr(attn_module, "qkv"):
        raise NotImplementedError(
            f"{type(attn_module).__name__} has no fused 'qkv' projection; attention "
            "extraction currently supports timm ViT Attention blocks only."
        )
    batch_size, num_tokens, _ = x.shape
    num_heads = int(attn_module.num_heads)
    head_dim = int(getattr(attn_module, "head_dim", x.shape[-1] // num_heads))
    qkv = (
        attn_module.qkv(x)
        .reshape(batch_size, num_tokens, 3, num_heads, head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, _ = qkv.unbind(0)
    # q_norm / k_norm are Identity unless the model uses QK-norm; apply them either way.
    q = attn_module.q_norm(q)
    k = attn_module.k_norm(k)
    q = q * attn_module.scale
    attn = q @ k.transpose(-2, -1)
    return attn.softmax(dim=-1)


def resolve_block_indices(blocks, num_blocks: int, *, encoder_name: str) -> list[int]:
    """Normalize a (possibly negative) block selection against ``num_blocks``.

    Preserves caller order (so the recorded ``[block]`` channel order is the order
    requested) and validates each index, failing loud on an out-of-range block.
    """
    resolved: list[int] = []
    for raw in blocks:
        idx = int(raw)
        if idx < 0:
            idx += num_blocks
        if not (0 <= idx < num_blocks):
            raise ValueError(
                f"Attention extraction for '{encoder_name}': block index {raw} is out "
                f"of range for a backbone with {num_blocks} transformer blocks."
            )
        resolved.append(idx)
    return resolved


def timm_trunk_attention(
    trunk,
    batch: Tensor,
    *,
    blocks: tuple[int, ...] = (-1,),
    include_registers: bool = False,
    encoder_name: str,
) -> Tensor:
    """Extract per-head prefix-token attention maps from a timm ViT trunk.

    The reusable core of :meth:`TimmTileEncoder.encode_tiles_attention`, factored
    out so wrapper encoders that embed a timm ``VisionTransformer`` (CONCH's
    ``visual.trunk``, CONCH v1.5's ``trunk``) reuse the exact same path on their
    inner trunk — the attention analog of how ``_encode_trunk_dense`` is shared.

    Captures each selected block's attention input via a forward-pre-hook on
    ``trunk.blocks[i].attn`` (the fused SDPA kernel never materializes the matrix),
    recomputes the softmax weights (:func:`timm_self_attention_weights`), and folds
    the prefix-token query rows into spatial grids (:func:`prefix_attention_to_grid`).
    Patch size and prefix-token count are read from the trunk
    (``patch_embed.patch_size`` / ``num_prefix_tokens``). Output ``(B, K, h, w)`` in
    ``[block][cls, reg…][head]`` order.
    """
    if batch.ndim != 4:
        raise ValueError(
            "encode_tiles_attention expects a (B, C, H, W) batch, got shape "
            f"{tuple(batch.shape)}."
        )
    _, _, height, width = batch.shape
    patch = trunk.patch_embed.patch_size
    patch_h, patch_w = (patch, patch) if isinstance(patch, int) else (int(patch[0]), int(patch[1]))
    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(
            f"Attention extraction for '{encoder_name}' requires input divisible by "
            f"the patch size: got {height}x{width}, patch {patch_h}x{patch_w}. Pad "
            "the tile up to a patch multiple first."
        )
    if not hasattr(trunk, "blocks"):
        raise NotImplementedError(
            f"{encoder_name} has no '.blocks' transformer stack; attention extraction "
            "supports timm ViT-style backbones only."
        )
    block_list = trunk.blocks
    resolved = resolve_block_indices(blocks, len(block_list), encoder_name=encoder_name)

    captured: dict[int, Tensor] = {}

    def _make_hook(index: int):
        def _hook(_module, inputs):
            captured[index] = inputs[0]

        return _hook

    handles = []
    for index in sorted(set(resolved)):
        handles.append(block_list[index].attn.register_forward_pre_hook(_make_hook(index)))
    try:
        trunk.forward_features(batch)
    finally:
        for handle in handles:
            handle.remove()

    grid_h, grid_w = height // patch_h, width // patch_w
    num_prefix = int(getattr(trunk, "num_prefix_tokens", 1))
    grids = []
    for index in resolved:
        attn_weights = timm_self_attention_weights(block_list[index].attn, captured[index])
        grids.append(
            prefix_attention_to_grid(
                attn_weights,
                num_prefix_tokens=num_prefix,
                include_registers=include_registers,
                grid_h=grid_h,
                grid_w=grid_w,
                encoder_name=encoder_name,
            )
        )
    return torch.cat(grids, dim=1)  # [block] outer (caller order), [cls, reg…][head] inner


def attentions_tuple_to_grids(
    attentions,
    *,
    num_prefix_tokens: int,
    blocks: tuple[int, ...],
    include_registers: bool,
    grid_h: int,
    grid_w: int,
    encoder_name: str,
) -> Tensor:
    """Fold an HF ``output_attentions`` tuple into stacked prefix-token grids.

    HF transformer ViTs expose every block's softmax attention directly (no
    fused-kernel recompute), as a per-layer tuple of ``(B, nh, N, N)`` tensors.
    This selects the requested blocks (:func:`resolve_block_indices`), folds each
    to spatial grids (:func:`prefix_attention_to_grid`), and concatenates them in
    ``[block][cls, reg…][head]`` order — the shared core of the HF-path encoders
    (Phikon, Hibou, Midnight), which differ only in ``num_prefix_tokens``.
    """
    if not attentions:
        raise NotImplementedError(
            f"{encoder_name} returned no attentions; the model must support "
            "output_attentions=True (an eager/recompute attention implementation, "
            "not a fused SDPA path that discards the weights)."
        )
    resolved = resolve_block_indices(blocks, len(attentions), encoder_name=encoder_name)
    grids = [
        prefix_attention_to_grid(
            attentions[index],
            num_prefix_tokens=num_prefix_tokens,
            include_registers=include_registers,
            grid_h=grid_h,
            grid_w=grid_w,
            encoder_name=encoder_name,
        )
        for index in resolved
    ]
    return torch.cat(grids, dim=1)


class Encoder(ABC):
    """Shared lifecycle contract for all encoders."""

    @property
    @abstractmethod
    def encode_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Current device of the encoder."""
        ...

    @abstractmethod
    def to(self, device: torch.device | str) -> "Encoder":
        """Move encoder to the given device. Returns self."""
        ...


class TileEncoder(Encoder):
    """Base class for encoders that operate directly on image tiles."""

    @abstractmethod
    def get_transform(self) -> Callable:
        """Image transform pipeline (PIL Image or ndarray -> Tensor)."""
        ...

    @abstractmethod
    def encode_tiles(self, batch: Tensor) -> Tensor:
        """Encode a batch of tiles. (B, C, H, W) -> (B, D)."""
        ...

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
        """Encode tiles into a dense spatial feature grid. (B, C, H, W) -> (B, d, h, w).

        Default: unsupported. ViT tile encoders with a recoverable patch grid
        override this; vision-language / slide-native encoders (no usable patch
        grid) do not. ``d`` is the per-token feature dim and ``h, w`` the token
        grid (``H / patch``, ``W / patch``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support dense (spatial-grid) feature "
            "extraction. Dense extraction requires a ViT tile encoder whose patch "
            "tokens can be reshaped into a spatial grid."
        )

    def encode_tiles_attention(
        self,
        batch: Tensor,
        *,
        blocks: tuple[int, ...] = (-1,),
        include_registers: bool = False,
    ) -> Tensor:
        """Encode tiles into per-head CLS/register self-attention maps.

        ``(B, C, H, W) -> (B, K, h, w)`` where each channel is one prefix-token
        query row's self-attention over the patch grid for one head, stacked in the
        deterministic order ``[block][cls, reg…][head]``. ``K = len(blocks) *
        (1 + M·include_registers) * nh`` with ``M`` the model's register-token count
        (``0`` for models without them) and ``nh`` the head count.

        The per-head CLS attention of a frozen ViT doubles as a dense per-pixel
        feature (Ramchandani et al., arXiv:2602.18747); ``include_registers`` adds
        the register-token query rows (Darcet et al.) as extra, optional channels.
        Default: unsupported — overridden by ViT tile encoders whose attention
        blocks can be hooked (timm ViTs and HF transformers ViTs).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support attention-map extraction. It "
            "requires a ViT tile encoder whose self-attention can be recovered "
            "(timm Attention blocks, or an HF transformer with output_attentions)."
        )

    @property
    def patch_size(self) -> tuple[int, int]:
        """Backbone patch size ``(patch_h, patch_w)`` — only for dense encoders.

        Encoder-authoritative (a property of the frozen model), used to resolve the
        dense token grid. Default: unsupported, mirroring ``encode_tiles_dense``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a patch size (only dense-capable "
            "ViT tile encoders do)."
        )

    def get_dense_transform(self) -> Callable:
        """Photometric (normalization-only) transform for dense extraction.

        Returns a transform that applies ONLY this encoder's normalization
        (per-channel mean/std) — **no Resize, no CenterCrop** — so the dense
        feature grid covers the *full* source tile and stays spatially registered
        to it. This deliberately differs from ``get_transform`` (the pooled recipe):
        some encoders resize-then-center-crop there (GigaPath ``Resize(256) ->
        CenterCrop(224)``; Lunit ``crop_pct=0.9 -> Resize(248) -> CenterCrop(224)``),
        which drops the tile margins and would misregister the grid against a dense
        target mask. Geometry (padding to a patch multiple, optional resize,
        cropping logits back) is the dense pipeline's responsibility, not the
        encoder's. Default: unsupported, mirroring ``encode_tiles_dense``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a dense transform. Only "
            "encoders that support dense (spatial-grid) extraction define one."
        )


class SlideEncoder(Encoder):
    """Base class for encoders that pool tile features into slide features."""

    tile_encoder: TileEncoder | None = None

    def encode_tiles(self, batch: Tensor) -> Tensor:
        if self.tile_encoder is None:
            raise AttributeError("slide encoders must attach a tile_encoder before encoding tiles")
        return self.tile_encoder.encode_tiles(batch)

    @abstractmethod
    def encode_slide(
        self,
        tile_features: Tensor,
        coordinates: Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> Tensor:
        """Pool tile-level features into a single slide-level embedding."""
        ...

    def prepare_coordinates(
        self,
        coordinates: Tensor,
        *,
        base_spacing_um: float,
        requested_spacing_um: float,
    ) -> Tensor:
        """Hook for model-specific coordinate normalization."""
        return coordinates


class PatientEncoder(Encoder):
    """Base class for encoders that aggregate slide embeddings into patient embeddings."""

    tile_encoder: TileEncoder | None = None

    def encode_tiles(self, batch: Tensor) -> Tensor:
        if self.tile_encoder is None:
            raise AttributeError("patient encoders must attach a tile_encoder before encoding tiles")
        return self.tile_encoder.encode_tiles(batch)

    @abstractmethod
    def encode_slide(
        self,
        tile_features: Tensor,
        coordinates: Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> Tensor:
        """Pool tile-level features into a single slide-level embedding."""
        ...

    @abstractmethod
    def encode_patient(self, slide_embeddings: Tensor) -> Tensor:
        """Aggregate slide embeddings [S, D] into a single patient-level embedding [D]."""
        ...


class TimmTileEncoder(TileEncoder):
    """Convenience base for timm-backed tile encoders."""

    def __init__(
        self,
        model_name: str,
        *,
        output_variant: str | None = None,
        **timm_kwargs,
    ):
        defaults = {"pretrained": True, "num_classes": 0}
        defaults.update(timm_kwargs)
        self._model = timm.create_model(model_name, **defaults).eval()
        self._device = preferred_default_device()
        if not hasattr(self, "_output_variant"):
            self._output_variant = resolve_requested_output_variant(output_variant)

    def get_transform(self) -> Callable:
        data_config = resolve_data_config(self._model.pretrained_cfg, model=self._model)
        return create_transform(**data_config)

    def get_dense_transform(self) -> Callable:
        # Normalization only — no Resize/CenterCrop (see TileEncoder.get_dense_transform).
        # mean/std come from the same resolved data config get_transform uses, so the
        # photometric pipeline matches pooled extraction even for encoders with custom
        # normalization (e.g. H-optimus 0.7072.../0.2119...); verified per-encoder.
        cfg = resolve_data_config(self._model.pretrained_cfg, model=self._model)
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ])

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return self._model(batch)

    def _dense_patch_size(self) -> tuple[int, int]:
        """Backbone patch size as ``(patch_h, patch_w)``."""
        patch = self._model.patch_embed.patch_size
        if isinstance(patch, int):
            return patch, patch
        patch_h, patch_w = patch
        return int(patch_h), int(patch_w)

    @property
    def patch_size(self) -> tuple[int, int]:
        return self._dense_patch_size()

    def _dense_num_prefix_tokens(self) -> int:
        """Number of leading non-patch tokens (CLS + register tokens)."""
        return int(self._model.num_prefix_tokens)

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
        """Encode tiles into a dense spatial grid. (B, C, H, W) -> (B, d, h, w).

        Runs the frozen backbone's ``forward_features`` and folds the patch-token
        sequence back into its spatial grid (CLS/register tokens discarded). The
        backbone must accept ``batch`` at its current spatial size (timm ViTs need
        ``dynamic_img_size=True`` for sizes other than their native input), and
        ``H, W`` must be divisible by the patch size.
        """
        if batch.ndim != 4:
            raise ValueError(
                "encode_tiles_dense expects a (B, C, H, W) batch, got shape "
                f"{tuple(batch.shape)}."
            )
        _, _, height, width = batch.shape
        patch_h, patch_w = self._dense_patch_size()
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Dense extraction for '{type(self).__name__}' requires input "
                f"divisible by the patch size: got {height}x{width}, patch "
                f"{patch_h}x{patch_w}. Pad the tile up to a patch multiple first."
            )
        tokens = self._model.forward_features(batch)
        return reshape_tokens_to_grid(
            tokens,
            grid_h=height // patch_h,
            grid_w=width // patch_w,
            num_prefix_tokens=self._dense_num_prefix_tokens(),
            encoder_name=type(self).__name__,
        )

    def encode_tiles_attention(
        self,
        batch: Tensor,
        *,
        blocks: tuple[int, ...] = (-1,),
        include_registers: bool = False,
    ) -> Tensor:
        """Encode tiles into per-head prefix-token attention maps (timm ViT family).

        Captures each selected block's attention input via a forward-pre-hook on
        ``blocks[i].attn`` (the fused SDPA kernel never materializes the attention
        matrix), then recomputes the softmax weights from the module's own
        projection (:func:`timm_self_attention_weights`) and folds the prefix-token
        query rows into spatial grids (:func:`prefix_attention_to_grid`). Output is
        ``(B, K, h, w)`` in ``[block][cls, reg…][head]`` order — see
        :meth:`TileEncoder.encode_tiles_attention`.
        """
        return timm_trunk_attention(
            self._model,
            batch,
            blocks=blocks,
            include_registers=include_registers,
            encoder_name=type(self).__name__,
        )

    @property
    def encode_dim(self) -> int:
        return self._model.num_features

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "TimmTileEncoder":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
