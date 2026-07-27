"""The one work-splitting kernel every distributed extraction path shares.

slide2vec fans a flat, ordered list of independent work units (dense ROIs, given-geometry
images) across ``world_size`` ranks. There is exactly one rule for how that split is made,
and it lives here so no path can grow a second one.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def plan_contiguous_shards(items: Sequence[T], world_size: int) -> list[list[T]]:
    """Partition an ordered work list into ``world_size`` contiguous shards.

    Contiguous :func:`numpy.array_split` over the index range: exactly ``world_size``
    shards, balanced to within one item, deterministic, and an exact ordered partition of
    the input (the union preserves order, the shards are pairwise disjoint). Contiguity is
    what keeps a slide's ROIs together on the dense path — only slides straddling a shard
    boundary are opened twice — and costs the given-image path nothing. Shards may be empty
    when ``world_size`` exceeds the item count.

    Every rank derives its own shard from the same ordered list, so no coordination,
    assignment file, or collective is needed to agree on who owns what.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be at least 1, got {world_size}")
    items = list(items)
    index_shards = np.array_split(np.arange(len(items)), world_size)
    return [[items[int(i)] for i in shard] for shard in index_shards]
