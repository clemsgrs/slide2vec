"""The shared contiguous work split every distributed extraction path rides on.

``plan_contiguous_shards`` is pure and generic: it partitions an ordered work list —
dense ROIs (issue #217) or given-geometry images (issue #234) — into ``world_size``
contiguous shards. Every rank derives its own shard from the same list, so these
properties (exact partition, order preserved, balanced to ±1, deterministic) are what
guarantee no unit is dropped, duplicated, or encoded twice.
"""

from __future__ import annotations

import pytest

from slide2vec.runtime.sharding import plan_contiguous_shards


def test_balanced_within_one():
    sizes = [len(shard) for shard in plan_contiguous_shards(list(range(10)), 4)]
    assert sizes == [3, 3, 2, 2]
    assert max(sizes) - min(sizes) <= 1


def test_shards_are_contiguous_slices():
    """Contiguity keeps neighbouring units (e.g. one slide's ROIs) on the same rank."""
    assert plan_contiguous_shards(list(range(10)), 3) == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]


def test_allows_empty_shards_when_more_ranks_than_items():
    assert [len(shard) for shard in plan_contiguous_shards(["a", "b"], 4)] == [1, 1, 0, 0]


def test_rejects_non_positive_world_size():
    with pytest.raises(ValueError, match="world_size"):
        plan_contiguous_shards(["a"], 0)
