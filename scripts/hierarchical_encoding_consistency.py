"""Compare hierarchical tile encoding outputs between two slide2vec checkouts.

Encodes four 2x2-subtile regions of the fixture WSI on CPU with a deterministic
stand-in encoder, through the local path and through the shard path plus shard
merge (simulated world sizes 1, 5 and 17; 17 leaves one rank without work).

Usage (``--checkout`` is the source tree whose ``slide2vec`` is imported; default:
this repo)::

    git archive <before-commit> slide2vec | tar -x -C /tmp/before
    python scripts/hierarchical_encoding_consistency.py dump --checkout /tmp/before --out before.npz
    python scripts/hierarchical_encoding_consistency.py dump --out after.npz
    python scripts/hierarchical_encoding_consistency.py compare before.npz after.npz \\
        --record scripts/hierarchical_encoding_consistency.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.transforms import v2

REPO_ROOT = Path(__file__).resolve().parents[1]
WSI_PATH = REPO_ROOT / "tests" / "fixtures" / "input" / "test-wsi.tif"
# Level-0 origins of full-tissue 888x888 regions on the fixture WSI.
REGION_ORIGINS = [(4060, 444), (4948, 444), (1396, 5772), (1396, 11544)]
WORLD_SIZES = (1, 5, 17)
BATCH_SIZE = 6
ATOL = 1e-6
SETTINGS = {
    "slide": "tests/fixtures/input/test-wsi.tif",
    "backend": "openslide",
    "device": "cpu",
    "precision": "fp32",
    "requested_spacing_um": 0.5,
    "requested_tile_size_px": 224,
    "region_tile_multiple": 2,
    "requested_region_size_px": 448,
    "region_origins_lv0": REGION_ORIGINS,
    "batch_size": BATCH_SIZE,
    "world_sizes": list(WORLD_SIZES),
    "encoder": "per-channel quadrant means (12 features)",
}


class QuadrantMeanEncoder:
    """Per-channel means of the four tile quadrants: sensitive to tile flips and swaps."""

    def encode_tiles(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        quadrants = image.float().reshape(batch, channels, 2, height // 2, 2, width // 2)
        return quadrants.mean(dim=(3, 5)).reshape(batch, channels * 4)


def _inputs():
    from slide2vec.api import ExecutionOptions, PreprocessingConfig
    from slide2vec.runtime.types import LoadedModel

    loaded = LoadedModel(
        name="quadrant-mean",
        level="tile",
        model=QuadrantMeanEncoder(),
        transforms=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        feature_dim=12,
        device=torch.device("cpu"),
    )
    slide = SimpleNamespace(sample_id="test-wsi", image_path=WSI_PATH, mask_path=None)
    tiling_result = SimpleNamespace(
        x=np.array([x for x, _ in REGION_ORIGINS], dtype=np.int64),
        y=np.array([y for _, y in REGION_ORIGINS], dtype=np.int64),
        base_spacing_um=0.25200000393750005,
        level_downsamples=[1.0, 4.0, 16.0, 64.0],
        read_level=0,
    )
    preprocessing = PreprocessingConfig(
        backend=SETTINGS["backend"],
        requested_spacing_um=SETTINGS["requested_spacing_um"],
        requested_tile_size_px=SETTINGS["requested_tile_size_px"],
        region_tile_multiple=SETTINGS["region_tile_multiple"],
        requested_region_size_px=SETTINGS["requested_region_size_px"],
    )
    execution = ExecutionOptions(batch_size=BATCH_SIZE, num_workers_per_gpu=0, num_gpus=1, precision="fp32")
    return loaded, slide, tiling_result, preprocessing, execution


def encode_fixture() -> dict[str, np.ndarray]:
    """Local grid, per-rank shard outputs and merged grids, keyed for ``np.savez``."""
    from slide2vec.runtime.distributed import merge_hierarchical_embedding_shards
    from slide2vec.runtime.embedding_pipeline import (
        compute_hierarchical_embedding_shard_for_slide,
        compute_hierarchical_embeddings_for_slide,
    )
    from slide2vec.runtime.hierarchical import build_hierarchical_index, resolve_hierarchical_geometry

    loaded, slide, tiling_result, preprocessing, execution = _inputs()
    outputs = {
        "local": compute_hierarchical_embeddings_for_slide(
            loaded, slide, tiling_result, preprocessing=preprocessing, execution=execution
        ).numpy(),
    }
    geometry = resolve_hierarchical_geometry(preprocessing, tiling_result)
    index = build_hierarchical_index(
        tiling_result,
        region_tile_multiple=int(geometry["region_tile_multiple"]),
        tile_size_lv0=int(geometry["tile_size_lv0"]),
    )
    for world_size in WORLD_SIZES:
        payloads = []
        # Same split as the distributed worker.
        for rank, flat_indices in enumerate(np.array_split(index.flat_index, world_size)):
            shard_indices, embeddings = compute_hierarchical_embedding_shard_for_slide(
                loaded,
                slide,
                tiling_result,
                preprocessing=preprocessing,
                execution=execution,
                flat_indices=flat_indices,
            )
            outputs[f"world{world_size}_rank{rank}_flat_index"] = np.asarray(shard_indices)
            outputs[f"world{world_size}_rank{rank}_embeddings"] = embeddings.numpy()
            payloads.append({"flat_index": shard_indices, "tile_embeddings": embeddings})
        outputs[f"world{world_size}_merged"] = merge_hierarchical_embedding_shards(
            payloads,
            num_regions=index.num_regions,
            tiles_per_region=index.tiles_per_region,
        ).numpy()
    return outputs


def compare(before: dict[str, np.ndarray], after: dict[str, np.ndarray], *, atol: float = ATOL) -> dict:
    """Index arrays must match exactly; features within ``atol``. Also local vs merged."""
    checks = {}
    for key in sorted(set(before) | set(after)):
        if key not in before or key not in after:
            checks[key] = {"passed": False, "reason": "missing in " + ("before" if key not in before else "after")}
            continue
        checks[key] = _check(before[key], after[key], exact=key.endswith("_flat_index"), atol=atol)
    for world_size in WORLD_SIZES:
        merged = after.get(f"world{world_size}_merged")
        if merged is not None and "local" in after:
            checks[f"after: local vs world{world_size}_merged"] = _check(after["local"], merged, exact=False, atol=atol)
    return {"passed": all(check["passed"] for check in checks.values()), "atol": atol, "checks": checks}


def _check(expected: np.ndarray, actual: np.ndarray, *, exact: bool, atol: float) -> dict:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {
            "passed": False,
            "reason": f"shape/dtype {expected.shape}/{expected.dtype} != {actual.shape}/{actual.dtype}",
        }
    if exact:
        return {"passed": bool(np.array_equal(expected, actual)), "shape": list(actual.shape)}
    max_abs_diff = float(np.max(np.abs(expected - actual))) if expected.size else 0.0
    return {"passed": max_abs_diff <= atol, "shape": list(actual.shape), "max_abs_diff": max_abs_diff}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    dump = commands.add_parser("dump", help="encode the fixture with the slide2vec in the current directory")
    dump.add_argument("--out", type=Path, required=True)
    dump.add_argument("--checkout", type=Path, default=REPO_ROOT, help="directory containing the slide2vec package to import")
    diff = commands.add_parser("compare", help="compare two dumps")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    diff.add_argument("--atol", type=float, default=ATOL)
    diff.add_argument("--record", type=Path, help="also write the result, with settings, to this JSON file")
    args = parser.parse_args(argv)

    if args.command == "dump":
        sys.path.insert(0, str(args.checkout.resolve()))
        import slide2vec

        print(f"encoding with {Path(slide2vec.__file__).parent}")
        np.savez(args.out, **encode_fixture())
        return 0
    with np.load(args.before) as before, np.load(args.after) as after:
        result = compare(dict(before), dict(after), atol=args.atol)
    record = {
        "command": "python scripts/hierarchical_encoding_consistency.py compare BEFORE.npz AFTER.npz",
        "before": args.before.name,
        "after": args.after.name,
        "settings": SETTINGS,
        **result,
    }
    text = json.dumps(record, indent=2)
    print(text)
    if args.record is not None:
        args.record.write_text(text + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
