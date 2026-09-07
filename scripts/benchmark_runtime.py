#!/usr/bin/env python3
"""Repeatable runtime benchmarks; see docs/performance.md for scope and comparison."""

import argparse
import hashlib
from importlib.metadata import version
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def hierarchical_workload(args):
    from slide2vec.data.tile_reader import OnTheFlyHierarchicalBatchCollator

    size, tile = args.region_size, args.tile_size
    if size % tile:
        raise ValueError('--region-size must be divisible by --tile-size')
    per_region = (size // tile) ** 2
    # Distinct pixels/channels/regions, without randomness or model weights.
    region = np.arange(size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    regions = [(region + i).astype(np.uint8) for i in range(args.regions)]
    collator = OnTheFlyHierarchicalBatchCollator(
        image_path=args.slide or Path('synthetic.svs'),
        tiling_result=SimpleNamespace(
            read_level=0,
            x=(np.arange(args.regions) % 2) * size if args.slide else np.arange(args.regions),
            y=(np.arange(args.regions) // 2) * size if args.slide else np.zeros(args.regions),
        ),
        region_index=np.repeat(np.arange(args.regions), per_region),
        subtile_index_within_region=np.tile(np.arange(per_region), args.regions),
        read_region_size_px=size, read_tile_size_px=tile,
        requested_tile_size_px=tile, backend='openslide',
    )

    class Reader:
        def read_region(self, location, level, size):
            return regions[location[0]]

    # Exclude storage/decoding so this measures the complete CPU collation path.
    if not args.slide:
        collator._reader._reader = Reader()
    indices = list(range(args.regions * per_region))

    def run():
        return collator(indices)[1]

    def digest(result):
        return hashlib.sha256(result.numpy().tobytes()).hexdigest()

    return run, digest, lambda: None


def process_list_workload(args):
    from slide2vec.runtime.persistence import update_process_list_after_embedding

    directory = tempfile.TemporaryDirectory(prefix='slide2vec-perf-')
    path = Path(directory.name) / 'process_list.csv'
    initial = pd.DataFrame({
        'sample_id': [f'slide-{i}' for i in range(args.rows)],
        'annotation': ['tissue'] * args.rows,
        'feature_status': ['tbp'] * args.rows,
    }).to_csv(index=False)
    count = min(args.completed, args.rows)
    slides = [SimpleNamespace(sample_id=f'slide-{i}') for i in range(count)]
    artifacts = [
        SimpleNamespace(sample_id=s.sample_id, annotation=None,
                        path=Path('/benchmark/embeddings') / f'{s.sample_id}.pt')
        for s in slides
    ]

    def reset():
        path.write_text(initial)

    def run():
        update_process_list_after_embedding(
            path, successful_slides=slides, persist_tile_embeddings=True,
            persist_hierarchical_embeddings=False, include_slide_embeddings=False,
            encoder_name='benchmark', output_variant='default', tile_artifacts=artifacts,
            hierarchical_artifacts=[], slide_artifacts=[],
        )
        return path

    def digest(result):
        return hashlib.sha256(result.read_bytes()).hexdigest()

    # Keep the temporary directory alive for the lifetime of these closures.
    run.directory = directory
    return run, digest, reset


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['hierarchical', 'process-list', 'inference'], required=True)
    parser.add_argument('--repeat', type=positive_int, default=5)
    parser.add_argument('--warmup', type=positive_int, default=1)
    parser.add_argument('--threads', type=positive_int, default=1)
    parser.add_argument('--regions', type=positive_int, default=2)
    parser.add_argument('--region-size', type=positive_int, default=1024)
    parser.add_argument('--tile-size', type=positive_int, default=256)
    parser.add_argument('--rows', type=positive_int, default=10000)
    parser.add_argument('--completed', type=positive_int, default=1000)
    parser.add_argument('--slide', type=Path, help='Read real level-0 regions with OpenSlide instead of synthetic pixels')
    parser.add_argument('--compare', type=Path, help='Require matching workload/output and report speedup against this JSON')
    parser.add_argument('--model', default='phikonv2', help='Pretrained encoder for inference measurements')
    parser.add_argument('--coordinates', type=Path, help='NPZ of fixed level-0 x/y coordinates for inference')
    parser.add_argument('--batch-size', type=positive_int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--backend', choices=['openslide', 'cucim', 'asap', 'vips'], default='openslide')
    parser.add_argument('--modes', nargs='+', choices=['model-only', 'cached', 'wsi'], default=['model-only', 'cached', 'wsi'])
    parser.add_argument('--use-supertiles', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--cache-policy', choices=['warm', 'fresh-reader', 'advised-client-drop'], default='warm')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--profile', action='store_true', help='Export separate untimed inference CPU/CUDA profiler traces')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.case == 'inference':
        if args.slide is None or args.coordinates is None:
            parser.error('--case inference requires --slide and --coordinates')
        from scripts.benchmark_inference import run_inference_benchmark

        print(json.dumps(run_inference_benchmark(args), indent=2))
        return
    if args.slide and args.case != 'hierarchical':
        parser.error('--slide is only supported for --case hierarchical')
    torch.set_num_threads(args.threads)
    factory = hierarchical_workload if args.case == 'hierarchical' else process_list_workload
    run, digest, reset = factory(args)
    for _ in range(args.warmup):
        reset()
        result = run()
        del result
    samples = []
    checksums = set()
    for _ in range(args.repeat):
        reset()
        start = time.perf_counter()
        result = run()
        samples.append(time.perf_counter() - start)
        checksums.add(digest(result))
        del result
    if len(checksums) != 1:
        raise RuntimeError('Repeated runs produced different outputs')
    report = {
        'case': args.case,
        'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                       if k in ('case', 'repeat', 'warmup', 'threads', 'regions', 'region_size',
                                'tile_size', 'rows', 'completed', 'slide')},
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
                        'hs2p': version('hs2p')},
        'seconds': samples, 'median_seconds': statistics.median(samples),
        'min_seconds': min(samples), 'max_seconds': max(samples),
        'output_sha256': checksums.pop(),
    }
    source = ('slide2vec/data/tile_reader.py' if args.case == 'hierarchical'
              else 'slide2vec/runtime/persistence.py')
    root = Path(__file__).resolve().parents[1]
    report['source_sha256'] = hashlib.sha256((root / source).read_bytes()).hexdigest()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
    report['revision'] = revision.stdout.strip() if revision.returncode == 0 else None
    if args.compare:
        baseline = json.loads(args.compare.read_text())
        if baseline['parameters'] != report['parameters'] or baseline['environment'] != report['environment']:
            raise ValueError('Baseline workload/environment differs; rerun with matching arguments and dependencies')
        if baseline['output_sha256'] != report['output_sha256']:
            raise ValueError('Baseline output differs; investigate correctness before comparing performance')
        report['speedup'] = baseline['median_seconds'] / report['median_seconds']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
