import os
import zarr
import time
import cv2
import argparse
import concurrent.futures
import multiprocessing as mp
import numpy as np
import pandas as pd
import wholeslidedata as wsd

from pathlib import Path
from numcodecs import Blosc

from slide2vec.utils import fix_random_seeds
from slide2vec.utils.config import get_cfg_from_file


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("slide2vec", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Name of output subdirectory",
    )
    return parser


_WSI = None


def _worker_init(wsi_path: str, backend: str):
    global _WSI
    _WSI = wsd.WholeSlideImage(Path(wsi_path), backend=backend)


def _fetch_tile(args):
    """
    args: (x, y, tile_size_resized, tile_level, ts)
    returns: uint8 ndarray (ts, ts, 3)
    """
    x, y, tile_size_resized, tile_level, ts = args
    global _WSI
    tile_spacing = _WSI.spacings[tile_level]
    tile = _WSI.get_patch(
        x,
        y,
        tile_size_resized,
        tile_size_resized,
        spacing=tile_spacing,
        center=False
    )
    if ts != tile_size_resized:
        tile = cv2.resize(tile, (ts, ts), interpolation=cv2.INTER_AREA)
    return tile.astype(np.uint8)


def save_tiles_to_zarr(
    *,
    wsi_path: Path,
    coordinates_dir: Path,
    save_dir: Path,
    target_spacing: float,
    backend: str,
    batch_size: int,
):
    name = wsi_path.stem.replace(" ", "_")
    tmp_path = save_dir / f"{name}.zarr.partial"
    save_path = save_dir / f"{name}.zarr"

    coords_path = coordinates_dir / f"{name}.npy"
    coords = np.load(coords_path, allow_pickle=True)
    xs = coords["x"].astype(int)
    ys = coords["y"].astype(int)
    coordinates = np.array([xs, ys]).T

    tile_level = coords["tile_level"].astype(int)
    tile_size_resized = coords["tile_size_resized"].astype(int)
    resize_factor = coords["resize_factor"]
    target_tile_size = np.round(tile_size_resized / resize_factor).astype(int)

    num_tiles = len(coordinates)

    # create zarr container
    ts = target_tile_size[0]
    assert np.all(target_tile_size == ts), "Found varying target tile size!"
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    z = zarr.open(
        tmp_path,
        mode="w",
        shape=(num_tiles, ts, ts, 3),
        chunks=(min(batch_size, num_tiles), ts, ts, 3),
        dtype="uint8",
        compressor=compressor
    )

    parallel_workers = mp.cpu_count()
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        parallel_workers = min(
            parallel_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=parallel_workers,
        initializer=_worker_init,
        initargs=(str(wsi_path), backend),
    ) as exc:

        # write tiles in batches to improve I/O
        start_time = time.time()
        for i in range(0, num_tiles, batch_size):
            batch_start_time = time.time()
            j = min(i + batch_size, num_tiles)

            # prepare tasks for workers (no wsi_path/backend to avoid reopening)
            args = [
                (
                    int(xs[idx]),
                    int(ys[idx]),
                    int(tile_size_resized[idx]),
                    int(tile_level[idx]),
                    int(ts),
                )
                for idx in range(i, j)
            ]

            # map preserves order so z[i:j] will match coordinates
            tiles = list(exc.map(_fetch_tile, args))

            # write batch directly into zarr
            z[i:j] = np.stack(tiles, axis=0)
            batch_end_time = time.time()
            batch_elapsed = batch_end_time - batch_start_time
            print(f"Saved {j-i} tiles in {batch_elapsed:.2f} seconds.")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Saved {num_tiles} tiles for slide {name} to zarr in {elapsed:.2f} seconds.")

    z.attrs.update({
        "slide_name": name,
        "spacing": target_spacing,
        "tile_size": ts,
        "channels": 3,
        "dtype": "uint8",
        "backend": backend,
        "chunks_tiles": int(min(batch_size, num_tiles)),
    })

    tmp_path.rename(save_path)


def main(args):
    # setup configuration
    cfg = get_cfg_from_file(args.config_file)
    output_dir = Path(cfg.output_dir, args.run_id)
    cfg.output_dir = str(output_dir)

    if cfg.tiling.read_coordinates_from:
        coordinates_dir = Path(cfg.tiling.read_coordinates_from)
    else:
        coordinates_dir = Path(cfg.output_dir, "coordinates")

    fix_random_seeds(cfg.seed)

    process_list = Path(cfg.output_dir, "process_list.csv")
    assert (
        process_list.is_file()
    ), "Process list CSV not found. Ensure tiling has been run."
    process_df = pd.read_csv(process_list)

    # select slides that were successfully tiled and not yet processed for feature extraction
    tiled_df = process_df[process_df.tiling_status == "success"]
    mask = tiled_df["feature_status"] != "success"
    process_stack = tiled_df[mask]
    wsi_paths_to_process = [Path(x) for x in process_stack.wsi_path.values.tolist()]

    tile_dir = Path("/tmp/buffered_tiles")
    tile_dir.mkdir(exist_ok=True, parents=True)

    buffer_size = 3 # max number of slides to keep in buffer
    for wsi_fp in wsi_paths_to_process:

        while len(list(tile_dir.glob("*.zarr"))) >= buffer_size:
            # wait for embedding to finish current slide
            time.sleep(2)

        save_tiles_to_zarr(
            wsi_path=wsi_fp,
            coordinates_dir=coordinates_dir,
            save_dir=tile_dir,
            target_spacing=cfg.tiling.params.spacing,
            backend=cfg.tiling.backend,
            batch_size=cfg.model.batch_size,
        )


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
