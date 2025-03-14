import os
import tqdm
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path

from slide2vec.utils import load_csv, fix_random_seeds
from slide2vec.utils.config import get_cfg_from_file
from slide2vec.wsi import extract_coordinates, save_coordinates, visualize_coordinates


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("slide2vec", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Name of output directory",
    )
    return parser


def process_slide(wsi_fp, mask_fp, cfg, mask_visualize_dir, tile_visualize_dir):
    """
    Process a single slide: extract tile coordinates and visualize if needed.
    Note: We force num_workers to 1 for extract_coordinates to disable its internal parallelism.
    """
    try:
        print(f"Preprocessing {wsi_fp.stem}")
        tissue_mask_visu_path = None
        if cfg.visualize and mask_visualize_dir is not None:
            tissue_mask_visu_path = Path(mask_visualize_dir, f"{wsi_fp.stem}.jpg")
        coordinates, _, tile_level, resize_factor, tile_size_lv0 = extract_coordinates(
            wsi_fp,
            mask_fp,
            cfg.tiling.spacing,
            cfg.tiling.tile_size,
            cfg.tiling.backend,
            tissue_val=cfg.tiling.tissue_pixel_value,
            downsample=cfg.tiling.downsample,
            segment_params=cfg.tiling.seg_params,
            tiling_params=cfg.tiling.params,
            mask_visu_path=tissue_mask_visu_path,
            num_workers=1,  # disable internal multiprocessing
        )
        coordinates_dir = Path(cfg.output_dir, "coordinates")
        coordinates_path = Path(coordinates_dir, f"{wsi_fp.stem}.npy")
        save_coordinates(
            coordinates,
            cfg.tiling.spacing,
            tile_level,
            cfg.tiling.tile_size,
            resize_factor,
            tile_size_lv0,
            coordinates_path,
        )
        if cfg.visualize and tile_visualize_dir is not None:
            visualize_coordinates(
                wsi_fp,
                coordinates,
                tile_level,
                cfg.tiling.tile_size,
                resize_factor,
                tile_visualize_dir,
                downsample=32,
                backend=cfg.tiling.backend,
            )
        return str(wsi_fp), {"status": "success"}

    except Exception as e:
        return str(wsi_fp), {
            "status": "failed",
            "error": str(e),
            "traceback": str(traceback.format_exc()),
        }


def main(args):
    # setup configuration
    cfg = get_cfg_from_file(args.config_file)
    output_dir = Path(cfg.output_dir, args.run_id)
    cfg.output_dir = str(output_dir)

    fix_random_seeds(cfg.seed)

    wsi_paths, mask_paths = load_csv(cfg)

    parallel_workers = min(mp.cpu_count(), cfg.speed.num_workers_tiling)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        parallel_workers = min(
            parallel_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        )

    process_list = Path(cfg.output_dir, "process_list.csv")
    if process_list.is_file() and cfg.resume:
        process_df = pd.read_csv(process_list)
    else:
        process_df = pd.DataFrame(
            {
                "wsi_path": [str(p) for p in wsi_paths],
                "mask_path": [str(p) if p is not None else p for p in mask_paths],
                "tiling_status": ["tbp"] * len(wsi_paths),
                "feature_status": ["tbp"] * len(wsi_paths),
                "error": [str(np.nan)] * len(wsi_paths),
                "traceback": [str(np.nan)] * len(wsi_paths),
            }
        )

    skip_tiling = process_df["tiling_status"].str.contains("success").all()

    if not skip_tiling:
        mask = process_df["tiling_status"] != "success"
        process_stack = process_df[mask]
        total = len(process_stack)

        wsi_paths_to_process = [Path(x) for x in process_stack.wsi_path.values.tolist()]
        mask_paths_to_process = [
            Path(x) if x is not None else x
            for x in process_stack.mask_path.values.tolist()
        ]

        # setup directories for coordinates and visualization
        coordinates_dir = Path(cfg.output_dir, "coordinates")
        coordinates_dir.mkdir(exist_ok=True, parents=True)
        mask_visualize_dir = None
        tile_visualize_dir = None
        if cfg.visualize:
            visualize_dir = Path(cfg.output_dir, "visualization")
            mask_visualize_dir = Path(visualize_dir, "mask")
            tile_visualize_dir = Path(visualize_dir, "tiling")
            mask_visualize_dir.mkdir(exist_ok=True, parents=True)
            tile_visualize_dir.mkdir(exist_ok=True, parents=True)

        tiling_updates = {}
        with mp.Pool(processes=parallel_workers) as pool:
            args_list = [
                (wsi_fp, mask_fp, cfg, mask_visualize_dir, tile_visualize_dir)
                for wsi_fp, mask_fp in zip(wsi_paths_to_process, mask_paths_to_process)
            ]
            results = list(
                tqdm.tqdm(
                    pool.starmap(process_slide, args_list),
                    total=total,
                    desc="Slide tiling",
                    unit="slide",
                )
            )
        for wsi_path, status_info in results:
            tiling_updates[wsi_path] = status_info

        for wsi_path, status_info in tiling_updates.items():
            process_df.loc[
                process_df["wsi_path"] == wsi_path, "tiling_status"
            ] = status_info["status"]
            if "error" in status_info:
                process_df.loc[
                    process_df["wsi_path"] == wsi_path, "error"
                ] = status_info["error"]
                process_df.loc[
                    process_df["wsi_path"] == wsi_path, "traceback"
                ] = status_info["traceback"]
        process_df.to_csv(process_list, index=False)

        # summary logging
        total_slides = len(process_df)
        slides_with_tiles = [
            str(p)
            for p in wsi_paths
            if Path(coordinates_dir, f"{p.stem}.npy").is_file()
        ]
        failed_tiling = process_df[process_df["tiling_status"] == "failed"]
        no_tiles = process_df[~process_df["wsi_path"].isin(slides_with_tiles)]
        print("=+=" * 10)
        print(f"Total number of slides: {total_slides}")
        print(f"Failed tiling: {len(failed_tiling)}")
        print(f"No tiles after tiling step: {len(no_tiles)}")
        print("=+=" * 10)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
