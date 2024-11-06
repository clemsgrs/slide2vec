import os
import tqdm
import wandb
import torch
import logging
import argparse
import traceback
import torchvision
import numpy as np
import pandas as pd
import multiprocessing as mp
import torch.distributed as dist

from pathlib import Path
from contextlib import nullcontext

import slide2vec.distributed as distributed

from slide2vec.utils import load_csv
from slide2vec.utils.config import setup, write_config, hf_login
from slide2vec.models import ModelFactory
from slide2vec.data import TileDataset, RegionUnfolding
from slide2vec.wsi import extract_coordinates, save_coordinates, visualize_coordinates

logger = logging.getLogger("slide2vec")

RESOURCE_PATH = Path("/opt/app/resources")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("slide2vec", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "opts",
        help="""Modify config options at the end of the command using "path.key=value".""".strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def main(args):
    # setup configuration
    cfg = setup(args)
    hf_login()
    wsi_paths, mask_paths = load_csv(cfg)

    if distributed.is_main_process():
        write_config(cfg, cfg.output_dir)

    num_workers_preprocessing = min(mp.cpu_count(), cfg.speed.num_workers_preprocessing)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers_preprocessing = min(
            num_workers_preprocessing, int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        )

    num_workers_data_loading = min(mp.cpu_count(), cfg.speed.num_workers_data_loading)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers_data_loading = min(
            num_workers_data_loading, int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        )

    process_list = Path(cfg.output_dir, "process_list.csv")
    if process_list.is_file() and cfg.resume:
        process_df = pd.read_csv(process_list)
    else:
        process_df = pd.DataFrame(
            {
                "path": [str(p) for p in wsi_paths],
                "mask_path": [str(p) if p is not None else p for p in mask_paths],
                "tiling_status": ["tbp"] * len(wsi_paths),
                "feature_status": ["tbp"] * len(wsi_paths),
                "error": [str(np.nan)] * len(wsi_paths),
                "traceback": [str(np.nan)] * len(wsi_paths),
            }
        )

    skip_tiling = process_df["tiling_status"].str.contains("done").all()
    skip_feature_extraction = process_df["feature_status"].str.contains("done").all()

    if not skip_tiling:
        mask = process_df["tiling_status"] != "done"
        process_stack = process_df[mask]
        total = len(process_stack)

        wsi_paths_to_process = [Path(x) for x in process_stack.path.values.tolist()]
        mask_paths_to_process = [
            Path(x) if x is not None else x
            for x in process_stack.mask_path.values.tolist()
        ]

        # extract tile coordinates input #

        tiling_updates = {}

        if distributed.is_main_process():
            coordinates_dir = Path(cfg.output_dir, "coordinates")
            coordinates_dir.mkdir(exist_ok=True, parents=True)
            if cfg.visualize:
                visualize_dir = Path(cfg.output_dir, "visualization")
                mask_visualize_dir = Path(visualize_dir, "mask")
                tile_visualize_dir = Path(visualize_dir, "tiling")
                mask_visualize_dir.mkdir(exist_ok=True, parents=True)
                tile_visualize_dir.mkdir(exist_ok=True, parents=True)
            with tqdm.tqdm(
                zip(wsi_paths_to_process, mask_paths_to_process),
                desc="Extracting tile coordinates",
                unit=" wsi",
                total=total,
                leave=True,
            ) as t:
                for wsi_fp, mask_fp in t:
                    logger.info(f"Preprocessing {wsi_fp.stem}")
                    tissue_mask_visu_path = None
                    try:
                        if cfg.visualize:
                            tissue_mask_visu_path = Path(
                                mask_visualize_dir, f"{wsi_fp.stem}.jpg"
                            )
                        coordinates, _, tile_level, resize_factor = extract_coordinates(
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
                            num_workers=num_workers_preprocessing,
                        )
                        coordinates_path = Path(coordinates_dir, f"{wsi_fp.stem}.npy")
                        save_coordinates(
                            coordinates,
                            cfg.tiling.spacing,
                            tile_level,
                            cfg.tiling.tile_size,
                            resize_factor,
                            coordinates_path,
                        )
                        if cfg.visualize:
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

                        tiling_updates[str(wsi_fp)] = {"status": "done"}

                    except Exception as e:
                        tiling_updates[str(wsi_fp)] = {
                            "status": "failed",
                            "error": str(e),
                            "traceback": str(traceback.format_exc()),
                        }

            logger.info("=+=" * 10)
            for wsi_path, status_info in tiling_updates.items():
                process_df.loc[
                    process_df["path"] == wsi_path, "tiling_status"
                ] = status_info["status"]
                if "error" in status_info:
                    process_df.loc[
                        process_df["path"] == wsi_path, "error"
                    ] = status_info["error"]
                    process_df.loc[
                        process_df["path"] == wsi_path, "traceback"
                    ] = status_info["traceback"]
            process_df.to_csv(process_list, index=False)

    # wait for all processes to finish preprocessing #

    if distributed.is_enabled():
        torch.distributed.barrier()

    if not skip_feature_extraction:
        # instantiate feature extractor #

        model = ModelFactory(cfg.model).get_model()
        if distributed.is_main_process():
            logger.info("=+=" * 10)

        coordinates_dir = Path(cfg.output_dir, "coordinates")
        wsi_with_tiles_paths = [
            str(p)
            for p in wsi_paths
            if Path(coordinates_dir, f"{p.stem}.npy").is_file()
        ]

        # extract features #

        sub_process_df = process_df[process_df.path.isin(wsi_with_tiles_paths)]
        mask = sub_process_df["feature_status"] != "done"
        process_stack = sub_process_df[mask]
        total = len(process_stack)
        already_processed = len(sub_process_df) - total

        wsi_paths_to_process = [Path(x) for x in process_stack.path.values.tolist()]

        features_dir = Path(cfg.output_dir, "features")
        if distributed.is_main_process():
            features_dir.mkdir(exist_ok=True, parents=True)

        if distributed.is_main_process():
            agg_processed_count = already_processed

        autocast_context = nullcontext()
        if cfg.speed.fp16:
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)

        feature_extraction_updates = {}

        with tqdm.tqdm(
            wsi_paths_to_process,
            desc="Inference",
            unit=" case",
            total=total,
            leave=True,
            disable=not distributed.is_main_process(),
            position=1,
        ) as t1:
            for wsi_fp in t1:
                try:
                    if cfg.model.level == "tile":
                        transforms = model.get_transforms()
                    elif cfg.model.level == "region":
                        transforms = torchvision.transforms.Compose(
                            [
                                torchvision.transforms.ToTensor(),
                                RegionUnfolding(model.tile_size),
                                model.get_transforms(),
                            ]
                        )
                    dataset = TileDataset(
                        wsi_fp,
                        coordinates_dir,
                        backend=cfg.tiling.backend,
                        transforms=transforms,
                    )
                    if distributed.is_enabled_and_multiple_gpus():
                        sampler = torch.utils.data.DistributedSampler(
                            dataset, shuffle=False
                        )
                    else:
                        sampler = None
                    dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=cfg.model.batch_size,
                        sampler=sampler,
                        num_workers=num_workers_data_loading,
                        pin_memory=True,
                    )
                    if cfg.model.level == "tile":
                        features = torch.empty(
                            (0, model.features_dim), device=model.device
                        )
                    elif cfg.model.level == "region":
                        ntile = cfg.tiling.tile_size // model.tile_size
                        num_tiles = ntile**2
                        features = torch.empty(
                            (0, num_tiles, model.tile_encoder.features_dim),
                            device=model.device,
                        )
                    indices = torch.empty((0,), dtype=torch.long, device=model.device)
                    with torch.inference_mode():
                        with autocast_context:
                            with tqdm.tqdm(
                                dataloader,
                                desc=f"GPU {distributed.get_local_rank()}: {wsi_fp.stem}",
                                unit=f" {cfg.model.level}",
                                unit_scale=cfg.model.batch_size,
                                leave=False,
                                position=2 + distributed.get_local_rank(),
                            ) as t2:
                                for batch in t2:
                                    idx, image = batch
                                    image = image.to(model.device, non_blocking=True)
                                    feature = model(
                                        image
                                    )  # (B, features_dim) or (B, npatch, features_dim)
                                    features = torch.cat(
                                        (features, feature), dim=0
                                    )  # (ntiles, features_dim) or (ntiles, npatch, features_dim)
                                    indices = torch.cat(
                                        (
                                            indices,
                                            idx.to(model.device, non_blocking=True),
                                        ),
                                        dim=0,
                                    )

                    if distributed.is_enabled_and_multiple_gpus():
                        torch.distributed.barrier()
                        # gather features and indices from all GPUs
                        wsi_feature = distributed.gather_features(
                            features,
                            indices,
                            model.device,
                            model.features_dim,
                            cfg.model.level,
                        )
                    else:
                        # handle duplicates
                        unique_indices, inverse_indices = torch.unique(
                            indices, sorted=False, return_inverse=True
                        )
                        if cfg.model.level == "tile":
                            wsi_feature = torch.zeros(
                                (unique_indices.size(0), model.features_dim),
                                device=model.device,
                            )
                        elif cfg.model.level == "region":
                            wsi_feature = torch.zeros(
                                (
                                    unique_indices.size(0),
                                    num_tiles,
                                    model.tile_encoder.features_dim,
                                ),
                                device=model.device,
                            )
                        # insert each feature into its correct position based on tile_indices
                        for idx in range(indices.size(0)):
                            index = inverse_indices[idx]
                            wsi_feature[index] = features[idx]

                    if distributed.is_main_process():
                        torch.save(wsi_feature, Path(features_dir, f"{wsi_fp.stem}.pt"))

                    if cfg.wandb.enable and distributed.is_main_process():
                        agg_processed_count += 1
                        wandb.log({"processed": agg_processed_count})

                    feature_extraction_updates[str(wsi_fp)] = {"status": "done"}

                except Exception as e:
                    feature_extraction_updates[str(wsi_fp)] = {
                        "status": "failed",
                        "error": str(e),
                        "traceback": str(traceback.format_exc()),
                    }

        if distributed.is_enabled():
            torch.distributed.barrier()

        # collect status updates from all processes
        status_updates_list = [None for _ in range(distributed.get_global_size())]
        dist.all_gather_object(status_updates_list, feature_extraction_updates)

        if distributed.is_main_process():
            for status_updates in status_updates_list:
                for wsi_path, status_info in status_updates.items():
                    process_df.loc[
                        process_df["path"] == wsi_path, "feature_status"
                    ] = status_info["status"]
                    if "error" in status_info:
                        process_df.loc[
                            process_df["path"] == wsi_path, "error"
                        ] = status_info["error"]
                        process_df.loc[
                            process_df["path"] == wsi_path, "traceback"
                        ] = status_info["traceback"]
            process_df.to_csv(process_list, index=False)

        # summary logging
        if distributed.is_main_process():
            total_slides = len(process_df)
            failed_tiling = process_df[process_df["tiling_status"] == "failed"]
            no_tiles = process_df[~process_df["path"].isin(wsi_with_tiles_paths)]
            failed_feature_extraction = process_df[
                ~(process_df["feature_status"] == "done")
            ]
            logger.info("=+=" * 10)
            logger.info("Summary:")
            logger.info(f"Total number of slides: {total_slides}")
            logger.info(f"Failed tiling: {len(failed_tiling)}")
            logger.info(f"No tiles after tiling step: {len(no_tiles)}")
            logger.info(f"Failed feature extraction: {len(failed_feature_extraction)}")
            logger.info(
                f"Completed feature extraction: {total_slides - len(failed_feature_extraction)}"
            )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message=".*Could not set the permissions.*")
    warnings.filterwarnings("ignore", message=".*antialias.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*TypedStorage.*", category=UserWarning)
    args = get_args_parser(add_help=True).parse_args()
    main(args)
