import os
import tqdm
import torch
import logging
import argparse
import torchvision
import multiprocessing as mp

from pathlib import Path

import slide2vec.distributed as distributed

from slide2vec.utils import load_csv
from slide2vec.utils.config import setup, write_config
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
        "--resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument(
        "opts",
        help="""Modify config options at the end of the command using "path.key=value".""".strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def gather_features(features, indices, device, features_dim, level):
    gathered_feature = [
        torch.zeros_like(features, device=device)
        for _ in range(distributed.get_global_size())
    ]
    gathered_indices = [
        torch.zeros_like(indices, device=device)
        for _ in range(distributed.get_global_size())
    ]
    torch.distributed.all_gather(gathered_feature, features)
    torch.distributed.all_gather(gathered_indices, indices)
    if distributed.is_main_process():
        # concatenate the gathered features and indices
        wsi_feature = torch.cat(gathered_feature, dim=0)
        tile_indices = torch.cat(gathered_indices, dim=0)
        # remove duplicates
        unique_indices = torch.unique(tile_indices)
        # create a final tensor to store the features in the correct order
        if level == "tile":
            wsi_feature_ordered = torch.zeros(
                (len(unique_indices), features_dim), device=device
            )
        elif level == "region":
            num_tiles = features.shape[1]
            wsi_feature_ordered = torch.zeros(
                (len(unique_indices), num_tiles, features_dim), device=device
            )
        # insert each feature into its correct position based on tile_indices
        wsi_feature_ordered[unique_indices] = wsi_feature[unique_indices]
    else:
        wsi_feature_ordered = None
    return wsi_feature_ordered


def main(args):
    cfg = setup(args)
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

    # extract tile coordinates input #

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
            zip(wsi_paths, mask_paths),
            desc="Extracting tile coordinates",
            unit=" wsi",
            total=len(wsi_paths),
            leave=True,
        ) as t:
            for wsi_fp, mask_fp in t:
                tqdm.tqdm.write(f"Preprocessing {wsi_fp.stem}")
                tissue_mask_visu_path = None
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
                        backend="asap",
                    )
        logger.info("=+=" * 10)

    # wait for all processes to finish preprocessing #

    if distributed.is_enabled():
        torch.distributed.barrier()

    # instantiate feature extractor #

    model = ModelFactory(cfg.model).get_model()
    if distributed.is_main_process():
        logger.info("=+=" * 10)

    coordinates_dir = Path(cfg.output_dir, "coordinates")
    wsi_paths = [
        p for p in wsi_paths if Path(coordinates_dir, f"{Path(p).stem}.npy").is_file()
    ]
    if distributed.is_main_process():
        logger.info(f"{len(wsi_paths)} slides with extracted tiles found\n")

    # extract features #

    features_dir = Path(cfg.output_dir, "features")
    if distributed.is_main_process():
        features_dir.mkdir(exist_ok=True, parents=True)

    with tqdm.tqdm(
        wsi_paths,
        desc="Inference",
        unit=" case",
        total=len(wsi_paths),
        leave=True,
        disable=not distributed.is_main_process(),
        position=1,
    ) as t:
        for fp in t:
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
                fp,
                coordinates_dir,
                backend=cfg.tiling.backend,
                transforms=transforms,
            )
            if distributed.is_enabled_and_multiple_gpus():
                sampler = torch.utils.data.DistributedSampler(dataset)
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
                features = torch.empty((0, model.features_dim), device=model.device)
            elif cfg.model.level == "region":
                ntile = cfg.tiling.tile_size // model.tile_size
                num_tiles = ntile**2
                features = torch.empty(
                    (0, num_tiles, model.tile_encoder.features_dim),
                    device=model.device,
                )
            indices = torch.empty((0,), dtype=torch.long, device=model.device)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                with tqdm.tqdm(
                    dataloader,
                    desc=f"GPU {distributed.get_local_rank()}: {fp.stem}",
                    unit=f" {cfg.model.level}",
                    unit_scale=cfg.model.batch_size,
                    leave=False,
                    position=2 + distributed.get_local_rank(),
                ) as t:
                    for batch in t:
                        idx, image = batch
                        image = image.to(model.device, non_blocking=True)
                        feature = model(
                            image
                        )  # (B, features_dim) or (B, ntile, features_dim)
                        features = torch.cat(
                            (features, feature), dim=0
                        )  # (ntiles, features_dim) or (ntiles, ntile, features_dim)
                        indices = torch.cat(
                            (indices, idx.to(model.device, non_blocking=True)), dim=0
                        )

            if distributed.is_enabled():
                torch.distributed.barrier()
                # gather features and indices from all GPUs
                wsi_feature = gather_features(
                    features, indices, model.device, model.features_dim, cfg.model.level
                )
            else:
                # remove duplicates and reorder features
                unique_indices = torch.unique(indices)
                wsi_feature = torch.zeros(
                    (len(dataset), model.features_dim), device=model.device
                )
                wsi_feature[unique_indices] = features[unique_indices]

            if distributed.is_main_process():
                torch.save(wsi_feature, Path(features_dir, f"{fp.stem}.pt"))


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
