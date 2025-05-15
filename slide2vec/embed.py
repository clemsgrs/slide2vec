import os
import tqdm
import torch
import argparse
import traceback
import torchvision
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from contextlib import nullcontext

import slide2vec.distributed as distributed

from slide2vec.utils import fix_random_seeds
from slide2vec.utils.config import get_cfg_from_file, setup_distributed
from slide2vec.models import ModelFactory
from slide2vec.data import TileDataset, RegionUnfolding


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


def create_transforms(cfg, model):
    if cfg.model.level in ["tile", "slide"]:
        return model.get_transforms()
    elif cfg.model.level == "region":
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                RegionUnfolding(model.tile_size),
                model.get_transforms(),
            ]
        )
    else:
        raise ValueError(f"Unknown model level: {cfg.model.level}")


def create_dataset(wsi_fp, coordinates_dir, cfg, transforms):
    return TileDataset(
        wsi_fp,
        coordinates_dir,
        cfg.tiling.params.spacing,
        backend=cfg.tiling.backend,
        transforms=transforms,
    )


def sort_features(features, indices):
    """
    Sort the features tensor based on the indices.
    """
    sorted_order = indices.argsort()
    indices_sorted = indices[sorted_order]
    features_sorted = features[sorted_order]
    assert len(torch.unique(indices_sorted)) == len(indices_sorted), "Indices are not unique."
    return features_sorted


def run_inference(dataloader, model, device, autocast_context, unit, batch_size):
    """
    Run inference on the provided dataloader and return concatenated features and indices.
    """
    features_list = []
    indices_list = []
    with torch.inference_mode():
        with autocast_context:
            for batch in tqdm.tqdm(
                dataloader,
                desc=f"Inference on GPU {distributed.get_local_rank()}",
                unit=unit,
                unit_scale=batch_size,
                leave=False,
                position=2 + distributed.get_local_rank(),
            ):
                idx, image = batch
                image = image.to(device, non_blocking=True)
                feature = model(image).cpu()
                features_list.append(feature)
                indices_list.append(idx)
    # concatenate features and indices
    # and move to device
    features = torch.cat(features_list, dim=0).to(device, non_blocking=True)
    indices = torch.cat(indices_list, dim=0).to(device, non_blocking=True)
    return features, indices


def main(args):
    # setup configuration
    cfg = get_cfg_from_file(args.config_file)
    output_dir = Path(cfg.output_dir, args.run_id)
    cfg.output_dir = str(output_dir)

    setup_distributed()

    coordinates_dir = Path(cfg.output_dir, "coordinates")
    fix_random_seeds(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers_embedding)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    process_list = Path(cfg.output_dir, "process_list.csv")
    assert (
        process_list.is_file()
    ), "Process list CSV not found. Ensure tiling has been run."
    process_df = pd.read_csv(process_list)
    skip_feature_extraction = process_df["feature_status"].str.contains("success").all()

    if skip_feature_extraction and distributed.is_main_process():
        print("Feature extraction already completed.")
        return

    model = ModelFactory(cfg.model).get_model()
    if distributed.is_main_process():
        print("Starting feature extraction...")
    torch.distributed.barrier()

    # select slides that were successfully tiled but not yet processed for feature extraction
    sub_process_df = process_df[process_df.tiling_status == "success"]
    mask = sub_process_df["feature_status"] != "success"
    process_stack = sub_process_df[mask]
    total = len(process_stack)
    wsi_paths_to_process = [Path(x) for x in process_stack.wsi_path.values.tolist()]

    features_dir = Path(cfg.output_dir, "features")
    if distributed.is_main_process():
        features_dir.mkdir(exist_ok=True, parents=True)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if cfg.speed.fp16
        else nullcontext()
    )
    unit = "tile" if cfg.model.level != "region" else "region"
    feature_extraction_updates = {}

    transforms = create_transforms(cfg, model)

    for wsi_fp in tqdm.tqdm(
        wsi_paths_to_process,
        desc="Inference",
        unit="slide",
        total=total,
        leave=True,
        disable=not distributed.is_main_process(),
        position=1,
    ):
        try:
            dataset = create_dataset(wsi_fp, coordinates_dir, cfg, transforms)
            if distributed.is_enabled_and_multiple_gpus():
                sampler = torch.utils.data.DistributedSampler(
                    dataset,
                    shuffle=False,
                    drop_last=False,
                )
            else:
                sampler = None
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.model.batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
            )

            features, indices = run_inference(
                dataloader,
                model,
                model.device,
                autocast_context,
                unit,
                cfg.model.batch_size,
            )

            # gather features from all gpus if needed
            if distributed.is_enabled_and_multiple_gpus():
                features_list = distributed.gather_tensor(features)
                indices_list = distributed.gather_tensor(indices)
                if distributed.is_main_process():
                    features_gathered = torch.cat(features_list, dim=0)
                    indices_gathered = torch.cat(indices_list, dim=0)
                else:
                    # For non-main processes, a placeholder is provided.
                    features_gathered = torch.rand(
                        (len(dataset), model.features_dim), device=model.device
                    )
                    indices_gathered = None

            if distributed.is_main_process():
                wsi_feature = sort_features(features_gathered, indices_gathered)
                indices = list(indices_gathered.cpu())

            torch.distributed.barrier()

            # for slide-level models, align coordinates with feature order
            # then run forward pass with slide encoder
            if cfg.model.level == "slide":
                if distributed.is_main_process():
                    if cfg.model.name == "prov-gigapath":
                        coordinates = torch.tensor(
                            dataset.scaled_coordinates[indices],
                            dtype=torch.int64,
                            device=model.device,
                        )
                    else:
                        coordinates = torch.tensor(
                            dataset.coordinates[indices],
                            dtype=torch.int64,
                            device=model.device,
                        )
                else:
                    coordinates = torch.randint(
                        10000,
                        (len(dataset), 2),
                        dtype=torch.int64,
                        device=model.device,
                    )
                with torch.inference_mode():
                    with autocast_context:
                        wsi_feature = model.forward_slide(
                            wsi_feature,
                            tile_coordinates=coordinates,
                            tile_size_lv0=dataset.tile_size_lv0,
                        )

            if distributed.is_main_process():
                torch.save(wsi_feature, Path(features_dir, f"{wsi_fp.stem}.pt"))

            feature_extraction_updates[str(wsi_fp)] = {"status": "success"}

        except Exception as e:
            feature_extraction_updates[str(wsi_fp)] = {
                "status": "failed",
                "error": str(e),
                "traceback": str(traceback.format_exc()),
            }

        # update process_df
        if distributed.is_main_process():
            status_info = feature_extraction_updates[str(wsi_fp)]
            process_df.loc[
                process_df["wsi_path"] == str(wsi_fp), "feature_status"
            ] = status_info["status"]
            if "error" in status_info:
                process_df.loc[
                    process_df["wsi_path"] == str(wsi_fp), "error"
                ] = status_info["error"]
                process_df.loc[
                    process_df["wsi_path"] == str(wsi_fp), "traceback"
                ] = status_info["traceback"]
            process_df.to_csv(process_list, index=False)

    if distributed.is_enabled_and_multiple_gpus():
        torch.distributed.barrier()

    if distributed.is_main_process():
        # summary logging
        slides_with_tiles = len(sub_process_df)
        total_slides = len(process_df)
        failed_feature_extraction = process_df[
            ~(process_df["feature_status"] == "success")
        ]
        print("=+=" * 10)
        print(f"Total number of slides with tiles: {slides_with_tiles}/{total_slides}")
        print(f"Failed feature extraction: {len(failed_feature_extraction)}")
        print(
            f"Completed feature extraction: {total_slides - len(failed_feature_extraction)}"
        )
        print("=+=" * 10)

    if distributed.is_enabled():
       torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
