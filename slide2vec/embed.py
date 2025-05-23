import gc
import os
import numpy as np
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

torchvision.disable_beta_transforms_warning()


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


def deduplicate_and_sort_features(features, indices):
    """
    Dediplicate and sort the features tensor based on the indices.
    """
    sorted_order = indices.argsort()
    indices_sorted = indices[sorted_order]
    features_sorted = features[sorted_order]

    dedup_dict = {}
    for i, idx in enumerate(indices_sorted):
        if idx.item() not in dedup_dict:
            dedup_dict[idx.item()] = features_sorted[i]

    del features_sorted
    del indices_sorted

    indices_sorted_unique = sorted(list(dedup_dict.keys()))
    assert len(set(indices_sorted_unique)) == len(indices_sorted_unique), "Indices are not unique."
    features_sorted_unique = torch.stack([dedup_dict[k] for k in indices_sorted_unique], dim=0)
    return features_sorted_unique, indices_sorted_unique


def run_inference(dataloader, model, device, autocast_context, unit, batch_size, features):
    with torch.inference_mode(), autocast_context:
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
            feature = model(image).cpu().numpy()
            features[idx] = feature

            # cleanup
            del image, feature

    # cleanup
    del features
    gc.collect()


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
    print(f"transforms: {transforms}")

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
                pin_memory=False,
            )

            feature_path = features_dir / f"{wsi_fp.stem.replace(' ', '_')}.pt"

            # get feature dimension using a dry run
            with torch.inference_mode(), autocast_context:
                sample_batch = next(iter(dataloader))
                sample_image = sample_batch[1].to(model.device)
                sample_feature = model(sample_image).cpu().numpy()
                if cfg.model.level == "region":
                    num_tiles = sample_feature.shape[-2]
                    feature_dim = sample_feature.shape[-1]
                else:
                    feature_dim = sample_feature.shape[-1]
                dtype = sample_feature.dtype

            # create a memory-mapped tensor on disk
            if cfg.model.level == "region":
                num_regions = len(dataset)
                tmp_feature_path = features_dir / f".{wsi_fp.stem.replace(' ', '_')}.npy"
                features = np.memmap(tmp_feature_path, dtype=dtype, mode='w+', shape=(num_regions, num_tiles, feature_dim))
            else:
                num_tiles = len(dataset)
                tmp_feature_path = features_dir / f".{wsi_fp.stem.replace(' ', '_')}.npy"
                features = np.memmap(tmp_feature_path, dtype=dtype, mode='w+', shape=(num_tiles, feature_dim))

            run_inference(
                dataloader,
                model,
                model.device,
                autocast_context,
                unit,
                cfg.model.batch_size,
                features,
            )

            torch.distributed.barrier()

            # for slide-level models, align coordinates with feature order
            # then run forward pass with slide encoder
            if cfg.model.level == "slide":
                if distributed.is_main_process():
                    if cfg.model.name == "prov-gigapath":
                        coordinates = torch.tensor(
                            dataset.scaled_coordinates,
                            dtype=torch.int64,
                            device=model.device,
                        )
                    else:
                        coordinates = torch.tensor(
                            dataset.coordinates,
                            dtype=torch.int64,
                            device=model.device,
                        )
                else:
                    coordinates = torch.randint(
                        10000,
                        (num_tiles, 2),
                        dtype=torch.int64,
                        device=model.device,
                    )
                with torch.inference_mode():
                    with autocast_context:
                        features = torch.from_numpy(
                            np.memmap(tmp_feature_path, dtype=dtype, mode='r', shape=(num_tiles, feature_dim))
                        ).to(model.device)
                        wsi_feature = model.forward_slide(
                            features,
                            tile_coordinates=coordinates,
                            tile_size_lv0=dataset.tile_size_lv0,
                        )

            else:
                if cfg.model.level == "region":
                    wsi_feature = torch.from_numpy(
                        np.memmap(tmp_feature_path, dtype=dtype, mode='r', shape=(num_regions, num_tiles, feature_dim))
                    ).to(model.device)
                else:
                    wsi_feature = torch.from_numpy(
                        np.memmap(tmp_feature_path, dtype=dtype, mode='r', shape=(num_tiles, feature_dim))
                    ).to(model.device)

            if distributed.is_main_process():
                torch.save(wsi_feature, feature_path)
                os.remove(tmp_feature_path)

            torch.distributed.barrier()
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
