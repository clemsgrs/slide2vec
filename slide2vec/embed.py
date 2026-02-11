import argparse
import gc
import multiprocessing as mp
import os
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm

import slide2vec.distributed as distributed

from slide2vec.data import RegionUnfolding, TileDataset
from slide2vec.hs2p.hs2p.wsi import SamplingParameters
from slide2vec.models import ModelFactory
from slide2vec.utils import fix_random_seeds
from slide2vec.utils.config import get_cfg_from_file, setup_distributed

torchvision.disable_beta_transforms_warning()


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("slide2vec", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="output directory to save logs and checkpoints",
    )
    parser.add_argument("--run-on-cpu", action="store_true", help="run inference on cpu")
    parser.add_argument(
        "opts",
        help='Modify config options at the end of the command using "path.key=value".',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def create_transforms(cfg, model):
    if cfg.model.level in ["tile", "slide"]:
        return model.get_transforms()
    if cfg.model.level == "region":
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                RegionUnfolding(model.tile_size),
                model.get_transforms(),
            ]
        )
    raise ValueError(f"Unknown model level: {cfg.model.level}")


def create_dataset(
    wsi_path,
    mask_path,
    coordinates_dir,
    target_spacing,
    tolerance,
    backend,
    segment_params,
    sampling_params,
    filter_params,
    transforms,
    restrict_to_tissue: bool,
):
    return TileDataset(
        wsi_path=wsi_path,
        mask_path=mask_path,
        coordinates_dir=coordinates_dir,
        target_spacing=target_spacing,
        tolerance=tolerance,
        backend=backend,
        segment_params=segment_params,
        sampling_params=sampling_params,
        filter_params=filter_params,
        transforms=transforms,
        restrict_to_tissue=restrict_to_tissue,
    )


def get_speed_option(cfg, key: str, default):
    speed_cfg = getattr(cfg, "speed", None)
    if speed_cfg is None:
        return default
    if not hasattr(speed_cfg, key):
        return default
    value = getattr(speed_cfg, key)
    if value is None:
        return default
    return value


def parse_slurm_cpus(value: str | None):
    if value is None:
        return None
    head = value.split("(")[0]
    digits = "".join(ch for ch in head if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def resolve_loader_settings(cfg, run_on_cpu: bool):
    world_size = max(1, distributed.get_global_size())
    cpu_count = mp.cpu_count()

    workers_cfg = get_speed_option(cfg, "num_workers_embedding", "auto")
    auto_workers = isinstance(workers_cfg, str) and workers_cfg.lower() == "auto"
    if auto_workers:
        workers_per_rank = cpu_count // world_size
        workers_per_rank = max(4, min(16, workers_per_rank))
    else:
        workers_per_rank = int(workers_cfg)

    slurm_cpus = parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
    if slurm_cpus is not None:
        workers_per_rank = min(workers_per_rank, max(1, slurm_cpus // world_size))

    workers_per_rank = max(0, workers_per_rank)

    storage_mode = str(get_speed_option(cfg, "storage_mode", "auto")).lower()
    prefetch_cfg = get_speed_option(cfg, "prefetch_factor_embedding", None)
    if prefetch_cfg is None:
        prefetch_factor = 4 if storage_mode in {"network", "auto"} else 2
    else:
        prefetch_factor = int(prefetch_cfg)

    persistent_workers = bool(get_speed_option(cfg, "persistent_workers_embedding", True))
    pin_memory = bool(get_speed_option(cfg, "pin_memory_embedding", True))
    loader_timeout_sec = int(get_speed_option(cfg, "loader_batch_timeout_sec", 0))

    if run_on_cpu:
        # CPU inference in containerized CI frequently has very limited /dev/shm.
        # Force single-process loading to avoid worker shared-memory crashes.
        workers_per_rank = 0
        pin_memory = False
        persistent_workers = False

    if workers_per_rank <= 0:
        persistent_workers = False
        prefetch_factor = None

    pipeline = str(get_speed_option(cfg, "embedding_pipeline", "v1")).lower()
    rank_sharding_mode = str(get_speed_option(cfg, "rank_sharding_mode", "auto")).lower()
    log_perf_embedding = bool(get_speed_option(cfg, "log_perf_embedding", False))

    return {
        "embedding_pipeline": pipeline,
        "rank_sharding_mode": rank_sharding_mode,
        "storage_mode": storage_mode,
        "num_workers": workers_per_rank,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
        "loader_timeout_sec": loader_timeout_sec,
        "log_perf_embedding": log_perf_embedding,
    }


def create_dataloader(dataset, cfg, runtime, sampler=None):
    kwargs = {
        "batch_size": cfg.model.batch_size,
        "sampler": sampler,
        "num_workers": runtime["num_workers"],
        "pin_memory": runtime["pin_memory"],
        "timeout": runtime["loader_timeout_sec"],
    }
    if runtime["num_workers"] > 0:
        kwargs["persistent_workers"] = runtime["persistent_workers"]
        if runtime["prefetch_factor"] is not None:
            kwargs["prefetch_factor"] = runtime["prefetch_factor"]
    return torch.utils.data.DataLoader(dataset, **kwargs)


def collect_pending_slides(process_df, coordinates_dir):
    tiled_df = process_df[process_df.tiling_status == "success"]
    pending_df = tiled_df[tiled_df["feature_status"] != "success"]

    slides = []
    for _, row in pending_df.iterrows():
        wsi_path = Path(row.wsi_path)
        name = wsi_path.stem.replace(" ", "_")
        coordinates_file = coordinates_dir / f"{name}.npy"
        tile_count = 0
        if coordinates_file.is_file():
            try:
                coordinates = np.load(coordinates_file, allow_pickle=True)
                tile_count = int(len(coordinates["x"]))
            except Exception:
                tile_count = 0
        mask_path = None
        if "mask_path" in row and row.mask_path is not None and not pd.isna(row.mask_path):
            mask_path = str(row.mask_path)
        slides.append(
            {
                "wsi_path": str(wsi_path),
                "mask_path": mask_path,
                "name": name,
                "tile_count": tile_count,
            }
        )
    return slides, tiled_df


def assign_slides_lpt(slides, world_size):
    assignments = {rank: [] for rank in range(world_size)}
    loads = {rank: 0 for rank in range(world_size)}
    for slide in sorted(slides, key=lambda x: x["tile_count"], reverse=True):
        rank = min(loads, key=lambda r: (loads[r], r))
        assignments[rank].append(slide)
        loads[rank] += max(1, int(slide["tile_count"]))
    return assignments


def decide_sharding_mode(cfg, pending_count, world_size):
    mode = str(get_speed_option(cfg, "rank_sharding_mode", "auto")).lower()
    if mode == "tile":
        return "tile"
    if mode == "slide":
        return "slide"
    if mode == "auto":
        return "slide" if pending_count >= world_size else "tile"
    raise ValueError(f"Unknown rank sharding mode: {mode}")


def get_feature_path(features_dir: Path, name: str, cfg):
    feature_path = features_dir / f"{name}.pt"
    if cfg.model.save_tile_embeddings:
        feature_path = features_dir / f"{name}-tiles.pt"
    return feature_path


def log_perf_summary(name: str, stats: dict, unit: str):
    total_time = stats["data_wait_s"] + stats["h2d_s"] + stats["forward_s"] + stats["write_s"]
    data_wait_pct = 100.0 * stats["data_wait_s"] / max(total_time, 1e-8)
    tiles_per_sec = stats["samples"] / max(stats["elapsed_s"], 1e-8)
    print(
        f"[perf] {name}: {stats['samples']} {unit}s, {tiles_per_sec:.2f} {unit}s/s, "
        f"data_wait={stats['data_wait_s']:.2f}s ({data_wait_pct:.1f}%), "
        f"h2d={stats['h2d_s']:.2f}s, forward={stats['forward_s']:.2f}s, write={stats['write_s']:.2f}s"
    )


def run_inference_to_h5(
    dataloader,
    model,
    device,
    autocast_context,
    unit,
    batch_size,
    feature_path,
    collect_indices,
    run_on_cpu,
    show_progress,
):
    device_name = f"GPU {distributed.get_global_rank()}" if not run_on_cpu else "CPU"

    stats = {
        "data_wait_s": 0.0,
        "h2d_s": 0.0,
        "forward_s": 0.0,
        "write_s": 0.0,
        "samples": 0,
        "batches": 0,
        "elapsed_s": 0.0,
    }

    start = time.perf_counter()
    with h5py.File(feature_path, "w") as f:
        features = None
        indices = None

        iterator = iter(dataloader)
        progress = tqdm.tqdm(
            total=len(dataloader),
            desc=f"Inference on {device_name}",
            unit=unit,
            unit_scale=batch_size,
            leave=False,
            position=2 + distributed.get_global_rank(),
            disable=not show_progress,
        )

        with torch.inference_mode(), autocast_context:
            while True:
                data_wait_start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                stats["data_wait_s"] += time.perf_counter() - data_wait_start

                idx, image = batch

                h2d_start = time.perf_counter()
                image = image.to(device, non_blocking=not run_on_cpu)
                stats["h2d_s"] += time.perf_counter() - h2d_start

                forward_start = time.perf_counter()
                feature_cpu = model(image)["embedding"].cpu()
                stats["forward_s"] += time.perf_counter() - forward_start

                write_start = time.perf_counter()
                feature_np = feature_cpu.numpy()
                if features is None:
                    feature_dim = feature_np.shape[1:]
                    dtype = feature_np.dtype
                    features = f.create_dataset(
                        "features",
                        shape=(0, *feature_dim),
                        maxshape=(None, *feature_dim),
                        dtype=dtype,
                        chunks=(batch_size, *feature_dim),
                    )
                    if collect_indices:
                        indices = f.create_dataset(
                            "indices",
                            shape=(0,),
                            maxshape=(None,),
                            dtype="int64",
                            chunks=(batch_size,),
                        )

                features.resize(features.shape[0] + feature_np.shape[0], axis=0)
                features[-feature_np.shape[0] :] = feature_np

                if collect_indices:
                    idx_np = idx.cpu().numpy() if hasattr(idx, "cpu") else np.asarray(idx)
                    indices.resize(indices.shape[0] + idx_np.shape[0], axis=0)
                    indices[-idx_np.shape[0] :] = idx_np

                stats["write_s"] += time.perf_counter() - write_start
                stats["samples"] += int(feature_np.shape[0])
                stats["batches"] += 1
                progress.update(1)

                del image, feature_cpu, feature_np

        progress.close()

    if stats["batches"] == 0:
        raise RuntimeError("No batches yielded by DataLoader.")

    stats["elapsed_s"] = time.perf_counter() - start

    if not run_on_cpu:
        torch.cuda.empty_cache()
    gc.collect()

    return stats


def load_features_from_h5(feature_path):
    with h5py.File(feature_path, "r") as f:
        features = torch.from_numpy(f["features"][:])
    return features


def load_sort_and_deduplicate_features(tmp_dir, name, expected_len=None):
    features_list, indices_list = [], []
    for rank in range(distributed.get_global_size()):
        fp = tmp_dir / f"{name}-rank_{rank}.h5"
        with h5py.File(fp, "r") as f:
            features_list.append(torch.from_numpy(f["features"][:]))
            indices_list.append(torch.from_numpy(f["indices"][:]))
        os.remove(fp)
    features = torch.cat(features_list, dim=0)
    indices = torch.cat(indices_list, dim=0)
    order = torch.argsort(indices)
    indices = indices[order]
    features = features[order]

    keep = torch.ones_like(indices, dtype=torch.bool)
    keep[1:] = indices[1:] != indices[:-1]
    indices_unique = indices[keep]
    features_unique = features[keep]
    if expected_len is not None:
        assert len(indices_unique) == expected_len, f"Got {len(indices_unique)} items, expected {expected_len}"
        assert torch.unique(indices_unique).numel() == len(indices_unique), "Indices are not unique after sorting"
    return features_unique


def resolve_output_dir(config_output_dir: str, cli_output_dir: str | None) -> Path:
    if cli_output_dir is None:
        return Path(config_output_dir)
    cli_path = Path(cli_output_dir)
    if cli_path.is_absolute():
        return cli_path
    return Path(config_output_dir, cli_output_dir)


def cleanup_tmp_features(tmp_dir: Path, name: str):
    for rank in range(distributed.get_global_size()):
        fp = tmp_dir / f"{name}-rank_{rank}.h5"
        if fp.exists():
            os.remove(fp)


def run_embed_v1(
    *,
    slides,
    process_df,
    process_list,
    model,
    cfg,
    coordinates_dir,
    sampling_params,
    transforms,
    runtime,
    autocast_context,
    features_dir,
    tmp_dir,
    run_on_cpu,
    unit,
):
    feature_extraction_updates = {}

    for slide in tqdm.tqdm(
        slides,
        desc="Inference",
        unit="slide",
        total=len(slides),
        leave=True,
        disable=not distributed.is_main_process(),
        position=1,
    ):
        wsi_fp = Path(slide["wsi_path"])
        mask_fp = Path(slide["mask_path"]) if slide["mask_path"] is not None else None
        name = slide["name"]

        feature_path = get_feature_path(features_dir, name, cfg)
        tmp_feature_path = tmp_dir / f"{name}-rank_{distributed.get_global_rank()}.h5"

        status_info = {"status": "success"}
        local_failed = False

        try:
            dataset = create_dataset(
                wsi_path=wsi_fp,
                mask_path=mask_fp,
                coordinates_dir=coordinates_dir,
                target_spacing=cfg.tiling.params.spacing,
                tolerance=cfg.tiling.params.tolerance,
                backend=cfg.tiling.backend,
                segment_params=cfg.tiling.seg_params,
                sampling_params=sampling_params,
                filter_params=cfg.tiling.filter_params,
                transforms=transforms,
                restrict_to_tissue=cfg.model.restrict_to_tissue,
            )
            if distributed.is_enabled_and_multiple_gpus():
                sampler = torch.utils.data.DistributedSampler(
                    dataset,
                    shuffle=False,
                    drop_last=False,
                )
            else:
                sampler = None

            dataloader = create_dataloader(dataset, cfg, runtime, sampler=sampler)
            perf_stats = run_inference_to_h5(
                dataloader=dataloader,
                model=model,
                device=model.device,
                autocast_context=autocast_context,
                unit=unit,
                batch_size=cfg.model.batch_size,
                feature_path=tmp_feature_path,
                collect_indices=True,
                run_on_cpu=run_on_cpu,
                show_progress=distributed.is_main_process(),
            )
            if runtime["log_perf_embedding"] and distributed.is_main_process():
                log_perf_summary(name, perf_stats, unit)

        except Exception as e:
            local_failed = True
            status_info = {
                "status": "failed",
                "error": str(e),
                "traceback": str(traceback.format_exc()),
            }

        any_rank_failed = local_failed
        if not run_on_cpu:
            torch.distributed.barrier()
            failure_flag = torch.tensor(
                1 if local_failed else 0, device=model.device, dtype=torch.int32
            )
            torch.distributed.all_reduce(failure_flag, op=torch.distributed.ReduceOp.MAX)
            any_rank_failed = bool(failure_flag.item())

        if any_rank_failed:
            if distributed.is_main_process():
                cleanup_tmp_features(tmp_dir, name)
                if status_info["status"] != "failed":
                    status_info = {
                        "status": "failed",
                        "error": "Feature extraction failed on at least one distributed rank.",
                        "traceback": "",
                    }
        elif distributed.is_main_process():
            try:
                wsi_feature = load_sort_and_deduplicate_features(
                    tmp_dir, name, expected_len=len(dataset)
                )
                torch.save(wsi_feature, feature_path)
            except Exception as e:
                any_rank_failed = True
                cleanup_tmp_features(tmp_dir, name)
                status_info = {
                    "status": "failed",
                    "error": str(e),
                    "traceback": str(traceback.format_exc()),
                }
            finally:
                if "wsi_feature" in locals():
                    del wsi_feature
                if not run_on_cpu:
                    torch.cuda.empty_cache()
                gc.collect()

        if not run_on_cpu:
            failure_flag = torch.tensor(
                1 if (distributed.is_main_process() and any_rank_failed) else 0,
                device=model.device,
                dtype=torch.int32,
            )
            torch.distributed.broadcast(failure_flag, src=0)
            torch.distributed.barrier()
            any_rank_failed = bool(failure_flag.item())

        if distributed.is_main_process():
            if any_rank_failed and status_info["status"] != "failed":
                status_info = {
                    "status": "failed",
                    "error": "Feature extraction failed on at least one distributed rank.",
                    "traceback": "",
                }
            feature_extraction_updates[str(wsi_fp)] = status_info

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


def run_embed_v2(
    *,
    slides,
    process_df,
    process_list,
    model,
    cfg,
    coordinates_dir,
    sampling_params,
    transforms,
    runtime,
    autocast_context,
    features_dir,
    tmp_dir,
    run_on_cpu,
    unit,
):
    world_size = distributed.get_global_size()
    sharding_mode = decide_sharding_mode(cfg, pending_count=len(slides), world_size=world_size)

    if sharding_mode == "tile":
        if distributed.is_main_process():
            print(
                "Embedding v2 requested but switching to tile-level sharding "
                f"(pending_slides={len(slides)} < world_size={world_size})."
            )
        return run_embed_v1(
            slides=slides,
            process_df=process_df,
            process_list=process_list,
            model=model,
            cfg=cfg,
            coordinates_dir=coordinates_dir,
            sampling_params=sampling_params,
            transforms=transforms,
            runtime=runtime,
            autocast_context=autocast_context,
            features_dir=features_dir,
            tmp_dir=tmp_dir,
            run_on_cpu=run_on_cpu,
            unit=unit,
        )

    if distributed.is_main_process():
        slides_to_assign = slides
    else:
        slides_to_assign = None

    if distributed.is_enabled():
        payload = [slides_to_assign]
        torch.distributed.broadcast_object_list(payload, src=0)
        slides_to_assign = payload[0]

    assignments = assign_slides_lpt(slides_to_assign, world_size=world_size)
    rank = distributed.get_global_rank()
    local_slides = assignments[rank]

    if distributed.is_main_process():
        print(
            f"Embedding v2 slide-sharding enabled. "
            f"Rank 0 assigned {len(local_slides)} / {len(slides_to_assign)} slides."
        )

    local_updates = {}

    for slide in tqdm.tqdm(
        local_slides,
        desc=f"Inference (rank {rank})",
        unit="slide",
        total=len(local_slides),
        leave=True,
        disable=not distributed.is_main_process(),
        position=1,
    ):
        wsi_fp = Path(slide["wsi_path"])
        mask_fp = Path(slide["mask_path"]) if slide["mask_path"] is not None else None
        name = slide["name"]

        feature_path = get_feature_path(features_dir, name, cfg)
        tmp_feature_path = tmp_dir / f"{name}-rank_{rank}-v2.h5"

        status_info = {"status": "success"}
        try:
            dataset = create_dataset(
                wsi_path=wsi_fp,
                mask_path=mask_fp,
                coordinates_dir=coordinates_dir,
                target_spacing=cfg.tiling.params.spacing,
                tolerance=cfg.tiling.params.tolerance,
                backend=cfg.tiling.backend,
                segment_params=cfg.tiling.seg_params,
                sampling_params=sampling_params,
                filter_params=cfg.tiling.filter_params,
                transforms=transforms,
                restrict_to_tissue=cfg.model.restrict_to_tissue,
            )
            dataloader = create_dataloader(dataset, cfg, runtime, sampler=None)
            perf_stats = run_inference_to_h5(
                dataloader=dataloader,
                model=model,
                device=model.device,
                autocast_context=autocast_context,
                unit=unit,
                batch_size=cfg.model.batch_size,
                feature_path=tmp_feature_path,
                collect_indices=False,
                run_on_cpu=run_on_cpu,
                show_progress=distributed.is_main_process(),
            )

            wsi_feature = load_features_from_h5(tmp_feature_path)
            torch.save(wsi_feature, feature_path)
            os.remove(tmp_feature_path)
            del wsi_feature

            if runtime["log_perf_embedding"]:
                print(f"[rank {rank}]", end=" ")
                log_perf_summary(name, perf_stats, unit)

            if not run_on_cpu:
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            status_info = {
                "status": "failed",
                "error": str(e),
                "traceback": str(traceback.format_exc()),
            }
            if tmp_feature_path.exists():
                os.remove(tmp_feature_path)

        local_updates[str(wsi_fp)] = status_info

    if distributed.is_enabled():
        gathered_updates = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_updates, local_updates)
    else:
        gathered_updates = [local_updates]

    if distributed.is_main_process():
        merged_updates = {}
        for update in gathered_updates:
            merged_updates.update(update)

        for wsi_path, status_info in merged_updates.items():
            process_df.loc[process_df["wsi_path"] == str(wsi_path), "feature_status"] = status_info[
                "status"
            ]
            if "error" in status_info:
                process_df.loc[process_df["wsi_path"] == str(wsi_path), "error"] = status_info[
                    "error"
                ]
                process_df.loc[
                    process_df["wsi_path"] == str(wsi_path), "traceback"
                ] = status_info["traceback"]

        process_df.to_csv(process_list, index=False)

    if distributed.is_enabled_and_multiple_gpus():
        torch.distributed.barrier()


def print_feature_summary(process_df, tiled_df, unit):
    slides_with_tiles = len(tiled_df)
    total_slides = len(process_df)
    failed_feature_extraction = process_df[process_df["feature_status"] == "failed"]
    print("=+=" * 10)
    print(f"Total number of slides with {unit}s: {slides_with_tiles}/{total_slides}")
    print(
        f"Failed {unit}-level feature extraction: {len(failed_feature_extraction)}/{slides_with_tiles}"
    )
    print(
        f"Completed {unit}-level feature extraction: {slides_with_tiles - len(failed_feature_extraction)}/{slides_with_tiles}"
    )
    print("=+=" * 10)


def main(args):
    run_on_cpu = args.run_on_cpu
    cfg = get_cfg_from_file(args.config_file)
    output_dir = resolve_output_dir(cfg.output_dir, args.output_dir)
    cfg.output_dir = str(output_dir)

    if not run_on_cpu:
        setup_distributed()

    if cfg.tiling.read_coordinates_from:
        coordinates_dir = Path(cfg.tiling.read_coordinates_from)
    else:
        coordinates_dir = Path(cfg.output_dir, "coordinates")

    fix_random_seeds(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    unit = "tile" if cfg.model.level != "region" else "region"
    runtime = resolve_loader_settings(cfg, run_on_cpu=run_on_cpu)
    if runtime["embedding_pipeline"] not in {"v1", "v2"}:
        raise ValueError(
            f"Unknown embedding pipeline: {runtime['embedding_pipeline']}. "
            "Expected one of: v1, v2."
        )

    process_list = Path(cfg.output_dir, "process_list.csv")
    assert process_list.is_file(), "Process list CSV not found. Ensure tiling has been run."
    process_df = pd.read_csv(process_list)

    if "feature_status" not in process_df.columns:
        process_df["feature_status"] = ["tbp"] * len(process_df)
    if "mask_path" not in process_df.columns:
        process_df["mask_path"] = [None] * len(process_df)

    cols = [
        "wsi_name",
        "wsi_path",
        "mask_path",
        "tiling_status",
        "feature_status",
        "error",
        "traceback",
    ]
    process_df = process_df[cols]
    process_df["error"] = process_df["error"].astype("object")
    process_df["traceback"] = process_df["traceback"].astype("object")

    skip_feature_extraction = process_df["feature_status"].str.contains("success").all()

    if skip_feature_extraction:
        if distributed.is_main_process():
            print("=+=" * 10)
            print(f"All slides have been embedded. Skipping {unit}-level feature extraction step.")
            print("=+=" * 10)
        if distributed.is_enabled():
            torch.distributed.destroy_process_group()
        return

    model = ModelFactory(cfg.model).get_model()
    if distributed.is_main_process():
        print(f"Starting {unit}-level feature extraction...")
    if not run_on_cpu:
        torch.distributed.barrier()

    pixel_mapping = {k: v for e in cfg.tiling.sampling_params.pixel_mapping for k, v in e.items()}
    tissue_percentage = {k: v for e in cfg.tiling.sampling_params.tissue_percentage for k, v in e.items()}
    if "tissue" not in tissue_percentage:
        tissue_percentage["tissue"] = cfg.tiling.params.min_tissue_percentage
    if cfg.tiling.sampling_params.color_mapping is not None:
        color_mapping = {k: v for e in cfg.tiling.sampling_params.color_mapping for k, v in e.items()}
    else:
        color_mapping = None

    sampling_params = SamplingParameters(
        pixel_mapping=pixel_mapping,
        color_mapping=color_mapping,
        tissue_percentage=tissue_percentage,
    )

    slides, tiled_df = collect_pending_slides(process_df, coordinates_dir)

    features_dir = Path(cfg.output_dir, "features")
    features_dir.mkdir(exist_ok=True, parents=True)

    tmp_dir = Path("/tmp")
    tmp_dir.mkdir(exist_ok=True, parents=True)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (cfg.speed.fp16 and not run_on_cpu)
        else nullcontext()
    )

    transforms = create_transforms(cfg, model)
    if distributed.is_main_process():
        print(f"transforms: {transforms}")
        print(
            "loader settings: "
            f"pipeline={runtime['embedding_pipeline']}, "
            f"sharding={runtime['rank_sharding_mode']}, workers={runtime['num_workers']}, "
            f"prefetch={runtime['prefetch_factor']}, persistent_workers={runtime['persistent_workers']}, "
            f"pin_memory={runtime['pin_memory']}"
        )

    if runtime["embedding_pipeline"] == "v2":
        run_embed_v2(
            slides=slides,
            process_df=process_df,
            process_list=process_list,
            model=model,
            cfg=cfg,
            coordinates_dir=coordinates_dir,
            sampling_params=sampling_params,
            transforms=transforms,
            runtime=runtime,
            autocast_context=autocast_context,
            features_dir=features_dir,
            tmp_dir=tmp_dir,
            run_on_cpu=run_on_cpu,
            unit=unit,
        )
    else:
        run_embed_v1(
            slides=slides,
            process_df=process_df,
            process_list=process_list,
            model=model,
            cfg=cfg,
            coordinates_dir=coordinates_dir,
            sampling_params=sampling_params,
            transforms=transforms,
            runtime=runtime,
            autocast_context=autocast_context,
            features_dir=features_dir,
            tmp_dir=tmp_dir,
            run_on_cpu=run_on_cpu,
            unit=unit,
        )

    if distributed.is_main_process():
        print_feature_summary(process_df, tiled_df, unit)

    if distributed.is_enabled():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
