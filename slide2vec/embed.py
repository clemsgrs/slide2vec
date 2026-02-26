import gc
import os
import tqdm
import torch
import argparse
import traceback
import torchvision
import pandas as pd
import numpy as np
import multiprocessing as mp
import inspect
import time
import stat

from pathlib import Path
from contextlib import nullcontext

import slide2vec.distributed as distributed

from slide2vec.utils import fix_random_seeds
from slide2vec.utils.config import get_cfg_from_file, setup_distributed
from slide2vec.models import ModelFactory
from slide2vec.data import (
    TileDataset,
    TileCatalogDataset,
    RegionUnfolding,
    ensure_tile_catalogs,
)
from slide2vec.hs2p.hs2p.wsi import SamplingParameters

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
    parser.add_argument(
        "--run-on-cpu", action="store_true", help="run inference on cpu"
    )
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command using \"path.key=value\".",
        default=None,
        nargs=argparse.REMAINDER,
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


def create_dataset(
    wsi_path,
    mask_path,
    coordinates_dir,
    catalog_path,
    target_spacing,
    tolerance,
    backend,
    segment_params,
    sampling_params,
    filter_params,
    transforms,
    restrict_to_tissue: bool,
    use_parquet: bool,
    max_open_slides_per_worker: int,
):
    if use_parquet:
        if catalog_path is None:
            raise ValueError("catalog_path must be provided when speed.use_parquet=true")
        return TileCatalogDataset(
            catalog_path=catalog_path,
            wsi_path=wsi_path,
            mask_path=mask_path,
            target_spacing=target_spacing,
            tolerance=tolerance,
            backend=backend,
            segment_params=segment_params,
            sampling_params=sampling_params,
            filter_params=filter_params,
            transforms=transforms,
            restrict_to_tissue=restrict_to_tissue,
            max_open_slides_per_worker=max_open_slides_per_worker,
        )

    if coordinates_dir is None:
        raise ValueError("coordinates_dir must be provided when speed.use_parquet=false")
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
        max_open_slides_per_worker=max_open_slides_per_worker,
    )


def run_inference(
    dataloader,
    model,
    device,
    autocast_context,
    unit,
    batch_size,
    shard_prefix,
    expected_num_samples,
    run_on_cpu: bool,
):
    device_name = f"GPU {distributed.get_global_rank()}" if not run_on_cpu else "CPU"
    features_path = Path(f"{shard_prefix}.features.npy")
    indices_path = Path(f"{shard_prefix}.indices.npy")
    for fp in (features_path, indices_path):
        if fp.exists():
            os.remove(fp)
    write_offset = 0
    features_mm = None
    indices_mm = None
    with torch.inference_mode(), autocast_context:
        for batch in tqdm.tqdm(
            dataloader,
            desc=f"Inference on {device_name}",
            unit=unit,
            unit_scale=batch_size,
            leave=False,
            position=2 + distributed.get_global_rank(),
        ):
            idx, image = batch
            image = image.to(device, non_blocking=True)
            feature = model(image)["embedding"].cpu().numpy()
            idx_np = idx.cpu().numpy()
            batch_len = int(feature.shape[0])

            if features_mm is None:
                features_mm = np.lib.format.open_memmap(
                    str(features_path),
                    mode="w+",
                    dtype=feature.dtype,
                    shape=(int(expected_num_samples), *feature.shape[1:]),
                )
                indices_mm = np.lib.format.open_memmap(
                    str(indices_path),
                    mode="w+",
                    dtype=np.int64,
                    shape=(int(expected_num_samples),),
                )

            end = write_offset + batch_len
            if end > expected_num_samples:
                raise RuntimeError(
                    f"Received {end} samples but expected {expected_num_samples} for {features_path}"
                )
            features_mm[write_offset:end] = feature
            indices_mm[write_offset:end] = idx_np
            write_offset = end

            # cleanup
            del image, feature, idx, idx_np, batch

    if features_mm is None:
        raise RuntimeError(f"No batches were produced for {features_path}")
    if write_offset != expected_num_samples:
        raise RuntimeError(
            f"Wrote {write_offset} samples but expected {expected_num_samples} for {features_path}"
        )
    features_mm.flush()
    indices_mm.flush()
    del features_mm, indices_mm
    for fp in (features_path, indices_path):
        try:
            os.chmod(
                fp,
                stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
            )
        except OSError:
            # Best effort only; some filesystems ignore chmod/ACL updates.
            pass

    # cleanup
    if not run_on_cpu:
        torch.cuda.empty_cache()
    gc.collect()


def _open_npy_for_read_with_retry(
    fp: Path,
    max_attempts: int = 60,
    initial_delay_s: float = 0.1,
):
    delay = max(0.01, float(initial_delay_s))
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return np.load(str(fp), mmap_mode="r", allow_pickle=False)
        except (BlockingIOError, FileNotFoundError, PermissionError, OSError) as exc:
            last_exc = exc
            errno_val = getattr(exc, "errno", None)
            msg = str(exc).lower()
            if isinstance(exc, PermissionError):
                try:
                    os.chmod(fp, 0o666)
                except OSError:
                    pass
            retryable = (
                isinstance(exc, (BlockingIOError, FileNotFoundError, PermissionError))
                or errno_val in (11, 13)
                or "unable to lock file" in msg
                or "resource temporarily unavailable" in msg
                or "permission denied" in msg
            )
            if (not retryable) or (attempt >= max_attempts - 1):
                break
            time.sleep(delay)
            delay = min(delay * 1.5, 1.0)

    if isinstance(last_exc, FileNotFoundError):
        raise FileNotFoundError(
            f"Missing shard file after retries: {fp}. "
            "If running multi-node, ensure shard temp dir is shared across ranks."
        ) from last_exc
    if last_exc is not None:
        raise RuntimeError(f"Unable to open shard file for reading: {fp}") from last_exc
    raise RuntimeError(f"Unable to open shard file for reading: {fp}")


def _tmp_shard_paths(tmp_dir: Path, name: str, rank: int) -> tuple[Path, Path]:
    prefix = Path(tmp_dir, f"{name}-rank_{rank}")
    return Path(f"{prefix}.features.npy"), Path(f"{prefix}.indices.npy")


def load_features_with_indexed_fill(tmp_dir, name, expected_len: int):
    if expected_len < 1:
        raise ValueError(f"expected_len must be >= 1, got {expected_len}")

    merged_features = None
    seen = torch.zeros(expected_len, dtype=torch.bool)

    for rank in range(distributed.get_global_size()):
        feat_path, idx_path = _tmp_shard_paths(tmp_dir, name, rank)
        feat_ds = _open_npy_for_read_with_retry(feat_path)
        idx_ds = _open_npy_for_read_with_retry(idx_path)
        if feat_ds.shape[0] != idx_ds.shape[0]:
            raise RuntimeError(
                f"Mismatched features/indices rows for rank {rank}: "
                f"{feat_ds.shape[0]} vs {idx_ds.shape[0]}"
            )

        if merged_features is None:
            probe = np.empty((), dtype=feat_ds.dtype)
            merged_features = torch.empty(
                (expected_len, *feat_ds.shape[1:]),
                dtype=torch.from_numpy(probe).dtype,
            )
        elif tuple(feat_ds.shape[1:]) != tuple(merged_features.shape[1:]):
            raise RuntimeError(
                f"Inconsistent feature shape for rank {rank}: got {feat_ds.shape[1:]}, "
                f"expected {tuple(merged_features.shape[1:])}"
            )

        chunk_rows = 8192
        total_rows = int(idx_ds.shape[0])
        for start in range(0, total_rows, chunk_rows):
            end = min(start + chunk_rows, total_rows)
            idx_np = np.asarray(idx_ds[start:end], dtype=np.int64)
            feat_np = np.asarray(feat_ds[start:end])

            if idx_np.shape[0] != feat_np.shape[0]:
                raise RuntimeError(
                    f"Mismatched chunk rows for rank {rank}: "
                    f"{idx_np.shape[0]} vs {feat_np.shape[0]}"
                )
            if idx_np.size == 0:
                continue
            if np.any(idx_np < 0) or np.any(idx_np >= expected_len):
                bad = idx_np[(idx_np < 0) | (idx_np >= expected_len)][0]
                raise RuntimeError(
                    f"Out-of-range tile index {int(bad)} for expected_len={expected_len}"
                )

            _, first_pos = np.unique(idx_np, return_index=True)
            first_mask_np = np.zeros(idx_np.shape[0], dtype=bool)
            first_mask_np[first_pos] = True

            idx_first = torch.from_numpy(idx_np[first_mask_np])
            feat_first = torch.from_numpy(feat_np[first_mask_np])

            unseen_mask = ~seen[idx_first]
            if unseen_mask.any():
                idx_write = idx_first[unseen_mask]
                merged_features[idx_write] = feat_first[unseen_mask]
                seen[idx_write] = True

        del feat_ds, idx_ds
        if feat_path.exists():
            os.remove(feat_path)
        if idx_path.exists():
            os.remove(idx_path)

    if merged_features is None:
        raise RuntimeError(f"No shard data found for {name}")

    missing = torch.nonzero(~seen, as_tuple=False)
    if missing.numel() > 0:
        missing_count = int(missing.numel())
        first_missing = int(missing[0].item())
        raise RuntimeError(
            f"Missing {missing_count} tile embeddings after merge for {name}. "
            f"First missing index: {first_missing}"
        )
    return merged_features


def resolve_output_dir(config_output_dir: str, cli_output_dir: str | None) -> Path:
    if cli_output_dir is None:
        return Path(config_output_dir)
    cli_path = Path(cli_output_dir)
    if cli_path.is_absolute():
        return cli_path
    return Path(config_output_dir, cli_output_dir)


def cleanup_tmp_features(tmp_dir: Path, name: str):
    for rank in range(distributed.get_global_size()):
        feat_path, idx_path = _tmp_shard_paths(tmp_dir, name, rank)
        if feat_path.exists():
            os.remove(feat_path)
        if idx_path.exists():
            os.remove(idx_path)


def cleanup_tmp_feature_dir(tmp_dir: Path):
    if not tmp_dir.exists():
        return
    for pattern in ("*.features.npy", "*.indices.npy"):
        for fp in tmp_dir.glob(pattern):
            try:
                fp.unlink()
            except OSError:
                pass
    try:
        tmp_dir.rmdir()
    except OSError:
        # Keep directory if it's not empty or cannot be removed on this filesystem.
        pass


def main(args):
    # setup configuration
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
    deterministic_inference = bool(cfg.speed.get("deterministic_inference", False))
    cudnn_benchmark = bool(cfg.speed.get("cudnn_benchmark", not deterministic_inference))
    torch.backends.cudnn.deterministic = deterministic_inference
    torch.backends.cudnn.benchmark = cudnn_benchmark and not deterministic_inference

    unit = "tile" if cfg.model.level != "region" else "region"

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers_embedding)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))
    persistent_workers = bool(cfg.speed.get("persistent_workers_embedding", True))
    prefetch_factor = max(1, int(cfg.speed.get("prefetch_factor_embedding", 4)))
    use_parquet = bool(cfg.speed.get("use_parquet", True))
    dataloader_supports_in_order = (
        "in_order" in inspect.signature(torch.utils.data.DataLoader.__init__).parameters
    )
    max_open_slides_per_worker = max(
        1, int(cfg.speed.get("max_open_slides_per_worker", 16))
    )

    process_list = Path(cfg.output_dir, "process_list.csv")
    assert (
        process_list.is_file()
    ), "Process list CSV not found. Ensure tiling has been run."
    process_df = pd.read_csv(process_list)
    cols = ["wsi_name", "wsi_path", "tiling_status", "error", "traceback"]
    if "feature_status" not in process_df.columns:
        process_df["feature_status"] = ["tbp"] * len(process_df)
    if "mask_path" not in process_df.columns:
        process_df["mask_path"] = [None] * len(process_df)
    cols = ["wsi_name", "wsi_path", "mask_path", "tiling_status", "feature_status", "error", "traceback"]
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

    else:
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

        # select slides that were successfully tiled but not yet processed for feature extraction
        tiled_df = process_df[process_df.tiling_status == "success"]
        mask = tiled_df["feature_status"] != "success"
        process_stack = tiled_df[mask]
        total = len(process_stack)

        wsi_paths_to_process = [Path(x) for x in process_stack.wsi_path.values.tolist()]
        mask_paths_to_process = [Path(x) if x is not None and not pd.isna(x) else None  for x in process_stack.mask_path.values.tolist()]
        slide_mask_pairs = list(zip(wsi_paths_to_process, mask_paths_to_process))

        features_dir = Path(cfg.output_dir, "features")
        if distributed.is_main_process():
            features_dir.mkdir(exist_ok=True, parents=True)

        if use_parquet:
            catalog_dir = Path(cfg.output_dir, "tile_catalog")
            if distributed.is_main_process():
                ensure_tile_catalogs(
                    slide_mask_pairs=slide_mask_pairs,
                    coordinates_dir=coordinates_dir,
                    catalog_dir=catalog_dir,
                )
            if distributed.is_enabled_and_multiple_gpus():
                torch.distributed.barrier()
            slide_to_catalog = {
                str(wsi_fp): Path(catalog_dir, f"{wsi_fp.stem.replace(' ', '_')}.parquet")
                for wsi_fp, _ in slide_mask_pairs
            }
        else:
            slide_to_catalog = {}

        tmp_dir = Path(cfg.output_dir, "tmp_feature_shards")
        tmp_dir.mkdir(exist_ok=True, parents=True)
        try:
            os.chmod(tmp_dir, 0o777)
        except OSError:
            pass
        if distributed.is_enabled_and_multiple_gpus():
            torch.distributed.barrier()

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (cfg.speed.fp16 and not run_on_cpu)
            else nullcontext()
        )
        feature_extraction_updates = {}

        transforms = create_transforms(cfg, model)
        print(f"transforms: {transforms}")

        for wsi_fp, mask_fp in tqdm.tqdm(
            slide_mask_pairs,
            desc="Inference",
            unit="slide",
            total=total,
            leave=True,
            disable=not distributed.is_main_process(),
            position=1,
        ):
            name = wsi_fp.stem.replace(" ", "_")
            feature_path = features_dir / f"{name}.pt"
            if cfg.model.save_tile_embeddings:
                feature_path = features_dir / f"{name}-tiles.pt"
            tmp_feature_prefix = tmp_dir / f"{name}-rank_{distributed.get_global_rank()}"

            status_info = {"status": "success"}
            local_failed = False
            try:
                catalog_path = slide_to_catalog[str(wsi_fp)] if use_parquet else None
                dataset = create_dataset(
                    wsi_path=wsi_fp,
                    mask_path=mask_fp,
                    coordinates_dir=coordinates_dir,
                    catalog_path=catalog_path,
                    target_spacing=cfg.tiling.params.spacing,
                    tolerance=cfg.tiling.params.tolerance,
                    backend=cfg.tiling.backend,
                    segment_params=cfg.tiling.seg_params,
                    sampling_params=sampling_params,
                    filter_params=cfg.tiling.filter_params,
                    transforms=transforms,
                    restrict_to_tissue=cfg.model.restrict_to_tissue,
                    use_parquet=use_parquet,
                    max_open_slides_per_worker=max_open_slides_per_worker,
                )
                if len(dataset) == 0:
                    source_desc = (
                        f"catalog {catalog_path}"
                        if use_parquet
                        else f"coordinates file {Path(coordinates_dir, f'{name}.npy')}"
                    )
                    raise ValueError(
                        f"No tiles found for slide {wsi_fp} ({source_desc})"
                    )
                if distributed.is_enabled_and_multiple_gpus():
                    sampler = torch.utils.data.DistributedSampler(
                        dataset,
                        shuffle=False,
                        drop_last=False,
                    )
                else:
                    sampler = None
                loader_kwargs = {
                    "dataset": dataset,
                    "batch_size": cfg.model.batch_size,
                    "sampler": sampler,
                    "num_workers": num_workers,
                    "pin_memory": not run_on_cpu,
                }
                if num_workers > 0:
                    loader_kwargs["persistent_workers"] = persistent_workers
                    loader_kwargs["prefetch_factor"] = prefetch_factor
                    if dataloader_supports_in_order:
                        loader_kwargs["in_order"] = False
                dataloader = torch.utils.data.DataLoader(**loader_kwargs)
                expected_num_samples = len(sampler) if sampler is not None else len(dataset)

                run_inference(
                    dataloader,
                    model,
                    model.device,
                    autocast_context,
                    unit,
                    cfg.model.batch_size,
                    tmp_feature_prefix,
                    expected_num_samples,
                    run_on_cpu,
                )

            except Exception as e:
                local_failed = True
                status_info = {
                    "status": "failed",
                    "error": str(e),
                    "traceback": str(traceback.format_exc()),
                }

            any_rank_failed = local_failed
            if not run_on_cpu:
                # Ensure every rank reaches sync points, even when one rank failed.
                torch.distributed.barrier()
                failure_flag = torch.tensor(
                    1 if local_failed else 0, device=model.device, dtype=torch.int32
                )
                torch.distributed.all_reduce(
                    failure_flag, op=torch.distributed.ReduceOp.MAX
                )
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
                    wsi_feature = load_features_with_indexed_fill(
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
                # Propagate post-processing failures from rank 0 to all ranks.
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
                else:
                    process_df.loc[
                        process_df["wsi_path"] == str(wsi_fp), "error"
                    ] = None
                    process_df.loc[
                        process_df["wsi_path"] == str(wsi_fp), "traceback"
                    ] = None
                process_df.to_csv(process_list, index=False)

        if distributed.is_enabled_and_multiple_gpus():
            torch.distributed.barrier()

        if distributed.is_main_process():
            # summary logging
            slides_with_tiles = len(tiled_df)
            total_slides = len(process_df)
            failed_feature_extraction = process_df[
                process_df["feature_status"] == "failed"
            ]
            print("=+=" * 10)
            print(f"Total number of slides with {unit}s: {slides_with_tiles}/{total_slides}")
            print(f"Failed {unit}-level feature extraction: {len(failed_feature_extraction)}/{slides_with_tiles}")
            print(
                f"Completed {unit}-level feature extraction: {slides_with_tiles - len(failed_feature_extraction)}/{slides_with_tiles}"
            )
            print("=+=" * 10)
            cleanup_tmp_feature_dir(tmp_dir)

        if distributed.is_enabled():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
