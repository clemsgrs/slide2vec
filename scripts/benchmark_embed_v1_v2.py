#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark slide2vec.embed v1 vs v2 using an existing tiling output "
            "(process_list.csv + coordinates)."
        )
    )
    parser.add_argument("--config-file", required=True, help="Path to embedding config YAML.")
    parser.add_argument(
        "--baseline-output-dir",
        required=True,
        help=(
            "Directory that already contains tiling artifacts: process_list.csv and coordinates/."
        ),
    )
    parser.add_argument(
        "--benchmark-output-dir",
        default="outputs/embed-benchmark",
        help="Directory where benchmark run folders and summary files are written.",
    )
    parser.add_argument(
        "--gpu-counts",
        default="1,4,8",
        help="Comma-separated GPU counts to benchmark (single-node only).",
    )
    parser.add_argument(
        "--pipelines",
        default="v1,v2",
        help="Comma-separated embedding pipelines to benchmark (v1,v2).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repetitions per (pipeline, gpu_count).",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to launch embed.py.",
    )
    parser.add_argument(
        "--extra-opt",
        action="append",
        default=[],
        help=(
            "Additional embed CLI overrides in path.key=value format. "
            "Repeat for multiple options."
        ),
    )
    parser.add_argument(
        "--run-on-cpu",
        action="store_true",
        help="Run all benchmark commands on CPU (for quick functional validation).",
    )
    return parser.parse_args()


def parse_csv_list(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def compute_total_tiles(process_df, coordinates_dir: Path):
    import numpy as np

    tiled_df = process_df[process_df["tiling_status"] == "success"]
    total_tiles = 0
    for wsi_path in tiled_df["wsi_path"].tolist():
        name = Path(wsi_path).stem.replace(" ", "_")
        coord_file = coordinates_dir / f"{name}.npy"
        if not coord_file.exists():
            continue
        arr = np.load(coord_file, allow_pickle=True)
        total_tiles += int(len(arr["x"]))
    return total_tiles, len(tiled_df)


def reset_process_list_for_embed(process_df):
    df = process_df.copy()
    if "feature_status" not in df.columns:
        df["feature_status"] = ["tbp"] * len(df)
    else:
        df["feature_status"] = ["tbp" if x == "success" else "tbp" for x in df["feature_status"]]
    if "error" in df.columns:
        df["error"] = ["" for _ in range(len(df))]
    if "traceback" in df.columns:
        df["traceback"] = ["" for _ in range(len(df))]
    return df


def build_command(
    *,
    python_exe: str,
    gpu_count: int,
    run_on_cpu: bool,
    config_file: Path,
    run_dir: Path,
    coords_dir: Path,
    pipeline: str,
    extra_opts: list[str],
):
    if gpu_count > 1 and not run_on_cpu:
        cmd = [
            python_exe,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={gpu_count}",
            "slide2vec/embed.py",
        ]
    else:
        cmd = [python_exe, "slide2vec/embed.py"]

    cmd.extend(
        [
            "--config-file",
            str(config_file.resolve()),
            "--output-dir",
            str(run_dir.resolve()),
        ]
    )
    if run_on_cpu:
        cmd.append("--run-on-cpu")

    opts = [
        f"tiling.read_coordinates_from={coords_dir.resolve()}",
        f"speed.embedding_pipeline={pipeline}",
        "speed.rank_sharding_mode=auto",
        "speed.log_perf_embedding=true",
    ]
    opts.extend(extra_opts)
    cmd.extend(opts)
    return cmd


def render_markdown_table(results):
    lines = []
    lines.append("| pipeline | gpus | repeat | wall_sec | tiles | tiles_per_sec | status |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in results:
        lines.append(
            "| {pipeline} | {gpu_count} | {repeat} | {wall_sec:.2f} | {tiles} | {tiles_per_sec:.2f} | {status} |".format(
                **row
            )
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    import pandas as pd

    config_file = Path(args.config_file)
    baseline_output_dir = Path(args.baseline_output_dir)
    benchmark_output_dir = Path(args.benchmark_output_dir)
    process_list = baseline_output_dir / "process_list.csv"
    coordinates_dir = baseline_output_dir / "coordinates"

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not process_list.exists():
        raise FileNotFoundError(f"process_list.csv not found: {process_list}")
    if not coordinates_dir.exists():
        raise FileNotFoundError(f"coordinates dir not found: {coordinates_dir}")

    benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    gpu_counts = [int(x) for x in parse_csv_list(args.gpu_counts)]
    pipelines = parse_csv_list(args.pipelines)
    if not pipelines:
        raise ValueError("No pipelines requested.")

    base_df = pd.read_csv(process_list)
    total_tiles, num_slides = compute_total_tiles(base_df, coordinates_dir)
    if total_tiles == 0:
        raise RuntimeError(
            "No tiles found in baseline coordinates. Ensure baseline output dir is a valid tiling output."
        )

    print(
        f"Benchmarking {pipelines} on GPU counts={gpu_counts}, repeats={args.repeats}. "
        f"Slides={num_slides}, total_tiles={total_tiles}."
    )

    results = []

    for pipeline in pipelines:
        if pipeline not in {"v1", "v2"}:
            raise ValueError(f"Unsupported pipeline: {pipeline}")
        for gpu_count in gpu_counts:
            for repeat in range(1, args.repeats + 1):
                run_name = f"{pipeline}-g{gpu_count}-r{repeat}"
                run_dir = benchmark_output_dir / run_name
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)

                run_process_df = reset_process_list_for_embed(base_df)
                run_process_df.to_csv(run_dir / "process_list.csv", index=False)

                cmd = build_command(
                    python_exe=args.python_exe,
                    gpu_count=gpu_count,
                    run_on_cpu=args.run_on_cpu,
                    config_file=config_file,
                    run_dir=run_dir,
                    coords_dir=coordinates_dir,
                    pipeline=pipeline,
                    extra_opts=args.extra_opt,
                )

                log_path = run_dir / "embed.log"
                print(f"\n[{run_name}] Running command:\n  {' '.join(cmd)}\n")

                start = time.perf_counter()
                with log_path.open("w", encoding="utf-8") as log_f:
                    proc = subprocess.run(
                        cmd,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        cwd=Path(__file__).resolve().parents[1],
                    )
                wall_sec = time.perf_counter() - start

                status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
                tiles_per_sec = total_tiles / wall_sec if proc.returncode == 0 else 0.0
                row = {
                    "pipeline": pipeline,
                    "gpu_count": gpu_count,
                    "repeat": repeat,
                    "wall_sec": wall_sec,
                    "tiles": total_tiles,
                    "tiles_per_sec": tiles_per_sec,
                    "status": status,
                    "log_path": str(log_path.resolve()),
                }
                results.append(row)

                print(
                    f"[{run_name}] status={status}, wall={wall_sec:.2f}s, "
                    f"tiles/sec={tiles_per_sec:.2f}, log={log_path}"
                )

    summary_json = benchmark_output_dir / "summary.json"
    summary_md = benchmark_output_dir / "summary.md"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    table = render_markdown_table(results)
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Embed v1 vs v2 Benchmark Summary\n\n")
        f.write(table)
        f.write("\n")

    print("\nBenchmark summary:")
    print(table)
    print(f"\nSaved summary JSON: {summary_json}")
    print(f"Saved summary Markdown: {summary_md}")


if __name__ == "__main__":
    main()
