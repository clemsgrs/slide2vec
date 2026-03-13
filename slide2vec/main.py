import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import wandb
from hs2p import tile_slides

from slide2vec.utils.config import hf_login, setup
from slide2vec.utils.tiling_io import build_tiling_configs, load_slide_manifest


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("slide2vec", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--skip-datetime", action="store_true", help="skip run id datetime prefix"
    )
    parser.add_argument(
        "--tiling-only", action="store_true", help="only run slide tiling"
    )
    parser.add_argument(
        "--run-on-cpu", action="store_true", help="run inference on cpu"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="output directory to save logs and checkpoints",
    )
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command using \"path.key=value\".",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def log_progress(features_dir: Path, stop_event: threading.Event, log_interval: int = 10):
    while not stop_event.is_set():
        if not features_dir.exists():
            time.sleep(log_interval)
            continue
        num_files = len(list(features_dir.glob("*.pt")))
        wandb.log({"processed": num_files})
        time.sleep(log_interval)


def run_tiling(cfg, output_dir: Path):
    print("Running tiling with hs2p...")
    whole_slides = load_slide_manifest(cfg.csv)
    tiling, segmentation, filtering, qc = build_tiling_configs(cfg)
    tile_slides(
        whole_slides,
        tiling=tiling,
        segmentation=segmentation,
        filtering=filtering,
        qc=qc,
        output_dir=output_dir,
        num_workers=cfg.speed.num_workers,
        resume=cfg.resume,
        read_tiles_from=Path(cfg.tiling.read_tiles_from) if cfg.tiling.read_tiles_from else None,
    )


def run_feature_extraction(config_file, output_dir, run_on_cpu: False):
    print("Running embed.py...")
    # find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        free_port = s.getsockname()[1]
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--master_port={free_port}",
        "--nproc_per_node=gpu",
        "-m",
        "slide2vec.embed",
        "--config-file",
        os.path.abspath(config_file),
        "--output-dir",
        os.path.abspath(output_dir),
    ]
    if run_on_cpu:
        cmd = [
            sys.executable,
            "-m",
            "slide2vec.embed",
            "--config-file",
            os.path.abspath(config_file),
            "--output-dir",
            os.path.abspath(output_dir),
            "--run-on-cpu",
        ]
    # launch in its own process group.
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("Received CTRL+C, terminating embed.py process group...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        sys.exit(1)
    if proc.returncode != 0:
        print("Feature extraction failed. Exiting.")
        sys.exit(proc.returncode)


def run_feature_aggregation(config_file, output_dir, run_on_cpu: False):
    print("Running aggregate.py...")
    cmd = [
        sys.executable,
        "-m",
        "slide2vec.aggregate",
        "--config-file",
        os.path.abspath(config_file),
        "--output-dir",
        os.path.abspath(output_dir),
    ]
    if run_on_cpu:
        cmd.append("--run-on-cpu")
    # launch in its own process group.
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("Received CTRL+C, terminating aggregate.py process group...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        sys.exit(1)
    if proc.returncode != 0:
        print("Feature aggregation failed. Exiting.")
        sys.exit(proc.returncode)


def main(args):
    run_on_cpu = args.run_on_cpu

    cfg, cfg_path = setup(args)
    output_dir = Path(cfg.output_dir)

    hf_login()

    run_tiling(cfg, output_dir)

    print("Tiling completed.")
    print("=+=" * 10)

    if args.tiling_only:
        print("Tiling only flag set, exiting.")
        return

    features_dir = output_dir / "features"
    if cfg.wandb.enable:
        stop_event = threading.Event()
        log_thread = threading.Thread(
            target=log_progress, args=(features_dir, stop_event), daemon=True
        )
        log_thread.start()

    run_feature_extraction(cfg_path, output_dir, run_on_cpu)

    if cfg.model.level == "slide":
        run_feature_aggregation(cfg_path, output_dir, run_on_cpu)
        print("Feature extraction completed.")
        print("=+=" * 10)
    else:
        print("Feature extraction completed.")
        print("=+=" * 10)


    if cfg.wandb.enable:
        stop_event.set()
        log_thread.join()

    print("All tasks finished successfully.")
    print("=+=" * 10)


if __name__ == "__main__":

    import warnings
    import torchvision

    torchvision.disable_beta_transforms_warning()

    warnings.filterwarnings("ignore", message=".*Could not set the permissions.*")
    warnings.filterwarnings("ignore", message=".*antialias.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*TypedStorage.*", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
    warnings.filterwarnings(
        "ignore",
        message=".*'frozen' attribute with value True was provided to the `Field`.*"
    )

    args = get_args_parser(add_help=True).parse_args()
    main(args)
