import os
import sys
import time
import wandb
import socket
import signal
import argparse
import threading
import subprocess

from pathlib import Path

from slide2vec.utils.config import setup, hf_login


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("slide2vec", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    return parser


def log_progress(features_dir: Path, total_slides: int, log_interval: int = 10):
    while True:
        if not features_dir.exists():
            time.sleep(log_interval)
            continue
        num_files = len(list(features_dir.glob("*.pt")))
        wandb.log({"processed": num_files})
        if num_files >= total_slides:
            break
        time.sleep(log_interval)


def run_tiling(config_file, run_id):
    print("Running tiling.py...")
    cmd = [
        sys.executable,
        "slide2vec/tiling.py",
        "--run-id",
        run_id,
        "--config-file",
        config_file,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Slide tiling failed. Exiting.")
        sys.exit(result.returncode)


def run_feature_extraction(config_file, run_id):
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
        "slide2vec/embed.py",
        "--run-id",
        run_id,
        "--config-file",
        config_file,
    ]
    # launch in its own process group.
    proc = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,
        text=True,
    )
    try:
        proc.communicate()
    except KeyboardInterrupt:
        print("Received CTRL+C, terminating embed.py process group...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        sys.exit(1)
    if proc.returncode != 0:
        print("Slide embedding failed. Exiting.")
        sys.exit(proc.returncode)


def main(args):
    config_file = args.config_file

    cfg = setup(config_file)
    hf_login()

    output_dir = Path(cfg.output_dir)
    run_id = output_dir.stem

    run_tiling(config_file, run_id)

    coordinates_dir = output_dir / "coordinates"
    total_slides = len(list(coordinates_dir.glob("*.npy")))

    features_dir = output_dir / "features"
    if cfg.wandb.enable:
        log_thread = threading.Thread(
            target=log_progress, args=(features_dir, total_slides), daemon=True
        )
        log_thread.start()

    run_feature_extraction(config_file, run_id)

    if cfg.wandb.enable:
        log_thread.join()
    print("Feature extraction and logging complete.")


if __name__ == "__main__":
    import warnings
    import torchvision

    warnings.filterwarnings("ignore", message=".*Could not set the permissions.*")
    warnings.filterwarnings("ignore", message=".*antialias.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*TypedStorage.*", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    torchvision.disable_beta_transforms_warning()

    args = get_args_parser(add_help=True).parse_args()
    main(args)
