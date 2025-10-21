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

from slide2vec.utils.config import hf_login, setup


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("slide2vec", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--skip-datetime", action="store_true", help="skip run id datetime prefix"
    )
    parser.add_argument(
        "--run-on-cpu", action="store_true", help="run inference on cpu"
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
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    # forward output in real-time
    for line in proc.stdout:
        print(line.rstrip())
        sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        print("Slide tiling failed. Exiting.")
        sys.exit(proc.returncode)


def run_feature_extraction(config_file, run_id, run_on_cpu: False):
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
    if run_on_cpu:
        cmd = [
            sys.executable,
            "slide2vec/embed.py",
            "--run-id",
            run_id,
            "--config-file",
            config_file,
            "--run-on-cpu",
        ]
    # launch in its own process group.
    proc = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    try:
        # forward output in real-time
        for line in proc.stdout:
            print(line.rstrip())
            sys.stdout.flush()
        proc.wait()
    except KeyboardInterrupt:
        print("Received CTRL+C, terminating embed.py process group...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        sys.exit(1)
    if proc.returncode != 0:
        print("Feature extraction failed. Exiting.")
        sys.exit(proc.returncode)


def run_feature_aggregation(config_file, run_id, run_on_cpu: False):
    print("Running aggregate.py...")
    # find a free port
    cmd = [
        sys.executable,
        "slide2vec/aggregate.py",
        "--run-id",
        run_id,
        "--config-file",
        config_file,
    ]
    if run_on_cpu:
        cmd.append("--run-on-cpu")
    # launch in its own process group.
    proc = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    try:
        # forward output in real-time
        for line in proc.stdout:
            print(line.rstrip())
            sys.stdout.flush()
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
    config_file = args.config_file
    skip_datetime = args.skip_datetime
    run_on_cpu = args.run_on_cpu

    cfg, run_id = setup(config_file, skip_datetime=skip_datetime)
    hf_login()

    run_tiling(config_file, run_id)

    print("Tiling completed.")
    print("=+=" * 10)

    output_dir = Path(cfg.output_dir)
    features_dir = output_dir / "features"
    if cfg.wandb.enable:
        stop_event = threading.Event()
        log_thread = threading.Thread(
            target=log_progress, args=(features_dir, stop_event), daemon=True
        )
        log_thread.start()

    run_feature_extraction(config_file, run_id, run_on_cpu)

    if cfg.model.level == "slide":
        run_feature_aggregation(config_file, run_id, run_on_cpu)
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
