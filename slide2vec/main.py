import subprocess
import sys
import typer
import logging
from pathlib import Path

from slide2vec.utils.config import setup, hf_login

app = typer.Typer(invoke_without_command=True)
logger = logging.getLogger("slide2vec")

# global variable to hold the run_id, which is computed from the config's output_dir.
CONFIG_FILE = None
RUN_ID = None


@app.callback()
def main(
    ctx: typer.Context,
    config_file: str = typer.Option(
        ..., "--config-file", help="path to yaml config file"
    ),
):
    """
    Global callback: This grabs the config file path, runs setup,
    performs HF login, writes out the config, and computes the run ID.
    """
    global CONFIG_FILE, RUN_ID
    CONFIG_FILE = config_file

    cfg = setup(CONFIG_FILE)
    hf_login()
    RUN_ID = Path(cfg.output_dir).stem

    # if no subcommand is provided, default to running run_all().
    if ctx.invoked_subcommand is None:
        run_all()


@app.command()
def tiling():
    """Run slide tiling."""
    typer.echo("Running tiling.py...")
    global CONFIG_FILE, RUN_ID
    cmd = [
        sys.executable,
        "slide2vec/tiling.py",
        "--run-id",
        RUN_ID,
        "--config-file",
        CONFIG_FILE,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        typer.echo("Slide tiling failed. Exiting.")
        sys.exit(result.returncode)


@app.command()
def feature_extraction():
    """Run slide embedding."""
    typer.echo("Running embed.py...")
    global CONFIG_FILE, RUN_ID
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=gpu",
        "slide2vec/embed.py",
        "--run-id",
        RUN_ID,
        "--config-file",
        CONFIG_FILE,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        typer.echo("Slide embedding failed. Exiting.")
        sys.exit(result.returncode)


@app.command()
def run_all():
    """Chain tiling and feature extraction."""
    typer.echo("")
    tiling()
    typer.echo("")
    feature_extraction()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message=".*Could not set the permissions.*")
    warnings.filterwarnings("ignore", message=".*antialias.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*TypedStorage.*", category=UserWarning)

    app()
