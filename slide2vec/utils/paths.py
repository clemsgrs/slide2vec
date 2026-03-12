from pathlib import Path


def resolve_output_dir(config_output_dir: str, cli_output_dir: str | None) -> Path:
    if cli_output_dir is None:
        return Path(config_output_dir)
    cli_path = Path(cli_output_dir)
    if cli_path.is_absolute():
        return cli_path
    return Path(config_output_dir, cli_output_dir)


def resolve_coordinates_dir(cfg) -> Path:
    read_coordinates_from = cfg.tiling.read_coordinates_from
    if read_coordinates_from:
        return Path(read_coordinates_from)
    return Path(cfg.output_dir, "coordinates")
