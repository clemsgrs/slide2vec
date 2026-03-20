"""Tests for TarTileReader — the tar-based batch tile reader."""

import io
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from slide2vec.data.tile_store import TarTileReader  # noqa: direct import avoids cucim dep in __init__


def _create_test_tar(tar_path: Path, colors: list[tuple[int, int, int]], tile_size: int = 64):
    """Create a tar with solid-color JPEG tiles."""
    with tarfile.open(tar_path, "w") as tf:
        for i, color in enumerate(colors):
            img = Image.new("RGB", (tile_size, tile_size), color)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            info = tarfile.TarInfo(name=f"{i:06d}.jpg")
            info.size = buf.getbuffer().nbytes
            tf.addfile(info, buf)


class TestTarTileReader:
    def test_read_batch_returns_correct_shape(self, tmp_path: Path):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        tar_path = tmp_path / "tiles.tar"
        _create_test_tar(tar_path, colors, tile_size=64)

        reader = TarTileReader(tar_path, tile_size_px=64)
        batch = reader.read_batch(np.array([0, 1, 2], dtype=np.int64))

        assert batch.shape == (3, 3, 64, 64)
        assert batch.dtype == torch.uint8

    def test_read_batch_pixel_values_within_jpeg_tolerance(self, tmp_path: Path):
        tar_path = tmp_path / "tiles.tar"
        _create_test_tar(tar_path, [(200, 100, 50)], tile_size=32)

        reader = TarTileReader(tar_path, tile_size_px=32)
        batch = reader.read_batch(np.array([0], dtype=np.int64))

        # JPEG is lossy — check within tolerance
        r_mean = batch[0, 0].float().mean().item()
        g_mean = batch[0, 1].float().mean().item()
        b_mean = batch[0, 2].float().mean().item()
        assert abs(r_mean - 200) < 5
        assert abs(g_mean - 100) < 5
        assert abs(b_mean - 50) < 5

    def test_read_batch_subset_indices(self, tmp_path: Path):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        tar_path = tmp_path / "tiles.tar"
        _create_test_tar(tar_path, colors, tile_size=32)

        reader = TarTileReader(tar_path, tile_size_px=32)
        batch = reader.read_batch(np.array([2], dtype=np.int64))

        assert batch.shape == (1, 3, 32, 32)
        # Blue tile: channel 2 should dominate
        assert batch[0, 2].float().mean() > batch[0, 0].float().mean()

    def test_read_batch_empty_indices(self, tmp_path: Path):
        tar_path = tmp_path / "tiles.tar"
        _create_test_tar(tar_path, [(128, 128, 128)], tile_size=16)

        reader = TarTileReader(tar_path, tile_size_px=16)
        batch = reader.read_batch(np.array([], dtype=np.int64))

        assert batch.shape == (0, 3, 16, 16)
        assert batch.dtype == torch.uint8
