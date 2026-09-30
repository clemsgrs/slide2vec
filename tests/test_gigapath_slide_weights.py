"""The GigaPath slide encoder must fetch its weights through the normal HF cache.

``gigapath.slide_encoder.create_model`` given an ``hf_hub:`` name downloads
``slide_encoder.pth`` into ``~/.cache/`` with ``force_download=True``: it ignores
``HF_HOME`` and downloads again on every load. slide2vec downloads the file itself
and hands ``create_model`` the local path, which skips that download.

Ways this can fail:
- the download bypasses the HF cache (``local_dir``) or re-downloads (``force_download``);
- ``create_model`` still receives an ``hf_hub:`` name and downloads on its own;
- ``create_model`` receives a path other than the downloaded file: it then silently
  keeps random weights.
"""

from __future__ import annotations

import sys
import types

import slide2vec.encoders.models.gigapath as gigapath_module


def test_slide_weights_are_downloaded_through_the_hf_cache(monkeypatch, tmp_path):
    cached_path = str(tmp_path / "hub" / "snapshots" / "abc" / "slide_encoder.pth")
    downloads = []
    create_model_calls = []

    def fake_hf_hub_download(*args, **kwargs):
        downloads.append((args, kwargs))
        return cached_path

    def fake_create_model(*args, **kwargs):
        create_model_calls.append((args, kwargs))
        return types.SimpleNamespace()

    fake_gigapath = types.ModuleType("gigapath")
    fake_slide_encoder = types.ModuleType("gigapath.slide_encoder")
    fake_slide_encoder.create_model = fake_create_model
    fake_gigapath.slide_encoder = fake_slide_encoder
    monkeypatch.setitem(sys.modules, "gigapath", fake_gigapath)
    monkeypatch.setitem(sys.modules, "gigapath.slide_encoder", fake_slide_encoder)
    monkeypatch.setattr(gigapath_module, "hf_hub_download", fake_hf_hub_download)

    gigapath_module.GigaPathSlideEncoder()

    assert downloads == [
        ((), {"repo_id": "prov-gigapath/prov-gigapath", "filename": "slide_encoder.pth"})
    ]
    assert len(create_model_calls) == 1
    args, kwargs = create_model_calls[0]
    assert args == (cached_path, "gigapath_slide_enc12l768d", 1536)
    assert kwargs == {}
