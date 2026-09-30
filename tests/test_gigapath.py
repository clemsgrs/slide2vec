"""The GigaPath slide encoder must fetch its weights through the normal HF cache.

Ways this can fail:
- the download bypasses the HF cache (``local_dir``, ``cache_dir``) or re-downloads
  (``force_download``): ``HF_HOME`` is ignored or every load downloads 330 MB;
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

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        downloads.append((repo_id, filename, kwargs))
        return cached_path

    def fake_create_model(pretrained, model_arch, in_chans, **kwargs):
        create_model_calls.append((pretrained, model_arch, in_chans, kwargs))
        return types.SimpleNamespace()

    fake_gigapath = types.ModuleType("gigapath")
    fake_slide_encoder = types.ModuleType("gigapath.slide_encoder")
    fake_slide_encoder.create_model = fake_create_model
    fake_gigapath.slide_encoder = fake_slide_encoder
    monkeypatch.setitem(sys.modules, "gigapath", fake_gigapath)
    monkeypatch.setitem(sys.modules, "gigapath.slide_encoder", fake_slide_encoder)
    monkeypatch.setattr(gigapath_module, "hf_hub_download", fake_hf_hub_download)

    gigapath_module.GigaPathSlideEncoder()

    assert len(downloads) == 1
    repo_id, filename, download_kwargs = downloads[0]
    assert (repo_id, filename) == ("prov-gigapath/prov-gigapath", "slide_encoder.pth")
    assert not {"local_dir", "cache_dir", "force_download"} & download_kwargs.keys()

    assert len(create_model_calls) == 1
    pretrained, model_arch, in_chans, create_kwargs = create_model_calls[0]
    assert (pretrained, model_arch, in_chans) == (cached_path, "gigapath_slide_enc12l768d", 1536)
    assert "local_dir" not in create_kwargs
