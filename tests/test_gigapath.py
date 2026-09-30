"""The GigaPath slide encoder must fetch its weights through the normal HF cache.

Ways this can fail:
- the download bypasses the HF cache (``local_dir``, ``cache_dir``) or re-downloads
  (``force_download``): ``HF_HOME`` is ignored or every load downloads 330 MB;
- a load with the checkpoint already cached still queries the Hub: a default
  ``hf_hub_download`` sends a metadata request on every call;
- ``create_model`` still receives an ``hf_hub:`` name and downloads on its own;
- ``create_model`` receives a path other than the downloaded file: it then silently
  keeps random weights.
"""

from __future__ import annotations

import sys
import types

from huggingface_hub.errors import LocalEntryNotFoundError

import slide2vec.encoders.models.gigapath as gigapath_module


def _stub_gigapath(monkeypatch, create_model_calls):
    def fake_create_model(pretrained, model_arch, in_chans, **kwargs):
        create_model_calls.append((pretrained, model_arch, in_chans, kwargs))
        return types.SimpleNamespace()

    fake_gigapath = types.ModuleType("gigapath")
    fake_slide_encoder = types.ModuleType("gigapath.slide_encoder")
    fake_slide_encoder.create_model = fake_create_model
    fake_gigapath.slide_encoder = fake_slide_encoder
    monkeypatch.setitem(sys.modules, "gigapath", fake_gigapath)
    monkeypatch.setitem(sys.modules, "gigapath.slide_encoder", fake_slide_encoder)


def test_slide_weights_are_downloaded_once_then_loaded_from_the_cache(monkeypatch, tmp_path):
    cached_path = str(tmp_path / "hub" / "snapshots" / "abc" / "slide_encoder.pth")
    cache = set()
    hub_requests = []
    create_model_calls = []

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        assert not {"local_dir", "cache_dir", "force_download"} & kwargs.keys()
        if kwargs.get("local_files_only"):
            if (repo_id, filename) not in cache:
                raise LocalEntryNotFoundError("not cached")
            return cached_path
        hub_requests.append((repo_id, filename))
        cache.add((repo_id, filename))
        return cached_path

    _stub_gigapath(monkeypatch, create_model_calls)
    monkeypatch.setattr(gigapath_module, "hf_hub_download", fake_hf_hub_download)

    gigapath_module.GigaPathSlideEncoder()
    assert hub_requests == [("prov-gigapath/prov-gigapath", "slide_encoder.pth")]

    gigapath_module.GigaPathSlideEncoder()
    assert len(hub_requests) == 1

    assert len(create_model_calls) == 2
    for pretrained, model_arch, in_chans, create_kwargs in create_model_calls:
        assert (pretrained, model_arch, in_chans) == (cached_path, "gigapath_slide_enc12l768d", 1536)
        assert "local_dir" not in create_kwargs
