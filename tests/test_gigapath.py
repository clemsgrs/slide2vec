"""The GigaPath slide encoder must fetch its weights through the normal HF cache
and aggregate in inference mode.

Ways this can fail:
- the download bypasses the HF cache (``local_dir``, ``cache_dir``) or re-downloads
  (``force_download``): ``HF_HOME`` is ignored or every load downloads 330 MB;
- a load with the checkpoint already cached still queries the Hub: a default
  ``hf_hub_download`` sends a metadata request on every call;
- ``create_model`` still receives an ``hf_hub:`` name and downloads on its own;
- ``create_model`` receives a path other than the downloaded file: it then silently
  keeps random weights;
- the slide model stays in the training mode ``create_model`` returns it in: dropout
  is active and two aggregations of the same tiles differ.
"""

from __future__ import annotations

import sys
import types

import torch
from huggingface_hub.errors import LocalEntryNotFoundError

import slide2vec.encoders.models.gigapath as gigapath_module


class _DropoutSlideModel(torch.nn.Module):
    """Stand-in for ``gigapath_slide_enc12l768d``: dropout, and training mode on creation."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1536, 768)
        self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, tile_features, coordinates):
        return [self.dropout(self.proj(tile_features)).mean(dim=1)]


def _stub_gigapath(monkeypatch, create_model_calls):
    def fake_create_model(pretrained, model_arch, in_chans, **kwargs):
        create_model_calls.append((pretrained, model_arch, in_chans, kwargs))
        return _DropoutSlideModel()

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


def test_slide_encoder_aggregates_in_eval_mode(monkeypatch):
    _stub_gigapath(monkeypatch, [])
    monkeypatch.setattr(
        gigapath_module, "hf_hub_download", lambda repo_id, filename, **kwargs: "slide_encoder.pth"
    )

    encoder = gigapath_module.GigaPathSlideEncoder()
    assert encoder._model.training is False
    encoder.to("cpu")
    assert encoder._model.training is False

    generator = torch.Generator().manual_seed(0)
    tile_features = torch.randn(32, 1536, generator=generator)
    coordinates = torch.randint(0, 4096, (32, 2), generator=generator)
    first = encoder.encode_slide(tile_features, coordinates)
    second = encoder.encode_slide(tile_features, coordinates)
    assert first.shape == (768,)
    assert torch.equal(first, second)
