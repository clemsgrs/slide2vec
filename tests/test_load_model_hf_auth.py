"""``load_model`` must never call ``huggingface_hub.login()``.

Under distributed extraction every rank re-runs ``load_model`` per chunk. ``login()``
truncates and re-reads a shared ``stored_tokens`` file with no lock, so concurrent
ranks race and crash with ``ValueError: Token ... not found``. The login was also
redundant: ``huggingface_hub.get_token()`` resolves ``HF_TOKEN`` from the environment
ahead of any stored file, which is the only thing the encoders rely on.
"""

from __future__ import annotations

import os

import huggingface_hub
import pytest
import torch
from huggingface_hub import constants as hf_constants
from torchvision.transforms import v2

import slide2vec.inference as inference
from slide2vec.runtime.encoder_input_contract import EncoderInputContract


class _StandInTileEncoder:
    """A patch-14 tile encoder that loads no weights and touches no network."""

    def __init__(self, *, output_variant=None):
        self.device = torch.device("cpu")
        self.encode_dim = 8

    @property
    def patch_size(self):
        return (14, 14)

    def get_transform(self):
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def to(self, device):
        self.device = torch.device(device)
        return self


@pytest.fixture
def hf_auth_sandbox(monkeypatch, tmp_path):
    """Stub the encoder registry and point the hub's token files at a tmp dir.

    Returns the sandboxed ``stored_tokens`` path. ``HF_TOKEN`` starts unset.
    """
    monkeypatch.setattr(inference.encoder_registry, "require", lambda name: _StandInTileEncoder)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    token_path = tmp_path / "token"
    stored_tokens_path = tmp_path / "stored_tokens"
    monkeypatch.setenv("HF_TOKEN_PATH", str(token_path))
    monkeypatch.setenv("HF_STORED_TOKENS_PATH", str(stored_tokens_path))
    # ``huggingface_hub.constants`` reads those env vars at import time, so repoint the
    # already-imported module constants too.
    monkeypatch.setattr(hf_constants, "HF_TOKEN_PATH", str(token_path))
    monkeypatch.setattr(hf_constants, "HF_STORED_TOKENS_PATH", str(stored_tokens_path))
    return stored_tokens_path


@pytest.fixture
def login_calls(hf_auth_sandbox, monkeypatch):
    """Replace ``huggingface_hub.login`` with a recorder; returns the call list."""
    calls: list[dict] = []

    def _record_login(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(huggingface_hub, "login", _record_login)
    return calls


def _load_stand_in(**kwargs):
    return inference.load_model(
        name="virchow2", encoder_input=EncoderInputContract.given(), device="cpu", **kwargs
    )


def test_load_model_with_hf_token_env_never_logs_in(login_calls, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")

    _load_stand_in()

    assert login_calls == []
    assert huggingface_hub.get_token() == "hf_from_env"


def test_load_model_with_explicit_token_exports_it_without_logging_in(login_calls):
    _load_stand_in(token="hf_explicit")

    assert login_calls == []
    assert os.environ["HF_TOKEN"] == "hf_explicit"
    assert huggingface_hub.get_token() == "hf_explicit"


def test_repeated_load_model_calls_leave_no_stored_tokens_file(hf_auth_sandbox, monkeypatch):
    """Nothing touches disk: neither ``HF_TOKEN`` nor ``token=`` writes a token file.

    Uses the real ``huggingface_hub.login`` so a regression that reintroduces the call
    would actually write the sandboxed ``stored_tokens``.
    """
    stored_tokens_path = hf_auth_sandbox
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")

    for _ in range(3):
        _load_stand_in()
    for _ in range(3):
        _load_stand_in(token="hf_explicit")

    assert not stored_tokens_path.exists()
    assert not (stored_tokens_path.parent / "token").exists()
