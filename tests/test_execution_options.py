"""ExecutionOptions keeps what the caller asked for and resolves machine values lazily."""

import pytest

import slide2vec.api as api
from slide2vec.api import ExecutionOptions


def _machine(monkeypatch, *, gpus: int, cpus: int) -> None:
    monkeypatch.setattr(api, "_default_num_gpus", lambda: gpus)
    monkeypatch.setattr(api, "cpu_worker_limit", lambda: cpus)
    monkeypatch.setattr(api, "slurm_cpu_limit", lambda: None)


def test_auto_num_gpus_stays_none_and_resolves_to_the_detected_count(monkeypatch):
    _machine(monkeypatch, gpus=4, cpus=16)

    execution = ExecutionOptions(num_gpus=None)

    assert execution.num_gpus is None
    assert execution.resolved_num_gpus() == 4


def test_auto_preprocessing_workers_stay_none_and_resolve_to_the_cpu_budget(monkeypatch):
    _machine(monkeypatch, gpus=1, cpus=12)

    execution = ExecutionOptions(num_preprocessing_workers=None)

    assert execution.num_preprocessing_workers is None
    assert execution.resolved_num_preprocessing_workers() == 12


def test_explicit_values_are_kept_as_given_and_resolution_caps_the_tiling_workers(monkeypatch):
    _machine(monkeypatch, gpus=8, cpus=6)

    execution = ExecutionOptions(num_gpus=2, num_preprocessing_workers=32)

    assert execution.num_gpus == 2
    assert execution.num_preprocessing_workers == 32
    assert execution.resolved_num_gpus() == 2
    assert execution.resolved_num_preprocessing_workers() == 6


@pytest.mark.parametrize("num_gpus", [0, -1])
def test_explicit_num_gpus_below_one_is_rejected(num_gpus):
    with pytest.raises(ValueError, match="num_gpus must be at least 1"):
        ExecutionOptions(num_gpus=num_gpus)


@pytest.mark.parametrize("workers", [-1, -8])
def test_a_negative_preprocessing_worker_count_is_rejected(workers):
    with pytest.raises(ValueError, match="num_preprocessing_workers must be non-negative"):
        ExecutionOptions(num_preprocessing_workers=workers)


def test_zero_preprocessing_workers_is_a_valid_request(monkeypatch):
    _machine(monkeypatch, gpus=1, cpus=6)

    execution = ExecutionOptions(num_preprocessing_workers=0)

    assert execution.num_preprocessing_workers == 0
    assert execution.resolved_num_preprocessing_workers() == 0


def test_options_built_on_different_machines_compare_equal(monkeypatch):
    _machine(monkeypatch, gpus=8, cpus=64)
    on_a_gpu_node = ExecutionOptions(batch_size=64)
    assert on_a_gpu_node.resolved_num_gpus() == 8
    assert on_a_gpu_node.resolved_num_preprocessing_workers() == 64

    _machine(monkeypatch, gpus=1, cpus=4)
    on_a_ci_runner = ExecutionOptions(batch_size=64)
    assert on_a_ci_runner.resolved_num_gpus() == 1
    assert on_a_ci_runner.resolved_num_preprocessing_workers() == 4

    assert on_a_gpu_node == on_a_ci_runner
    assert hash(on_a_gpu_node) == hash(on_a_ci_runner)
    assert repr(on_a_gpu_node) == repr(on_a_ci_runner)


def test_resolution_happens_once_on_first_use_and_logs_once(monkeypatch, caplog):
    _machine(monkeypatch, gpus=2, cpus=16)
    caplog.set_level("INFO", logger="slide2vec.api")

    execution = ExecutionOptions()
    assert not [record for record in caplog.records if "ExecutionOptions:" in record.getMessage()]

    assert execution.resolved_num_gpus() == 2
    _machine(monkeypatch, gpus=4, cpus=32)
    assert execution.resolved_num_gpus() == 2
    assert execution.resolved_num_preprocessing_workers() == 16
    execution.resolved_num_workers_per_gpu()

    messages = [record.getMessage() for record in caplog.records if "ExecutionOptions:" in record.getMessage()]
    assert len(messages) == 1
    assert messages[0].startswith(
        "ExecutionOptions: num_gpus=2 (requested=auto), num_workers_per_gpu=8 (requested=auto), "
        "num_preprocessing_workers=16 (preprocessing cap=16 via "
    )


def test_a_pickled_copy_compares_equal_and_resolves_on_the_machine_that_loads_it(monkeypatch):
    import pickle

    _machine(monkeypatch, gpus=8, cpus=64)
    original = ExecutionOptions(num_workers_per_gpu=2)
    assert original.resolved_num_gpus() == 8

    _machine(monkeypatch, gpus=1, cpus=4)
    restored = pickle.loads(pickle.dumps(original))

    assert restored == original
    assert restored.num_gpus is None
    assert restored.num_preprocessing_workers is None
    assert restored.resolved_num_gpus() == 1
    assert restored.resolved_num_preprocessing_workers() == 4
