"""Recipe-aware resume for dense image artifacts (issue #257)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slide2vec.api import DenseImageOptions, ExecutionOptions, ImageSpec, Model
from slide2vec.artifacts import dense_image_paths
from slide2vec.runtime.dense_image_stage import partition_dense_images_by_resume
from slide2vec.runtime.dense_image_recipe import resolve_dense_image_recipe
from slide2vec.runtime.dense_image_reading import DenseImageReadPlan
from slide2vec.runtime.image_specs import (
    build_image_specs_request,
    image_specs_from_request,
)
from slide2vec.runtime.serialization import (
    deserialize_dense_image_recipe,
    serialize_dense_image_recipe,
)


def _resolved_recipe(tmp_path, **dense_kwargs):
    model = Model(name="virchow2")
    dense = DenseImageOptions(**{"target_size": 224, **dense_kwargs})
    return resolve_dense_image_recipe(
        model=model,
        contract=model._declare_dense_encoder_input(dense, emit_run_info=False),
        dense=dense,
        execution=ExecutionOptions(output_dir=tmp_path, precision="fp32"),
    )


def _publish_pair(out_dir: Path, spec: ImageSpec, compatibility: dict) -> None:
    payload_path, sidecar_path = dense_image_paths(out_dir, sample_id=spec.sample_id)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(b"complete payload")
    sidecar_path.write_text(
        json.dumps(
            {
                "feature_dim": 8,
                "grid_shape": [2, 2],
                "compatibility": compatibility,
            }
        ),
        encoding="utf-8",
    )


def test_dense_image_recipe_contains_the_complete_canonical_identity(tmp_path):
    model = Model(name="virchow2", output_variant="cls")
    dense = DenseImageOptions(
        target_size=(60, 92),
        spacing_um=0.504,
        pad_mode="constant",
        image_pad_value=0.25,
        window_size=31,
        overlap=0.25,
        feature_kind="cls_attention",
        attention_blocks=(-2, -1),
        attention_include_registers=True,
    )
    contract = model._declare_dense_encoder_input(dense, emit_run_info=False)

    first = resolve_dense_image_recipe(
        model=model,
        contract=contract,
        dense=dense,
        execution=ExecutionOptions(
            output_dir=tmp_path / "one",
            precision="bf16",
            output_dtype="fp16",
            num_gpus=1,
            batch_size=2,
            num_workers_per_gpu=0,
            prefetch_factor=1,
        ),
    )
    assert first.to_dict() == {
        "encoder_name": "virchow2",
        "output_variant": "cls",
        "reader_regime": "raster",
        "spacing_source": "explicit",
        "declared_spacing_um": 0.504,
        "source_spacing_um": 0.504,
        "effective_spacing_um": 0.504,
        "requested_backend": "auto",
        "backend": "pil",
        "tolerance": None,
        "read_level": None,
        "read_tile_size_px": None,
        "requested_tile_size_px": None,
        "target_size": [60, 92],
        "patch_size": [14, 14],
        "encoded_size": [70, 98],
        "pad": [10, 6],
        "grid_shape": [5, 7],
        "pad_mode": "constant",
        "image_pad_value": 0.25,
        "window_size": 31,
        "overlap": 0.25,
        "feature_kind": "cls_attention",
        "attention_blocks": [-2, -1],
        "attention_include_registers": True,
        "precision": "bf16",
        "dtype": "float16",
    }


def test_dense_image_recipe_excludes_execution_mechanics(tmp_path):
    model = Model(name="virchow2")
    dense = DenseImageOptions(target_size=224)
    contract = model._declare_dense_encoder_input(dense, emit_run_info=False)
    first = resolve_dense_image_recipe(
        model=model,
        contract=contract,
        dense=dense,
        execution=ExecutionOptions(
            output_dir=tmp_path / "one",
            precision="fp32",
            num_gpus=1,
            batch_size=2,
            num_workers_per_gpu=0,
            prefetch_factor=1,
        ),
    )
    second = resolve_dense_image_recipe(
        model=model,
        contract=contract,
        dense=dense,
        execution=ExecutionOptions(
            output_dir=tmp_path / "two",
            precision="fp32",
            num_gpus=8,
            batch_size=128,
            num_workers_per_gpu=12,
            prefetch_factor=16,
        ),
    )

    assert first == second


def test_dense_image_recipe_round_trips_exactly_through_json(tmp_path):
    model = Model(name="virchow2")
    dense = DenseImageOptions(target_size=(60, 92), window_size=31, overlap=0.25)
    recipe = resolve_dense_image_recipe(
        model=model,
        contract=model._declare_dense_encoder_input(dense, emit_run_info=False),
        dense=dense,
        execution=ExecutionOptions(
            output_dir=tmp_path, precision="fp16", output_dtype="fp32"
        ),
    )

    payload = json.loads(json.dumps(serialize_dense_image_recipe(recipe)))

    assert deserialize_dense_image_recipe(payload) == recipe


def test_unknown_raster_spacing_recipe_records_null_physical_scale(tmp_path):
    recipe = _resolved_recipe(tmp_path, spacing_um=None)

    assert {
        key: recipe.to_dict()[key]
        for key in (
            "reader_regime",
            "spacing_source",
            "declared_spacing_um",
            "source_spacing_um",
            "effective_spacing_um",
        )
    } == {
        "reader_regime": "raster",
        "spacing_source": "unknown",
        "declared_spacing_um": None,
        "source_spacing_um": None,
        "effective_spacing_um": None,
    }


def test_current_image_specs_round_trip_exactly_through_json(tmp_path):
    specs = [
        ImageSpec(
            sample_id="sample-1",
            image_path=str((tmp_path / "images" / "sample-1.png").resolve()),
            spacing_at_level_0=0.252,
        )
    ]
    payload = json.loads(json.dumps(build_image_specs_request(specs)))

    assert image_specs_from_request(payload) == specs


def test_resume_skips_only_a_complete_exactly_compatible_pair(tmp_path):
    source = (tmp_path / "source.png").resolve()
    spec = ImageSpec(sample_id="sample-1", image_path=str(source))
    recipe = _resolved_recipe(tmp_path)
    _publish_pair(tmp_path / "out", spec, recipe.for_image(spec))

    remaining, skipped = partition_dense_images_by_resume(
        [spec], tmp_path / "out", recipe
    )

    assert remaining == []
    assert skipped == 1


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("source_spacing_um", 0.26),
        ("backend", "openslide"),
    ],
)
def test_resume_recomputes_when_parent_resolved_source_or_auto_backend_changes(
    tmp_path, changed_field, changed_value
):
    from dataclasses import replace

    source = (tmp_path / "source.svs").resolve()
    spec = ImageSpec(sample_id="sample-1", image_path=str(source))
    recipe = _resolved_recipe(tmp_path)
    original = DenseImageReadPlan(
        reader_regime="spacing-readable",
        spacing_source="explicit",
        declared_spacing_um=0.5,
        source_spacing_um=0.25,
        spacing_at_level_0=None,
        read_spacing_um=0.25,
        effective_spacing_um=0.5,
        requested_backend="auto",
        backend="vips",
        tolerance=0.05,
        read_level=0,
        is_within_tolerance=False,
        read_size=(448, 448),
        output_size=(224, 224),
    )
    _publish_pair(
        tmp_path / "out",
        spec,
        recipe.for_image(spec, original),
    )
    current = replace(original, **{changed_field: changed_value})

    remaining, skipped = partition_dense_images_by_resume(
        [spec],
        tmp_path / "out",
        recipe,
        read_plans={spec.sample_id: current},
    )

    assert remaining == [spec]
    assert skipped == 0


@pytest.mark.parametrize(
    ("condition", "differing_fields"),
    [
        ("missing-pair", "payload, sidecar"),
        ("missing-payload", "payload"),
        ("missing-sidecar", "sidecar"),
        ("legacy", "compatibility"),
        ("incomplete", "patch_size"),
        ("mismatch", "overlap"),
    ],
)
def test_noncompatible_pairs_recompute_and_log_fields(
    tmp_path, caplog, condition, differing_fields
):
    out_dir = tmp_path / "out"
    recipe = _resolved_recipe(tmp_path)
    spec = ImageSpec(
        sample_id=condition,
        image_path=str((tmp_path / "images" / f"{condition}.png").resolve()),
    )
    if condition != "missing-pair":
        recorded = recipe.for_image(spec)
        if condition == "legacy":
            _publish_pair(out_dir, spec, recorded)
            sidecar = dense_image_paths(out_dir, sample_id=spec.sample_id)[1]
            sidecar.write_text(
                json.dumps({"feature_dim": 8, "grid_shape": [2, 2]}),
                encoding="utf-8",
            )
        else:
            if condition == "incomplete":
                recorded.pop("patch_size")
            elif condition == "mismatch":
                recorded["overlap"] = 0.5
            _publish_pair(out_dir, spec, recorded)
            if condition == "missing-payload":
                dense_image_paths(out_dir, sample_id=spec.sample_id)[0].unlink()
            elif condition == "missing-sidecar":
                dense_image_paths(out_dir, sample_id=spec.sample_id)[1].unlink()

    with caplog.at_level("INFO", logger="slide2vec.runtime.dense_image_stage"):
        remaining, skipped = partition_dense_images_by_resume([spec], out_dir, recipe)

    assert remaining == [spec]
    assert skipped == 0
    assert (
        f"'{condition}'; differing fields: {differing_fields}" in caplog.text
    )


@pytest.mark.parametrize(
    ("condition", "error_type", "message"),
    [
        ("unreadable-json", json.JSONDecodeError, None),
        ("non-object-sidecar", ValueError, "expected a JSON object"),
        ("non-object-compatibility", ValueError, "'compatibility' must be a JSON object"),
        ("malformed-field", ValueError, "target_size"),
        ("non-integral-window", ValueError, "window_size"),
    ],
)
def test_unreadable_or_malformed_sidecar_hard_fails(
    tmp_path, condition, error_type, message
):
    out_dir = tmp_path / "out"
    spec = ImageSpec(
        sample_id="sample-1", image_path=str((tmp_path / "source.png").resolve())
    )
    payload_path, sidecar_path = dense_image_paths(out_dir, sample_id=spec.sample_id)
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(b"complete payload")
    recipe = _resolved_recipe(
        tmp_path, window_size=32 if condition == "non-integral-window" else None
    )
    if condition == "unreadable-json":
        text = "{"
    elif condition == "non-object-sidecar":
        text = "[]"
    elif condition == "non-object-compatibility":
        text = json.dumps({"compatibility": []})
    else:
        malformed = recipe.for_image(spec)
        if condition == "non-integral-window":
            malformed["window_size"] = 32.5
        else:
            malformed["target_size"] = "224x224"
        text = json.dumps({"compatibility": malformed})
    sidecar_path.write_text(text, encoding="utf-8")

    with pytest.raises(error_type, match=message):
        partition_dense_images_by_resume(
            [spec], out_dir, recipe
        )


def test_every_recorded_compatibility_field_participates_in_resume(tmp_path):
    recipe = _resolved_recipe(
        tmp_path,
        target_size=(224, 238),
        pad_mode="constant",
        image_pad_value=0.25,
        window_size=112,
        overlap=0.25,
        feature_kind="cls_attention",
        attention_blocks=(-2, -1),
        attention_include_registers=True,
    )
    fields = tuple(recipe.for_image(
        ImageSpec(sample_id="template", image_path=tmp_path / "template.png")
    ))
    assert fields == (
        "sample_id",
        "image_path",
        "encoder_name",
        "output_variant",
        "reader_regime",
        "spacing_source",
        "declared_spacing_um",
        "source_spacing_um",
        "spacing_at_level_0",
        "read_spacing_um",
        "effective_spacing_um",
        "requested_backend",
        "backend",
        "tolerance",
        "read_level",
        "is_within_tolerance",
        "read_size",
        "output_size",
        "read_tile_size_px",
        "requested_tile_size_px",
        "target_size",
        "patch_size",
        "encoded_size",
        "pad",
        "grid_shape",
        "pad_mode",
        "image_pad_value",
        "window_size",
        "overlap",
        "feature_kind",
        "attention_blocks",
        "attention_include_registers",
        "precision",
        "dtype",
    )

    for field in fields:
        spec = ImageSpec(
            sample_id=field,
            image_path=str((tmp_path / "images" / f"{field}.png").resolve()),
        )
        recorded = recipe.for_image(spec)
        value = recorded[field]
        if isinstance(value, bool):
            recorded[field] = not value
        elif value is None:
            recorded.pop(field)
        elif isinstance(value, str):
            recorded[field] = f"{value}-different"
        elif isinstance(value, list):
            recorded[field] = [value[0] + 1, *value[1:]]
        else:
            recorded[field] = value + 1
        out_dir = tmp_path / field
        _publish_pair(out_dir, spec, recorded)

        remaining, skipped = partition_dense_images_by_resume(
            [spec], out_dir, recipe
        )

        assert remaining == [spec], field
        assert skipped == 0, field
