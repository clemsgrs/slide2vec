from pathlib import Path

import pandas as pd
import pytest

from slide2vec.utils.tiling_io import load_patient_id_mapping, load_slide_manifest


@pytest.mark.parametrize("invalid_patient_id", [None, "", "   "])
def test_patient_mapping_rejects_invalid_patient_ids_and_names_every_sample(
    tmp_path: Path,
    invalid_patient_id: str | None,
):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "sample_id": ["sample-b", "sample-a"],
            "image_path": ["/slides/b.svs", "/slides/a.svs"],
            "patient_id": [invalid_patient_id, invalid_patient_id],
        }
    ).to_csv(manifest, index=False)

    with pytest.raises(
        ValueError,
        match=r"^Invalid patient_id values for samples: sample-b, sample-a$",
    ):
        load_patient_id_mapping(manifest)


def test_patient_mapping_preserves_textual_ids_and_grouping(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,image_path,patient_id\n"
        "0007,/slides/a.svs,0012\n"
        "0008,/slides/b.svs,0012\n"
        "0009,/slides/c.svs,0013\n"
    )

    mapping = load_patient_id_mapping(manifest)

    assert mapping == {
        "0007": "0012",
        "0008": "0012",
        "0009": "0013",
    }


def test_slide_manifest_preserves_leading_zero_sample_ids(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,image_path\n"
        "0007,/slides/a.svs\n"
        "0008,/slides/b.svs\n"
    )

    slides = load_slide_manifest(manifest)

    assert [slide.sample_id for slide in slides] == ["0007", "0008"]
