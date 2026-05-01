"""Patient-level pipeline: tile -> slide -> patient embedding aggregation."""

from pathlib import Path
from typing import Sequence

import torch
from hs2p import SlideSpec

from slide2vec.api import ExecutionOptions, PreprocessingConfig
from slide2vec.artifacts import (
    PatientEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
    write_patient_embeddings,
)
from slide2vec.progress import emit_progress
from slide2vec.runtime.embedding import (
    build_slide_embedding_metadata,
    build_tile_embedding_metadata,
    write_slide_embedding_artifact,
    write_tile_embedding_artifact,
)
from slide2vec.runtime.embedding_pipeline import compute_tile_embeddings_for_slide
from slide2vec.runtime.hierarchical import num_embedding_items
from slide2vec.runtime.slide_encode import encode_slide_from_tiles
from slide2vec.runtime.tiling import resolve_slide_backend


def run_patient_pipeline(
    model,
    *,
    embeddable_slides: Sequence[SlideSpec],
    embeddable_tiling_results,
    patient_id_map: dict[str, str],
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    output_dir: Path,
) -> tuple[list[TileEmbeddingArtifact], list[SlideEmbeddingArtifact], list[PatientEmbeddingArtifact]]:
    """Run the patient-level embedding pipeline.

    For each slide: extract tile features and compute a slide-level embedding.
    After processing all slides for a patient: aggregate slide embeddings into
    a single patient embedding via the case transformer.
    """
    loaded = model._load_backend()
    tile_artifacts: list[TileEmbeddingArtifact] = []
    slide_artifacts: list[SlideEmbeddingArtifact] = []

    patient_slide_embeddings: dict[str, list[torch.Tensor]] = {}
    patient_slide_counts: dict[str, int] = {}

    for slide, tiling_result in zip(embeddable_slides, embeddable_tiling_results):
        emit_progress(
            "embedding.slide.started",
            sample_id=slide.sample_id,
            total_tiles=num_embedding_items(tiling_result, preprocessing),
        )
        tile_embeddings = compute_tile_embeddings_for_slide(
            loaded,
            model,
            slide,
            tiling_result,
            preprocessing=preprocessing,
            execution=execution,
        )

        if execution.save_tile_embeddings:
            tile_artifact = write_tile_embedding_artifact(
                slide.sample_id,
                tile_embeddings,
                execution=execution,
                metadata=build_tile_embedding_metadata(
                    model,
                    tiling_result=tiling_result,
                    image_path=slide.image_path,
                    mask_path=slide.mask_path,
                    tile_size_lv0=int(tiling_result.tile_size_lv0),
                    backend=resolve_slide_backend(preprocessing, tiling_result),
                ),
            )
            tile_artifacts.append(tile_artifact)

        emit_progress(
            "aggregation.started",
            sample_id=slide.sample_id,
            total_tiles=num_embedding_items(tiling_result, preprocessing),
        )
        slide_emb = encode_slide_from_tiles(
            loaded,
            tile_embeddings,
            tiling_result,
            execution=execution,
        )
        emit_progress("aggregation.finished", sample_id=slide.sample_id, has_latents=False)

        if execution.save_slide_embeddings:
            slide_artifact = write_slide_embedding_artifact(
                slide.sample_id,
                slide_emb,
                execution=execution,
                metadata=build_slide_embedding_metadata(model, image_path=slide.image_path),
            )
            slide_artifacts.append(slide_artifact)

        patient_id = patient_id_map.get(slide.sample_id, slide.sample_id)
        patient_slide_embeddings.setdefault(patient_id, []).append(slide_emb)
        patient_slide_counts[patient_id] = patient_slide_counts.get(patient_id, 0) + 1

        emit_progress(
            "embedding.slide.finished",
            sample_id=slide.sample_id,
            num_tiles=num_embedding_items(tiling_result, preprocessing),
        )

    patient_artifacts: list[PatientEmbeddingArtifact] = []
    for patient_id, slide_embs in patient_slide_embeddings.items():
        stacked = torch.stack(slide_embs, dim=0).to(loaded.device)
        with torch.inference_mode():
            patient_emb = loaded.model.encode_patient(stacked).detach().cpu()
        artifact = write_patient_embeddings(
            patient_id,
            patient_emb,
            output_dir=output_dir,
            output_format=execution.output_format,
            metadata={"encoder_name": model.name, "encoder_level": model.level},
            num_slides=patient_slide_counts[patient_id],
        )
        patient_artifacts.append(artifact)

    return tile_artifacts, slide_artifacts, patient_artifacts
