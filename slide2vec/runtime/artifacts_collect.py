"""Collect pipeline-level tile/slide/hierarchical artifacts (local + distributed paths)."""

from pathlib import Path
from typing import Sequence

from hs2p import SlideSpec

from slide2vec.api import EmbeddedSlide, ExecutionOptions, PreprocessingConfig
from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
)
from slide2vec.runtime import embedding as runtime_embedding
from slide2vec.runtime.distributed_stage import run_distributed_embedding_stage
from slide2vec.runtime.embedding_persist import persist_embedded_slide
from slide2vec.runtime.hierarchical import is_hierarchical_preprocessing
from slide2vec.runtime.persistence import (
    collect_pipeline_artifacts,
    update_process_list_after_embedding,
)
from slide2vec.runtime.process_list import resolved_process_list_output_variant


def collect_local_pipeline_artifacts(
    *,
    model,
    embedded_slides: Sequence[EmbeddedSlide],
    tiling_results,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
) -> tuple[list[TileEmbeddingArtifact], list[HierarchicalEmbeddingArtifact], list[SlideEmbeddingArtifact]]:
    tile_artifacts: list[TileEmbeddingArtifact] = []
    hierarchical_artifacts: list[HierarchicalEmbeddingArtifact] = []
    slide_artifacts: list[SlideEmbeddingArtifact] = []
    for embedded_slide, tiling_result in zip(embedded_slides, tiling_results):
        tile_artifact, slide_artifact = persist_embedded_slide(
            model,
            embedded_slide,
            tiling_result,
            preprocessing=preprocessing,
            execution=execution,
        )
        if isinstance(tile_artifact, HierarchicalEmbeddingArtifact):
            hierarchical_artifacts.append(tile_artifact)
        elif tile_artifact is not None:
            tile_artifacts.append(tile_artifact)
        if slide_artifact is not None:
            slide_artifacts.append(slide_artifact)
    return tile_artifacts, hierarchical_artifacts, slide_artifacts


def collect_distributed_pipeline_artifacts(
    *,
    model,
    successful_slides: Sequence[SlideSpec],
    process_list_path: Path,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    output_dir: Path,
    tiling_input_dir: Path | None = None,
) -> tuple[
    list[TileEmbeddingArtifact],
    list[HierarchicalEmbeddingArtifact],
    list[SlideEmbeddingArtifact],
]:
    persist_tile_embeddings = runtime_embedding.should_persist_tile_embeddings(model, execution)
    persist_hierarchical_embeddings = is_hierarchical_preprocessing(preprocessing)
    include_slide_embeddings = model.level == "slide"
    include_tile_embeddings = persist_tile_embeddings and not persist_hierarchical_embeddings
    run_distributed_embedding_stage(
        model=model,
        successful_slides=successful_slides,
        preprocessing=preprocessing,
        execution=execution,
        output_dir=output_dir,
        tiling_input_dir=tiling_input_dir,
    )
    tile_artifacts, hierarchical_artifacts, slide_artifacts = collect_pipeline_artifacts(
        successful_slides,
        output_dir=output_dir,
        output_format=execution.output_format,
        include_tile_embeddings=include_tile_embeddings,
        include_hierarchical_embeddings=persist_hierarchical_embeddings,
        include_slide_embeddings=include_slide_embeddings,
    )
    update_process_list_after_embedding(
        process_list_path,
        successful_slides=successful_slides,
        persist_tile_embeddings=persist_tile_embeddings,
        persist_hierarchical_embeddings=persist_hierarchical_embeddings,
        include_slide_embeddings=include_slide_embeddings,
        encoder_name=model.name,
        output_variant=resolved_process_list_output_variant(model),
        tile_artifacts=tile_artifacts,
        hierarchical_artifacts=hierarchical_artifacts,
        slide_artifacts=slide_artifacts,
    )
    return tile_artifacts, hierarchical_artifacts, slide_artifacts
