"""Construct EmbeddedSlide objects and persist their tile/slide artifacts."""

from pathlib import Path

from hs2p import SlideSpec

from slide2vec.api import EmbeddedSlide, ExecutionOptions, PreprocessingConfig
from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
    write_tile_embedding_metadata,
)
from slide2vec.runtime.embedding import (
    build_hierarchical_embedding_metadata,
    build_tile_embedding_metadata,
    build_slide_embedding_metadata,
    should_persist_tile_embeddings,
    write_hierarchical_embedding_artifact,
    write_slide_embedding_artifact,
    write_tile_embedding_artifact,
)
from slide2vec.runtime.hierarchical import is_hierarchical_preprocessing
from slide2vec.runtime.process_list import num_rows
from slide2vec.runtime.tiling import resolve_slide_backend
from slide2vec.utils.coordinates import coordinate_arrays


def make_embedded_slide(
    *,
    slide: SlideSpec,
    tiling_result,
    tile_embeddings,
    slide_embedding=None,
    latents=None,
) -> EmbeddedSlide:
    x_values, y_values = coordinate_arrays(tiling_result)
    if num_rows(tile_embeddings) != len(x_values):
        raise ValueError(
            f"Tile embedding count ({num_rows(tile_embeddings)}) does not match coordinate count ({len(x_values)})"
        )
    n_tiles = tiling_result.num_tiles if hasattr(tiling_result, "num_tiles") else None
    mask_preview_path = (
        tiling_result.mask_preview_path if hasattr(tiling_result, "mask_preview_path") else None
    )
    tiling_preview_path = (
        tiling_result.tiling_preview_path if hasattr(tiling_result, "tiling_preview_path") else None
    )
    return EmbeddedSlide(
        sample_id=slide.sample_id,
        tile_embeddings=tile_embeddings,
        slide_embedding=slide_embedding,
        x=x_values,
        y=y_values,
        tile_size_lv0=int(tiling_result.tile_size_lv0),
        image_path=slide.image_path,
        mask_path=slide.mask_path,
        num_tiles=int(n_tiles) if n_tiles is not None else len(x_values),
        mask_preview_path=Path(mask_preview_path) if mask_preview_path is not None else None,
        tiling_preview_path=Path(tiling_preview_path) if tiling_preview_path is not None else None,
        latents=latents,
    )


def persist_embedded_slide(
    model,
    embedded_slide: EmbeddedSlide,
    tiling_result,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
) -> tuple[TileEmbeddingArtifact | HierarchicalEmbeddingArtifact | None, SlideEmbeddingArtifact | None]:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist embedded slides")
    if num_rows(embedded_slide.tile_embeddings) == 0:
        write_tile_embedding_metadata(
            embedded_slide.sample_id,
            output_dir=execution.output_dir,
            output_format=execution.output_format,
            feature_dim=None,
            num_tiles=0,
            metadata=build_tile_embedding_metadata(
                model,
                tiling_result=tiling_result,
                image_path=embedded_slide.image_path,
                mask_path=embedded_slide.mask_path,
                tile_size_lv0=embedded_slide.tile_size_lv0,
                backend=resolve_slide_backend(preprocessing, tiling_result),
            ),
        )
        return None, None
    if is_hierarchical_preprocessing(preprocessing):
        hierarchical_artifact = write_hierarchical_embedding_artifact(
            embedded_slide.sample_id,
            embedded_slide.tile_embeddings,
            execution=execution,
            metadata=build_hierarchical_embedding_metadata(
                model,
                tiling_result=tiling_result,
                image_path=embedded_slide.image_path,
                mask_path=embedded_slide.mask_path,
                backend=resolve_slide_backend(preprocessing, tiling_result),
                preprocessing=preprocessing,
            ),
        )
        return hierarchical_artifact, None
    tile_artifact = None
    if should_persist_tile_embeddings(model, execution):
        tile_artifact = write_tile_embedding_artifact(
            embedded_slide.sample_id,
            embedded_slide.tile_embeddings,
            execution=execution,
            metadata=build_tile_embedding_metadata(
                model,
                tiling_result=tiling_result,
                image_path=embedded_slide.image_path,
                mask_path=embedded_slide.mask_path,
                tile_size_lv0=embedded_slide.tile_size_lv0,
                backend=resolve_slide_backend(preprocessing, tiling_result),
            ),
        )
    slide_artifact = None
    if embedded_slide.slide_embedding is not None:
        slide_artifact = write_slide_embedding_artifact(
            embedded_slide.sample_id,
            embedded_slide.slide_embedding,
            execution=execution,
            metadata=build_slide_embedding_metadata(model, image_path=embedded_slide.image_path),
            latents=embedded_slide.latents,
        )
    return tile_artifact, slide_artifact
