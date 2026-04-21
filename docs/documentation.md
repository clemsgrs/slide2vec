# Recent Fixes

- Distributed `run_pipeline_with_coordinates()` runs now pass the coordinate artifact directory through to multi-GPU workers, so workers reload precomputed tiling artifacts from the requested coordinates source instead of the embedding output directory.
- On-the-fly CuCIM worker budgeting now rejects `num_cucim_workers <= 0` with a clear `ValueError` instead of crashing with division-by-zero during worker allocation.
- Distributed execution payload serialization now preserves `save_slide_embeddings` and `num_preprocessing_workers` across serialize/deserialize round-trips.
