# 2026-03-14 In-Memory Embed Slide API

- [x] Add contract tests for `EmbeddedSlide`, `Model.embed_slide(...)`, and `Model.embed_slides(...)`.
- [x] Refactor the inference layer so tiling/embedding can return in-memory results before artifact writers run.
- [x] Add `EmbeddedSlide` to the public API and support string/path slide inputs with optional `sample_id` and `mask_path`.
- [x] Keep `Pipeline.run(...)` artifact-oriented and requiring `ExecutionOptions.output_dir`.
- [x] Support optional artifact persistence for direct embedding when `execution.output_dir` is provided.
- [x] Update README and API docs to show `embed_slide(...)` for interactive workflows and keep `Pipeline` for manifest-driven runs.
- [x] Run targeted and full verification.
