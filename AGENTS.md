# Agent instructions

## Testing

- Never write unit tests after you write code.
- Highly prefer E2E tests as the sole testing mechanism. Use them to verify complex features work. At the end of E2E tests, produce a verifiable and repeatable artifact.
- If you must test a system in isolation, first write down all the ways it could fail, then write the code.

E2E tests in this repo:

- `tests/test_output_consistency.py`: runs the `slide2vec` CLI on a real slide and compares coordinates and embeddings against ground truth in `tests/fixtures/gt/`.
- Tests marked `heavy`: real-weight inference per model. Run with `python -m pytest -m heavy`. CI runs the fast suite (`-m "not heavy"`) on every code PR and the PRISM consistency test in a parallel job when a PR touches the pipeline; the other heavy tests run only through the manual heavy workflow, before a release.
- Tests marked `gpu_integration`: one-GPU versus multi-GPU parity. Run with `CUDA_VISIBLE_DEVICES=0,1 python -m pytest -m gpu_integration --no-cov`.

Environments: `uv.lock` pins every extra. After editing dependencies in `pyproject.toml`, run `uv lock` and check with `uv lock --check`. For tests, create the CI environment with `uv sync --extra cpu --extra fm --group dev`.
`release.py` runs `uv lock` after `bumpver update`, because `uv.lock` records the project version.
