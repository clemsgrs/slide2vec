# Agent instructions

## Testing

- Never write unit tests after you write code.
- Highly prefer E2E tests as the sole testing mechanism. Use them to verify complex features work. At the end of E2E tests, produce a verifiable and repeatable artifact.
- If you must test a system in isolation, first write down all the ways it could fail, then write the code.

E2E tests in this repo:

- `tests/test_output_consistency.py`: runs the `slide2vec` CLI on a real slide and compares coordinates and embeddings against ground truth in `tests/fixtures/gt/`.
- Tests marked `heavy`: real-weight inference per model. Run with `python -m pytest -m heavy`.
- Tests marked `gpu_integration`: one-GPU versus multi-GPU parity. Run with `CUDA_VISIBLE_DEVICES=0,1 python -m pytest -m gpu_integration --no-cov`.
