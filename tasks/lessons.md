# Lessons Learned

## 2026-02-10

- When a git submodule shows unexpected local modifications, explicitly confirm scope with the user before editing.
- If the user restricts changes to suggestions-only for a submodule, keep all code edits outside that submodule and document recommendations separately.

## 2026-02-25

- When the user sets a canonical config/API key (for example `model.name: "pathojepa"`), use that exact key consistently across factory code, presets, docs, and tests.
- In this repository session, run Python commands with `/Users/clems/Code/venv/slide2vec/bin/python` when the user explicitly requests that interpreter.
- For performance discrepancies between two launch paths, verify effective config parity first (especially expensive tiling knobs like segmentation/visualization downsample) before attributing the slowdown to orchestration code.
- If multi-GPU logs show NCCL "device currently unknown" right before startup hangs, prioritize explicit rank-to-device binding in distributed init/barriers (`device_id` and `barrier(device_ids=[local_rank])`) before broader refactors.
- In this environment, `torch.distributed.init_process_group(..., device_id=...)` requires a `torch.device` object, not an integer CUDA ordinal; use `torch.device(f"cuda:{local_rank}")`.
- Do not add an unconditional `dist.barrier()` immediately after `init_process_group` in startup paths unless strictly required; this extra collective can hang even when NCCL communicator init succeeds.

## 2026-03-12

- In this repository session, use `~/Code/venv/slide2vec/bin/python` for Python-related commands when the user specifies that interpreter.

## 2026-03-13

- When a ratio is defined in physical-resolution terms, derive it from spacing metadata directly and make the regression test use values where spacing ratio and size ratio differ, so the test catches accidental proxy calculations.
- After a refactor that changes integration boundaries or helper APIs, do a quick unused-import and dead-symbol sweep so stale type imports do not linger in the touched modules.
- Remove narrow temporary tests once the fix is verified unless they protect a stable contract or key structural behavior; avoid leaving implementation-detail assertions behind.
- For HS2P integration in this repository, stick to the actual public `TilingResult` coordinate fields (`x` and `y`) instead of inventing alternate names from memory.
- For user-facing workflow APIs in this repository, prefer long-lived configured objects (for example `Pipeline(model, preprocessing, execution=...)`) over passing core configuration pieces again at `run(...)` time.
- In README examples for this repository, prefer the minimal happy path and rely on defaults when reasonable; move fuller option matrices into dedicated docs instead of crowding the first example.
- For interactive analysis workflows in this repository, prefer direct in-memory APIs (for example `model.embed_slide(...)`) over presenting the batch `Pipeline` path as the first example.
- When the repository supports two primary workflows (interactive in-memory use and batch/pipeline use), show both in the README rather than forcing readers to infer the second one from deeper docs.
- For multi-GPU restore work in this repository, prefer sharding manifest slides across ranks and reusing the normal per-slide embedding path over reintroducing cross-rank collectives inside the core embedding loop.
- For direct multi-GPU embedding in this repository, keep the strategies explicit and simple: shard tiles for `embed_slide(...)`, and balance whole slides by tile count for `embed_slides(...)` instead of inventing a hybrid scheduler.
- For single slide-level embeddings in this repository, prefer the natural public shape `(D)` over a fake batch shape `(1, D)`; let callers `unsqueeze(0)` themselves if they want batching.
- Name regression tests after the behavior they actually prove, and remove permanent no-op checks once the deleted/stale code is gone.
- When a structural regression test must inspect source, prefer resilient AST- or structure-based assertions over exact string matches that break on harmless formatting changes.
- For public typing in this repository, prefer a small number of high-signal aliases and protocols over near-duplicate “accepted input shape” types that mostly restate the same contract.
- For Python-facing execution knobs in this repository, prefer explicit API defaults over inferred “model defaults” when the value is really hardware/user dependent; batch size should default to `1` unless the caller sets it.
- When persistence is mandatory in this repository, validate required paths at the top of the public or inference helper before loading models or building datasets; fail-fast beats loop-local errors.
- When multi-GPU execution is requested in this repository, validate device/runtime feasibility before expensive tiling or artifact preparation so impossible runs fail immediately.
- For tile-to-slide embedding steps in this repository, prefer `aggregate_tiles(...)` over names like `aggregate_slides(...)`; the input is tile artifacts/features and the output is a slide embedding.
- Remove `from __future__ import annotations` when it is no longer buying us anything; keep it only when a module truly needs postponed evaluation rather than as leftover boilerplate.
