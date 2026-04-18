# Lessons Learned

## 2026-04-18

- Prefer neutral package names like `runtime/` for internal implementation modules unless the user explicitly wants a private-style namespace; leading underscores in directory names read as accidental or overly internal.

- When slide2vec depends on bridged HS2P progress events, keep the bridge whitelist in sync with every reporter stage the UI renders; otherwise the code can define a preview bar and still never receive preview events.

## 2026-04-18

- Keep `tiling.finished` for closing the live bar and emit the final summary on a separate `tiling.summary` event; otherwise the reporter ends up printing the same panel twice.
- Split coordinate extraction and preview flushing into separate progress stages so the tiling bar stays live through the actual slide work instead of going stale at 0% during preview cleanup.
- When a live progress renderer needs to stay readable, keep backend-selection notices on plain console output and avoid buffering them behind a broad stdout/stderr capture wrapper.

## 2026-04-12

- When refactoring CLI parsing to support `parse_known_args()`, prefer updating the test double to match the real parser API instead of adding a production fallback for mocks. Keep the runtime code clean unless the fallback is genuinely needed by real callers.
- When a regression test stubs a helper that normally returns a structured object, keep the production code strict and make the stub return the expected attributes instead of broadening the runtime API to accept a placeholder value.
- When stdout noise comes from both a local wrapper and an upstream dependency, suppress the common-case message at the upstream boundary and keep the per-item local note at debug level instead of stacking multiple INFO lines.
- When a diagnostic line is still useful, move it to the run boundary and log it once per run instead of per slide or per worker.
- When a backend-selection reason is user-relevant, emit it as a structured progress event and let the reporter choose how to render it instead of filtering it out in the API.
- When a CLI depends on both slide2vec and hs2p progress systems, activate them with a shared reporter bridge so upstream events do not disappear into the default null reporter.

## 2026-04-10

- In this environment, never route `apply_patch` through `exec_command`; use the dedicated `apply_patch` tool directly for file edits.

## 2026-04-17

- When the workspace contains both `/data/pathology/projects/clement/code/slide2vec` and `/tmp/slide2vec`, treat the `/data/...` checkout as the source of truth for edits and reporting unless the user explicitly says otherwise.

## 2026-04-01

- When the user says backward compatibility is not needed, do not add compatibility-preserving behavior or messaging; implement the simpler direct behavior instead.
- Before declaring a module missing from the repo, check `git ls-files` and ignore rules as well as the filesystem; a local file can exist in the workspace while still being absent from the committed branch.
- When integrating with hs2p 3.x tiling artifacts, treat `tissue_mask_path` as the upstream CSV field and normalize it to the legacy `mask_path` alias inside slide2vec instead of hard-coding the old name everywhere.
- When hs2p moves a config flag into a nested field, update slide2vec to read the nested value directly and remove the root-level compatibility key from configs, docs, and tests at the same time.
- When removing a config knob that is now derived from the preset or runtime object, scrub it from test fixtures and docs as well; leaving it in any YAML example will cause the validation path to fail before the behavior under test is exercised.
- If a fluent method already knows its concrete class, prefer an explicit return annotation like `"CONCH"` or `"Pipeline"` over `Self`; it keeps the type story simple and avoids extra typing dependencies on older interpreters.
- When a user says a file is local-only, never add it to git history or include it in a PR even if they ask to modify it; keep the change local and confine the PR to repository-tracked files.
- When the user explicitly rejects backward compatibility, remove compatibility shims and update writers/tests to emit the new schema instead of accepting old inputs.

## 2026-04-11

- When a config namespace may come from tests, fixtures, or older CLI payloads, use `getattr(..., default)` for optional flags instead of assuming every field is present.
- When an upstream helper disappears, add a local wrapper at the call site rather than importing the missing symbol directly from the dependency.
- When the sibling checkout for a dependency already exports the helper you need, align with that source instead of compensating for an older installed wheel.
- When a field is part of the current config contract, update the test fixtures to include it rather than adding a backward-compatibility branch in production code.

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
- When the user authorizes multiple project virtualenvs, prefer the one that can run tests with fewer system-level build dependencies, and install missing Python packages directly into that approved venv.

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
- In README workflow descriptions for this repository, explicitly contrast the interactive in-memory path with the manifest-driven batch-to-disk path; phrases like "persisted artifact generation" alone are too abstract.
- In the README for this repository, keep a brief multi-GPU sharding note near both the Python and CLI quickstarts so the capability is visible without forcing readers into the deeper API docs.
- For GPU-count defaults in this repository, treat "unset" as "use all available GPUs" rather than `1`, for both the Python API and the CLI config path.
- When the CLI is config-driven in this repository, provide a dedicated `docs/cli.md` and link it from the README; otherwise new users cannot easily infer when to use the CLI or how overrides/configs work.
- When I promise a new public doc in this repository, make sure the file is actually tracked and pushed; a local untracked doc is effectively invisible to users and reviewers.
- When tests deliberately clear and re-import `slide2vec.*` modules in this repository, later monkeypatch-based tests should patch and instantiate through the same re-imported module object rather than a stale class imported at test-module load time.

## 2026-03-18

- When `slide2vec` mirrors an upstream `hs2p` data model, prefer the upstream type (`SlideSpec`) as the single canonical reference instead of maintaining a parallel local name.
- When adding a new slide-spec field in this repository, wire it through every ingestion and resume path in the same change: direct API normalization, manifest parsing, process-list persistence, distributed worker reloads, and docs.
- For distributed CLI orchestration in this repository, launch worker subprocesses in their own process group and tear down the whole group on the first `KeyboardInterrupt`; otherwise users may need to spam `Ctrl+C` to stop torchrun descendants.
- For dependency-split work in this repository, classify packages by actual import-time usage in core modules before moving them to extras; `slide2vec.data` and inference helpers still require packages like `torch`, `torchvision`, `einops`, and `wholeslidedata` even when model-factory integrations remain optional.
- When the user asks to keep version constraints in repository requirements files, preserve the requested comparator style (`>=` ranges vs exact `==` pins) rather than tightening it on your own.
- When aligning `slide2vec` to an upstream `hs2p` release in this repository, use the exact user-specified minimum version and do not add migration notes the user explicitly said to omit.
- For small mechanical cleanups in this repository, do not add a new regression test when the user explicitly says to keep the modification simple; prefer updating the implementation and verifying with existing coverage.
- When the user says benchmark or bookkeeping code is local-only in this repository, keep it out of the PR even if it sits next to general fixes; split PR scope from local experimentation explicitly.
- When removing local-only bookkeeping code from a PR in this repository, keep the files in the working tree unless the user explicitly asks to delete them locally too.

## 2026-03-19

- When optimizing performance paths in this repository, remove no-op parameters and dead scaffolding immediately; do not leave constructor args or branches that are accepted only to be discarded.
- Prefer lazy imports at the exact call sites that need optional heavy dependencies over broad module-level `try/except` fallbacks; keep the code simple and only defensive where it materially improves behavior.
- When the repository’s own `requirements.in` / `requirements-models.in` define a dependency as core, import it plainly at module scope instead of hiding it behind lazy-loading indirection for the sake of a thinner local test env.
- For core libraries like `transformers`, keep the imports at module scope in the model implementation files; constructor-local imports make the code harder to read without buying us anything once the dependency is mandatory.
- Do not apply that rule to genuinely optional extras like `gigapath`; those can stay lazily imported if the package is not part of the core install.
- Avoid naming a method after a built-in if later annotations in the same class use that built-in name, because class-body evaluation can resolve the annotation to the method object instead of the built-in and crash at import time.
- For slide-native encoders, the tile-embedding path should explicitly use the registered tile encoder dependency, not the slide encoder object itself; keep a small fallback for callable test doubles, but make the production dispatch depend on the tile encoder.
- For cuCIM pyramid reads in this repository, resolve level-0 spacing from `CuImage.spacing()` and fail fast on missing pyramid metadata instead of threading redundant spacing overrides or silently assuming a single level.
- For multi-GPU CLI progress in this repository, tag worker-emitted events with a stable source label such as `cuda:{local_rank}` and key Rich progress tasks by that label; otherwise concurrent ranks overwrite each other’s rows and the UX becomes misleading.
- When creating branches, PR titles, or commit messages in this repository, do not add a `[codex]` prefix unless the user explicitly asks for it.
- When config code starts needing `hasattr` or other compatibility branches, update the fixtures and default schema to be explicit instead of making the runtime tolerate half-shaped inputs.

## 2026-03-17

- In this repository session, when the user explicitly tells me to use `~/Code/venv/ijepath/bin/python`, prefer that interpreter for Python commands instead of defaulting to another project venv.
