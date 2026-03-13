# 2026-03-13 Slide2vec API Refactor Implementation

- [x] Inspect the current `slide2vec` package layout, entrypoints, and HS2P integration points.
- [x] Confirm the desired public API direction (`Model.from_pretrained(...)`, typed Python-first API, hard cutover artifact layout).
- [x] Add contract tests for the new package root exports, lazy imports, artifact layout, writer round-trips, CLI delegation, and package resources.
- [x] Implement the new public API layer (`Model`, `Pipeline`, `RunOptions`, result objects) with lazy root exports.
- [x] Extract reusable inference and aggregation logic into internal modules that no longer depend on CLI/process-list orchestration.
- [x] Introduce the hard-cutover artifact layer with `tile_embeddings/` and `slide_embeddings/` outputs plus JSON sidecars.
- [x] Rebuild CLI/config entrypoints as thin adapters over the package API and remove legacy `features/` writes.
- [x] Update HS2P boundary helpers to consume the current public tiling result contract directly.
- [x] Update packaging/resource loading so bundled YAML configs work from installed distributions.
- [x] Refresh README and `docs/documentation.md` to document the new API-first usage and cutover.
- [x] Run targeted verification and mark off any remaining gaps explicitly.
