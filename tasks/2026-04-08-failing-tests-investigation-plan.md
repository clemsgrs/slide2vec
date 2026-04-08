# Failing Tests Investigation Plan

- [x] Reproduce the GitHub test failures locally from the branch diff and CI logs.
- [x] Patch process-list loading so older CSVs without `requested_backend` and `backend` still load.
- [x] Make the cuCIM stderr suppression path safe on systems without an available CUDA runtime.
- [x] Add or update regression coverage for the compatibility and CUDA-safety cases.
- [x] Run the targeted test subset to verify the fixes before reporting back.
