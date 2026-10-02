# Changelog

## [Unreleased]

### Added
- Agent labelling over MCP (`apps/mcp/label_server.py`, `vlm_labeling/agent/`): Claude Code views images with a
  coordinate grid, zooms, places one point per head; SAM2 masks with a shape check; labels saved as pending review
  or pushed to Label Studio as predictions. `.mcp.json` and the `/label-trichomes` command; extra `[agent]`
- `docker/cpu/`: CPU-only server stack (backend, production web UI, MLflow, Label Studio), images built from `uv.lock`

### Fixed
- A fresh install could not start: the SQLite folder (`db/`) was not created
- Background-task history was written to a hard-coded `./data` instead of `DATA_ROOT`
- Training: the configured gradient accumulation (`batch_size × gradient_accumulation_steps`) was never passed to
  Ultralytics, which always used a nominal batch of 64 — small datasets made one optimizer step every few dozen
  epochs. Now passed as `nbs`.

All notable changes to CTIP. The project follows [Semantic Versioning](https://semver.org/); while in alpha, minor
versions may break APIs.

## [0.1.0-alpha] — 2026-10-01

First public release.

### Security
- The web UI supports `API_TOKEN`: it asks once and keeps the token in an HttpOnly, SameSite=Strict cookie;
  WebSocket handshakes are checked too (they bypassed the HTTP middleware before)
- Docker: direct service ports listen on 127.0.0.1 by default (`DIRECT_BIND`); only nginx is public
- Dependabot and CodeQL

### Added
- `notebooks/quickstart.ipynb` — focus, optical maturity, µm scale and the session split without GPU or data (runs in CI)
- GPU detection benchmark (`benchmarks/detection/gpu_detection_benchmark.py`) with RTX 4060 results
- Dataset and model card templates (`docs/templates/`), research notes (`docs/research/`)
- `uv.lock` for reproducible installs; ruff lint gate and test timeouts in CI
- Session-grouped dataset splits for the Label Studio export (leakage prevention), with warnings when sessions are
  missing or too few
- CI: CPU unit tests and frontend build on every push and pull request
- CONTRIBUTING, SECURITY, CITATION, issue and PR templates
- CTIP colour design for the web UI (light by default, dark variant)

### Changed
- Core dependencies trimmed to what the code imports (fiftyone, dvc, gradio, weasyprint, statsmodels, … removed);
  `onnxruntime` and `wandb` are extras
- Licence: AGPL-3.0-or-later (LICENSE text added; README badge and package metadata were inconsistent)
- README is a short front page; full EN/DE/ES manuals live in `docs/manual/`
- CLI and UI name: CTIP (was TrichomeLab)

### Fixed
- Backend lifespan used `asyncio` before importing it — the task-expiry loop never started
- Background tasks could be garbage-collected mid-run; training callbacks never reached the WebSocket from the
  worker thread
- Scale-bar detection crashed with OpenCV 5; a missing image made Ultralytics try to pip-install `pi-heif`
- VLM JSON code-fence stripping removed leading characters of the answer
- Fresh installs: missing `sqlmodel`/`alembic` dependencies, wheel package list, task store schema on first use
- TensorRT helpers report missing files before missing TensorRT
- 20 unit tests that only passed with an implicit asyncio event loop
- Frontend production build (ESLint)
- Documentation of CLI commands that did not exist

### Removed
- Local databases, logs, MLflow runs and environment files from the repository
