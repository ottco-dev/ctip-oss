# Technical Debt Register

Last updated: 2026-05-29 (Moondream bitsandbytes incompatibility resolved; accelerate pinned)

## RESOLVED

### ~~TDB-019~~: SchemaEnforcer instantiation in all 6 remote provider files ✅ FIXED 2026-05-28
- All remote provider files had `_SCHEMA_ENFORCER = SchemaEnforcer()` without required `schema` arg
- Also called `enforce_maturity(dict)` which doesn't exist and takes raw text not dict
- Fix: removed enforcer entirely from providers — schema validation is pipeline-level (HallucinationFilter)

### ~~TDB-020~~: OpenAI provider PROMPT_REGISTRY access pattern ✅ FIXED 2026-05-28
- `PROMPT_REGISTRY.get("key", {}).get("prompt")` assumes dict but returns `PromptTemplate` object
- Also used wrong key `"image_quality_screen"` (correct: `"image_quality"`)
- Fix: `_get_user_prompt()` helper accesses `.user_prompt_template` with isinstance guard

### ~~TDB-021~~: HuggingFace provider forced API key ✅ FIXED 2026-05-28
- Raised `ValueError` on empty token, preventing anonymous tier usage
- Fix: removed mandatory check; `is_available = True` always (anonymous rate-limited access)

### ~~TDB-018~~: Docker build fails — OSError: License file does not exist: LICENSE ✅ FIXED 2026-05-28
- hatchling requires LICENSE, README.md, and all declared packages at build time
- Fix: dep-cache layer COPYs pyproject.toml + LICENSE + README.md; `create_pkg_stubs.py` creates empty `__init__.py` stubs for all 19 packages before `uv pip install "[all,dev]"`
- Also fixed: CUDA mismatch (cu130 vs 12.8), flash-attn soft-fail, SAM2 git install

### ~~TDB-023~~: Moondream2 bitsandbytes incompatibility ✅ RESOLVED 2026-05-29
- moondream2 uses `F.linear(x, w.weight, w.bias)` directly, bypassing bitsandbytes module forward
- 4-bit/8-bit quantization → `RuntimeError: Half and Byte dtype mismatch`
- accelerate 1.13.0 also broken: `dispatch_model→model.to()` on bnb models raises ValueError
- Fix: `quantization='none'` (FP16, ~4.2 GB); accelerate pinned to 0.26.1
- VRAM budget OK: 4.2 GB moondream + 1.5 GB YOLO < 8 GB; semaphore ensures exclusivity

## ACTIVE

### ~~TDB-022~~: Active VLM provider setting is ephemeral ✅ RESOLVED 2026-05-29
- Root cause: `_get_active_provider_id()` called `get_settings()` which is `@lru_cache(maxsize=1)` —
  even after `.env` write, the cached Settings object had the old value
- Fix 1: `_get_active_provider_id()` and `_get_active_model()` now read `os.environ` first
  (always up-to-date after the write); fall back to `get_settings()` only if not in env
- Fix 2: Both `set_active_provider` and `configure_provider` call `get_settings.cache_clear()`
  after `write_env_keys()` — ensures next cold read gets fresh `.env` values after restart
- Fix needed: persist to `.env` file or SQLite settings table

### TDB-023: Modal training submission is a stub
- `ModalBackend.run_training_job()` returns failure with documentation pointer
- Real implementation requires dataset upload to Modal Volumes + training script
- Deferred until training-outsource feature is prioritized

## RESOLVED

### ~~TDB-013~~: maturity_pipeline.py broken imports ✅ FIXED 2026-05-25
- `detect_degradation` → `assess_degradation`
- `classify_by_scientific_rules` → `rule_based_maturity_estimate`

### ~~TDB-014~~: cv2.COLORMAP_RdYlGn not available ✅ FIXED 2026-05-25
- `focus/metrics/composite.py` + `focus/guidance/heatmap.py`
- Now uses `getattr(cv2, "COLORMAP_RdYlGn", cv2.COLORMAP_JET)`

### ~~TDB-004~~: In-memory ProfileManager loses state on server restart ✅ FIXED 2026-05-25
- `measurement/api/router.py` — persistent JSON-backed singleton via `_get_profile_manager()`
- Storage path from `TRICHOME_PROFILES_PATH` env var or `DATA_ROOT/calibration/profiles.json`

### ~~TDB-005~~: Video API router temp file cleanup not guaranteed ✅ FIXED 2026-05-25
- `video_pipeline/api/router.py` — `_temp_video()` context manager with guaranteed `os.unlink()`
  in `finally` block; uses `tempfile.mkstemp()` for explicit FD control

### ~~TDB-006~~: MorphologyClassifier inline imports ✅ FIXED 2026-05-25
- `morphology/api/router.py` — `from collections import Counter` moved to module level

### ~~TDB-002~~: CalibrationScale constructor validation ✅ ALREADY FIXED (pre-existing)
- `shared/core/value_objects.py` — `__post_init__` raises `ValueError` for `um_per_pixel <= 0`
- Tech debt was stale; validation was already in place

### ~~TDB-003~~: MaturityAnalyzer.analyze() type assertion ✅ FIXED 2026-05-25
- `maturity/application/maturity_pipeline.py` — `analyze_crop()` now has explicit
  `isinstance(result, MaturityLabel)` guard with informative `TypeError` on contract breakage

---

## CRITICAL (blocks scientific validity)

### ~~TDB-001~~: Stage micrometer auto-detection not integrated ✅ FIXED 2026-05-25
**Files changed**:
- `measurement/calibration/stage_micrometer.py` — Added `detect_scale_bar_px()` function
  and `ScaleBarDetectionResult` dataclass. Algorithm: CLAHE → Gaussian blur → Canny → HoughLinesP
  → horizontal filter → Y-cluster → span measurement → confidence score.
- `measurement/api/router.py` — Added `POST /measurement/profiles/calibrate/auto` endpoint
  accepting image upload + known `scale_bar_um`, runs auto-detection, saves profile on success.
- `tests/unit/test_measurement.py` — 6 new `TestScaleBarDetector` tests (45 total passing).
**Confidence heuristic**: span_px / image_width × 2, clamped to [0, 1]. Below 0.5 = unreliable.
**Remaining**: UI for auto-calibrate endpoint in frontend calibration page.

---

## HIGH (degrades reliability)

*(all resolved — see RESOLVED section above)*

---

## MEDIUM (code quality)

### ~~TDB-007~~: asyncio.Semaphore(1) not used in new module routers ✅ FIXED 2026-05-25
**Solution**:
- `backend/dependencies/gpu.py` (NEW) — shared `asyncio.Semaphore(1)` with:
  - `acquire_gpu_slot(timeout)` async context manager
  - `gpu_slot()` FastAPI Depends()-compatible async generator
  - `wire_task_router_semaphore()` — unifies REST and background task semaphores
  - `gpu_semaphore_status()` — for /queue health endpoint
- `maturity/api/router.py` — `analyze_crop` + `analyze_population` guarded
- `morphology/api/router.py` — `classify_instance` guarded
- `backend/main.py` — wires semaphores in lifespan startup
- `backend/api/v1/system.py` — `/queue` endpoint exposes `gpu_semaphore` status
- 12 new tests in `tests/unit/test_gpu_dependency.py` — all passing

### ~~TDB-008~~: _experiments_router is inline in backend/api/v1/router.py ✅ FIXED 2026-05-25
- Extracted to `backend/api/v1/experiments.py` — full CRUD + new `/archive` toggle endpoint.
  `ExperimentUpdate` schema extended with `is_archived`, `best_map50`, `best_run_id` fields.
  `backend/api/v1/router.py` now imports via `from backend.api.v1 import experiments`.

### ~~TDB-009~~: pytest asyncio mode mismatch ✅ FIXED 2026-05-25
- Removed `[tool.pytest.ini_options]` block from `pyproject.toml` — `pytest.ini` is now
  the single authoritative pytest config. Warning "ignoring pytest config in pyproject.toml"
  eliminated. `asyncio_mode = auto` was already in `pytest.ini`; 328 tests still pass.

### ~~TDB-010~~: MorphologyPipeline._classify_instance imports at call time ✅ FIXED 2026-05-25
- `morphology/application/morphology_pipeline.py` — Removed inline
  `from morphology.domain.geometric import extract_geometric_descriptors as _ged`.
  The function was already imported at module level (line 26); now used directly.

---

### ~~TDB-015~~: `REPO_ROOT = parents[4]` in `containers.py` ✅ FIXED 2026-05-27
- `backend/api/v1/containers.py` was 3 directories deep, not 4
- Resolved to home directory instead of `/path/to/trichome-analysis`
- All `docker compose` operations failed with `[Errno 2] No such file or directory`
- Fixed: `parents[3]`

### ~~TDB-016~~: `.env` crash values ✅ FIXED 2026-05-27
- `VRAM_INFERENCE_BUDGET_GB=""` — empty string → Pydantic `float_parsing` ValidationError on startup
- `DATA_ROOT="/mnt/data/trichome"` — `/mnt/data` not mounted → `PermissionError` in `Settings.ensure_dirs()`
- Fixed in `.env`: `VRAM_INFERENCE_BUDGET_GB="2.0"`, `DATA_ROOT="./data"`

---

## OPEN

### TDB-017: Background task store is in-memory only
- `_bg_tasks: dict[str, BgTask]` in `containers.py` is per-worker, lost on restart
- **Impact**: Task history, running tasks from before restart are invisible
- **Fix**: Persist to SQLite via SQLModel `BgTask` table, or use Redis for multi-worker
- **Priority**: LOW (single-worker dev mode, tasks are short-lived)

---

## LOW (nice to have)

### ~~TDB-011~~: Confidence value object float comparison ✅ FIXED 2026-05-26
- `shared/core/value_objects.py` — Added `__eq__`, `__hash__`, `__le__`, `__ge__`, `__lt__`, `__gt__`
  with `NotImplemented` for unknown types. Comparison with both `Confidence` and `float`/`int` now works.
- 85 active-learning tests (including comparison-dependent assertions) all pass.

### ~~TDB-012~~: Empty `__init__.py` files inconsistent ✅ FIXED 2026-05-26
- 18 `__init__.py` files updated with one-liner docstrings:
  analytics, annotation, inference, research, segmentation, tests subdirectory packages.
- Consistent pattern: `"""<module path> module."""`
**Effort**: 15 minutes
