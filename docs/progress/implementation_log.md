# Implementation Log

---

## 2026-05-26 — TensorRT 10.x Runner + Engine Builder + Tiling Tests + Bug Fix

### WHAT WAS IMPLEMENTED

**TRT stack operational:**
- System packages: TRT 10.14.1.48+cuda13.0 (18 apt packages), pycuda 2026.1 (compiled from source with CUDA 12.6)
- `system_trt.pth` in venv site-packages links system `tensorrt` / `tensorrt_lean` / `tensorrt_dispatch`
- CUDA 12.6 path persisted in `.venv/bin/activate`
- Verified: `RTX 4060, 7.6 GB, compute 8.9, TRT 10.14.1.48`

**`inference/tensorrt_engine/runner.py` (full TRT 10.x rewrite):**
- `_allocate_buffers_trt10()`: `num_io_tensors` + `get_tensor_name/mode/shape/dtype` + `set_tensor_address`
- `_run_engine()`: `execute_async_v3(stream.handle)` (TRT 10.x)
- `_postprocess()`: shape guard (size==0), smarter transpose (shape[0]<100), coordinate clip, NMS
- Context manager, full timing (pre/inf/post ms), `__repr__`

**`inference/tensorrt_engine/builder.py` (new):**
- `build_engine_from_onnx(config, overwrite=False)`: IBuilder + IBuilderConfig + OnnxParser + profile
- `set_memory_pool_limit(WORKSPACE, bytes)` (TRT 10.x API — replaces deprecated max_workspace_size)
- `inspect_engine(path)` → I/O tensor summary dict
- `TRTBuildConfig` dataclass (fp16, workspace_gb, batch profile)

**`inference/tensorrt_engine/__init__.py`:** exports all public symbols

**`tests/unit/test_tensorrt_runner.py`:** 35 passing + 2 GPU-skipped
**`tests/unit/test_inference_tiling.py`:** 57 passing

### WHAT WAS FIXED

**Bug: `detect_tiled()` mutation of shared Detection.bounding_box**
- File: `detection/domain/tiled_inference.py:297-302`
- Cause: `det.bounding_box = global_bbox` mutated the Detection object returned by mock/real detector; when the same object was returned for multiple tiles, coordinates were accumulated (e.g., +640 × 4 times → x_min=2560 on a 1280px image)
- Fix: Create new `Detection(id=uuid4(), bounding_box=global_bbox, ...)` for each tile; degenerate bbox (falls outside image after shift) caught with try/except and dropped
- Consequence: Any code that previously relied on in-place mutation was incorrect; production code is now safe for real models too

### WHY

- TRT 10.x dropped the `num_bindings` / `binding_is_input` / `get_binding_shape` API entirely
- `execute_async_v2` requires explicit `bindings` list; `execute_async_v3` uses `set_tensor_address` → cleaner
- The tiling mutation bug would cause silent coordinate corruption in production inference on large images

### WHAT REMAINS

- TRT E2E test (requires YOLO .pt weights for ONNX export → engine build)
- ONNX runner unit tests
- Frontend TypeScript type-check
- README update (960 tests)

---

## 2026-05-26 — Annotation Stats Tests + VLM Schema Enforcer Tests + README Update

### IMPLEMENTED

**tests/unit/test_annotation_stats.py (NEW — 36 tests, all passing)**

Coverage for `annotation/statistics/stats.py`:
- `TestCohensKappa` (6) — perfect agreement=1.0, complete disagreement≤0, substantial agreement,
  symmetry property, returns float, single-element list is finite
- `TestClassImbalanceRatio` (5) — balanced=1.0, 400:100 ratio=4.0, single class=1.0,
  extreme 10000:1, returns float
- `TestEffectiveImbalance` (4) — balanced classes equal weights, rare class higher weight,
  all positive, keys match input
- `TestAnnotationAggregatorThroughput` (8) — empty→total=0, count, approved/rejected,
  approval_rate, annotations_per_hour, class_distribution, imbalance_ratio, add_single_event
- `TestAnnotationAggregatorQuality` (5) — mean confidence, std=0 on homogeneous, low/high
  confidence counts with configurable threshold, empty→mean=0.0
- `TestAnnotationAggregatorAgreement` (4) — perfect kappa=1.0, agreement_rate∈[0,1],
  note is string, insufficient data→None or float
- `TestCumulativeCurve` (4) — length matches events, monotone, empty safe, required keys

**Bug fixed in production code:**
`annotation/statistics/stats.py` — `get_cumulative_curve()` used `"count"` key;
renamed to `"cumulative_count"` for self-documenting API.

**tests/unit/test_vlm_schema_enforcer.py (NEW — 63 tests, all passing)**

Coverage for `vlm_labeling/prompts/schema_enforcer.py`:
- `TestExtractJson` (6) — plain JSON, markdown-fenced, no-lang-fence, prose-embedded,
  whitespace, raw_text stored
- `TestEmergencyExtract` (4) — quoted kv patterns, returns None on garbage, repaired flag,
  total failure returns defaults
- `TestFieldValidation` (8) — missing required field invalid, missing optional uses default,
  valid complete no errors, invalid enum→default, case-insensitive enum repair, range clamp
  min/max, unknown extra fields passed through
- `TestFieldCoercion` (12) — float from string, int from float, bool from "true"/"false"/"yes",
  list from JSON string, list from comma-sep, nullable_float/int None preserved, nullable_float
  with value, string coercion from int, dict passthrough via _coerce_field direct call
- `TestFractionEnforcement` (5) — already normalised, renormalises outside tolerance, all-zero
  distributed uniformly, slightly above tolerance, warnings added when renormalised
- `TestEnforcementResultProperties` (5) — valid alias, error_message first error, error_message
  None when valid, was_repaired=False default, data returned on valid result
- `TestEnforceMaturity` (6) — valid JSON, invalid stage→unknown, fractions renormalised,
  confidence clamped, completely invalid text→defaults, all 5 stages accepted
- `TestEnforceQuality` (6) — valid JSON, invalid level→unknown, focus_score clamped,
  bool coercion from strings, empty {} returns defaults, all 5 levels accepted
- `TestEnforceMorphology` (8) — valid JSON, missing required→invalid, invalid type→unknown,
  nullable int None, count clamped to 5000, types_present list, density invalid→default,
  all 6 dominant types accepted
- `TestRepairTracking` (3) — clean input (bool baseline), out-of-range marks repaired,
  invalid enum case-insensitive marks repaired

**README.md (UPDATED)**
- All three language sections (EN/DE/ES) updated:
  - Test count: 50 → 807
  - Endpoint count: 72+ → 128+ (all 6 occurrences)
  - Overview: added active learning, calibration, video analysis, annotation statistics
  - API section: added video thumbnail, auto-calibration, GPU queue, analytics, AL query endpoints

### BUG FIXED
`annotation/statistics/stats.py:322` — `"count"` → `"cumulative_count"` in `get_cumulative_curve()`
The old key was not self-documenting; cumulative semantics implied by position alone.
No frontend consumers existed, safe to rename.

### TEST COUNTS
- **807 passed, 2 skipped** — 0 regressions

### WHY
- VLM schema enforcer is the last safety layer before VLM outputs reach the HITL queue.
  Testing it ensures the enforcer correctly handles: markdown-wrapped JSON, missing fields,
  out-of-range values, fraction sum violations — all real failure modes from LLMs.
- Annotation statistics (Cohen's κ, throughput, imbalance ratio) are scientific metrics that
  inform dataset quality decisions. Testing them is required for scientific defensibility.

---

## 2026-05-26 — Phase 15 Research Docs + MetricsChart Historical Overlay

### IMPLEMENTED

**research/microscopy/image_acquisition_methodology.md** (NEW — 15 sections)
- §1 Purpose & scope (why acquisition consistency matters for ML)
- §2 Hardware requirements: microscope types, camera specs (Sony IMX477), illumination CRI/flicker table
- §3 Specimen preparation: material types, slide mounting, artefact prevention
- §4 Illumination protocol: Köhler step-by-step, reflected-light bilateral, stability thresholds (< 2% drift), flat-field correction formula
- §5 Objective selection: µm/px table for IMX477, detector-compatible ranges per trichome class, Rayleigh resolution limits
- §6 Focus/DoF: DOF per NA, focus-on-head protocol, EDOF by z-stacking (2/1 µm step)
- §7 Exposure: 120–160/255 target, < 1% saturation, AWB lock, PNG/TIFF file format policy
- §8 Session metadata: 18 mandatory fields, 9 optional fields
- §9 Calibration per session: stage micrometer procedure, stability check table
- §10 Image quality gates: automated (Laplacian < 50, BRISQUE < 70, saturation < 3%) + manual tags (excellent/good/marginal/reject)
- §11 Acquisition workflow: start/end checklists + per-field protocol
- §12 Maturity imaging: D65, CRI ≥ 85, transmitted light, scientific caveat (verbatim, mandatory)
- §13 Training data diversity requirements: 4 objectives, 2 illumination types, ≥ 10 plants minimum
- §14 Relationship to dataset pipeline (acquisition → ingest → annotate → split → train)
- §15 References (5 citations: Murphy 2001, ISO 9345-1, GUM BIPM, Russ 2011, Köhler tutorial)

**research/evaluation_methodology/trichome_evaluation_methodology.md** (NEW — 9 sections, written in previous sub-sprint)
- Detection: mAP@0.5, per-class AP, size-stratified (domain pixel bins, not COCO thresholds), COCO IoU matching protocol, tiled evaluation
- Calibration: ECE (15 bins), MCE, reliability diagram protocol, calibration training split
- Maturity: weighted/macro F1, confusion matrix, population CI, maturity index formula, scientific caveat
- Segmentation: mIoU, Dice, Boundary IoU, AP_mask, SAM2 evaluation
- Measurement: GUM expanded uncertainty, scale bar traceability
- Dataset splits: plant_id + session_id leakage prevention, 5 leakage risk table, 5 required metadata fields
- Reproducibility: GLOBAL_SEED=42 code block, model checkpointing fields, MLflow artifact schema
- Hardware baseline: RTX 4060 / i5-13400F / 16GB / Ubuntu 22.04 / CUDA 12.x / PyTorch 2.x
- Reporting checklist: 11-item gate

**research/literature_reviews/key_papers_annotated.md** (EXTENDED)
- Added entries 15–19: Gal & Ghahramani 2016 (MC Dropout), Kirsch et al. 2019 (BALD), Lewis & Gale 1994 (least-confidence), Platt 1999 (Platt scaling), Guo et al. 2017 (temperature scaling/ECE), Gretton et al. 2012 (MMD), Bearman et al. 2016 (point annotation), Zhu et al. 2019 (small object eval)

**frontend/src/components/charts/MetricsChart.tsx** (REFACTORED)
- Added `RunSelector` sub-component: dropdown listing completed runs (filtered: != activeRunUuid)
- Added `buildComparisonData(MetricPoint[]) → Map<epoch, cmp:keys>` helper
- Added `mergeData(live, cmpMap)` — epoch-keyed union, missing values left undefined for connect-nulls
- Added `CMP_LINES` config: dark blue/red/green/purple, 2-4 dash, "(ref)" label suffix
- SubChart: conditional `cmp:*` lines via `showComparison` prop
- Delta badge on bestMap50: `(+4.1% vs ref)` in green/red
- `useQuery` for historical metrics with `staleTime=60_000`
- Training page updated: `<MetricsChart runs={runs ?? []} />`

### WHY

**Microscopy methodology**: Without documented acquisition constraints, the ML models are
uninterpretable — you cannot diagnose performance degradation if illumination, objective, or
focus were not documented. 18 mandatory metadata fields ensure leakage-safe dataset splitting
by plant_id/session_id. Flat-field correction and AWB locking are required for maturity
classification reliability.

**Evaluation methodology**: Establishes the scientific contract for model evaluation. Domain-
specific pixel bins replace COCO thresholds (meaningless at 1280px microscopy resolution).
GUM uncertainty reporting makes measurement values traceable and defensible. Reporting
checklist prevents publishing uncalibrated models.

**MetricsChart overlay**: Training comparison against a baseline run is the most practical
diagnostic for detecting regression. The live chart alone cannot identify whether the current
run is improving over a prior best. Run selection is scoped to the same training session
(already fetched), so it adds no extra backend roundtrips.

### TEST STATUS
- 628 passed, 2 skipped, 0 failing (unchanged — no new tests; frontend TypeScript build not executed in CI)

---

## 2026-05-26 — Active Learning Test Coverage (Phase 11)

### IMPLEMENTED

**tests/unit/test_active_learning.py (NEW) — 85 tests, all passing**

Previously: active_learning/ had 2,020 lines across 8 modules, zero test coverage.

Sections:

- **TestEntropyFunctions** (6) — `compute_entropy`, `compute_normalized_entropy`:
  uniform→max, one-hot→~0, entropy monotonic, normalized in [0,1], edge cases
- **TestEntropySampler** (8) — `EntropySampler`:
  score_sample, auto-normalize, score_batch length, select_top_k count+sort+filter,
  quartile spread, dataset stats keys, empty stats
- **TestDisagreementComputation** (6) — `compute_disagreement`:
  empty raises, unanimous low BALD, 50/50 split high BALD, vote_counts sum,
  num_members field, composite in [0,1], KL ≥ 0
- **TestDisagreementSampler** (3) — `DisagreementSampler`:
  compute_all sorted, skips insufficient members, select_top_k respects k
- **TestHardNegativeMiner** (6) — `HardNegativeMiner`:
  basic return, confirmed hard negative (high conf + wrong label),
  correct prediction not hard, low conf hardness=0, labeled_only filter,
  ensemble disagreement score
- **TestComputePriority** (6) — `compute_priority`:
  uncertainty→priority, class_rarity multiplies, age cap at 0.2, manual boost,
  non-negative edge case, returns float
- **TestAnnotationPriorityQueue** (10) — `AnnotationPriorityQueue`:
  push+len, pop highest priority, pop empty→None, peek no-remove, boost elevates,
  boost unknown→False, complete by item_id, complete nonexistent→False,
  remove by item_id, global singleton
- **TestRetrainingTrigger** (13) — `RetrainingTrigger`:
  no trigger below threshold, annotation count fires, performance drop fires,
  drift fires, no trigger when all below, cooldown blocks, mark_retrained resets,
  retrain_count increments, manual always fires, urgency levels valid,
  trigger_history populated, get_status keys, on_trigger callback called
- **TestDriftFunctions** (8) — pure functions in drift.py:
  extract_image_statistics shape+consistent, MMD identical≈0, MMD different>0,
  KS same data no drift, KS strong shift detects, prediction TVD same→no, different→yes,
  severity field in expected set
- **TestDriftDetector** (6) — `DriftDetector`:
  analyze before fit raises, identical data lenient thresholds, 10σ shift detected,
  recommendation non-empty, num_samples fields, prediction distribution test included
- **TestALPipelineConfig** (3) — config defaults, custom values, annotations_per_cycle>0
- **TestUncertaintySamplerMath** (6) — `compute_entropy`, `compute_least_confidence`,
  `UncertaintySampler` from uncertainty.py: uniform, one-hot, confident, uncertain,
  invalid strategy raises, all valid strategies instantiate

### KEY BUGS FOUND AND FIXED IN TESTS
- `AnnotationPriorityQueue.push()` returns `QueueEntry` with auto-generated UUID `item_id`.
  `boost()`, `complete()`, `remove()` all take `item_id`, NOT `sample_id`.
- `QueueStats` field is `pending`, not `total_pending`.
- `compute_ks_drift` returns `DriftResult` with `test_name="KolmogorovSmirnov"` (not "KS").
- `compute_entropy` with epsilon=1e-10 clipping produces ~1e-8 for one-hot (not exact 0).

### TEST COUNTS
- Total: **628 passed**, 2 skipped — 0 regressions

### WHY
- Active learning loop is the primary scientific feedback mechanism for the platform:
  model → inference → uncertainty sampling → annotation queue → HITL review → retraining trigger
- Without tests, drift detection and trigger conditions could silently break
- BALD (Bayesian Active Learning by Disagreement) and entropy sampling are scientifically
  critical; entropy=0 for one-hot and KL≥0 are mathematically mandated properties

---

## 2026-05-25 — GPU Rate Limiting Full Implementation + MLflow Callback Fix

### IMPLEMENTED

**MLflow MlflowClient fix — training/callbacks/metrics_callback.py + checkpoint_callback.py**
- Root cause: both callbacks used `mlflow.start_run(run_id=...)` inside an already-active
  run context. MLflow 3.x treats this as a nested run creation — metrics/artifacts landed
  in a child run, not the original run_id.
- Fix: replaced `with mlflow.start_run(run_id=...):` + fluent API calls with direct
  `mlflow.tracking.MlflowClient().log_metric(run_id, ...)` and
  `MlflowClient().log_artifact(run_id, ...)` calls, bypassing the active-run context.
- Both callbacks now work correctly regardless of whether called inside or outside an
  active run context.

**inference API GPU guard — backend/api/v1/inference.py**
- Extracted `_run_detection(image, conf, iou, model, tiled) → (list[DetectionBox], elapsed_ms)`
  internal helper (no GPU guard — guards live in the endpoint layer).
- `POST /inference/detect` — added `_slot: None = Depends(_gpu_slot)` (gpu_slot_or_429)
- `POST /inference/detect/batch` — added `_slot: None = Depends(_gpu_slot)` (holds slot
  for entire batch — intentional: no concurrent GPU on RTX 4060 8 GB). Batch endpoint
  now calls `_run_detection()` directly instead of `await detect_image()` to avoid
  double-acquire on the semaphore.
- `POST /inference/maturity` — no guard (color_rules = CPU-only; VLM backends raise 503)
- `_gpu_slot` imported with ImportError fallback (no-op generator) for unit-test isolation.
- Docstrings document rate-limiting behavior (429 + Retry-After) for API consumers.

**tests/integration/test_mlflow_callbacks.py — 22 tests ALL PASSING**
- Fixed via MlflowClient fix above; previously 6/22 failed.

**tests/unit/test_inference_api_gpu_guard.py (NEW) — 11 tests**
- TestDetectEndpointGpuGuard (3): 200 on free slot, 429 on busy slot, 429 body has retry_after_s
- TestBatchDetectEndpointGpuGuard (3): 200 free, 429 busy, 422 on >50 images
- TestMaturityEndpointNoGpuGuard (3): color_rules passes when GPU busy, 503 VLM, 422 unknown backend
- TestRunDetectionHelper (2): graceful empty result, correct shape

### TEST COUNTS
- Total: 543 passed, 2 skipped (no regressions)

### WHY
- TDB-007 closed: GPU semaphore now fully propagated to REST inference endpoints.
- MLflow MlflowClient pattern is the correct idiom when run_id is known and you want
  to log to a specific run regardless of the current active-run stack.

---

## 2026-05-25 — Auth Middleware + Alembic Migrations (Phase 2)

### IMPLEMENTED

**APITokenMiddleware — backend/middleware/auth.py (NEW)**
- Single-user API token authentication. Completely transparent when `API_TOKEN` env var is empty.
- Token extraction: `Authorization: Bearer <t>` (priority 1) → `X-API-Key: <t>` (priority 2) → `?api_key=<t>` (priority 3)
- Validation: `hmac.compare_digest` constant-time comparison — no timing side channel
- Excluded paths: /health, /docs, /redoc, /openapi.json, /ws/*, OPTIONS (CORS preflight)
- HTTP 401 on missing/empty token, 403 on invalid token (correct RFC 7235 semantics)
- Wired into `backend/main.py` (registered before GPU guard and request logger)
- `backend/config.py` — added `api_token: str = ""` setting
- `.env.example` — documented API_TOKEN + SECRET_KEY with instructions and `secrets.token_urlsafe(32)` generation hint

**Alembic Database Migrations — alembic/ (NEW)**
- `alembic init` → customized `alembic/env.py`:
  - Imports all SQLModel tables from `backend/models/*`
  - `get_database_url()` reads `DATABASE_URL` env var (SQLite default → PostgreSQL prod)
  - `render_as_batch=True` for SQLite ALTER TABLE compatibility
  - `compare_type=True` for column type change detection in --autogenerate
- `alembic.ini` — date-prefixed migration filenames: `YYYYMMDD_HHMM_<rev>_<slug>.py`
- `alembic/versions/20260525_1319_481ac4c6717a_initial_schema.py` — Full DDL for 8 tables:
  experiments, runs, metrics, datasets, samples, jobs, model_versions, analysis_sessions
  - Idempotent: uses `sa.inspect(bind).get_table_names()` to skip existing tables
  - Indexes: all FK columns + UUID columns + name columns indexed
  - `alembic upgrade head` applied successfully ✅

**Tests — tests/unit/test_auth_middleware.py (NEW: 28 tests)**
- `TestAuthDisabled` (2): routes accessible without token when API_TOKEN empty
- `TestAuthEnabled` (7): Bearer/X-API-Key/query param accepted, 401/403 cases
- `TestExcludedPaths` (3): /health, /system/health, /docs bypass auth
- `TestOptionsBypass` (1): OPTIONS not 401/403
- `TestConstantTimeCompare` (5): equal/different/empty/case-sensitive comparisons
- `TestTokenExtraction` (7): all 3 extraction methods, priority order, None on empty
- `TestMiddlewareState` (3): is_enabled with/without token, whitespace-only → disabled

### FULL SUITE
472 passed, 2 skipped — 0 failures.

---

## 2026-05-25 — Evaluation Pipeline (Phase 12) + Inference Benchmark/Tests (Phase 13)

### IMPLEMENTED

**Evaluation Pipeline — training/evaluation/evaluator.py (NEW)**
- `EvaluationConfig` — model_path, data_yaml, iou_threshold, conf_threshold, imgsz, num_bins,
  max_images, mlflow_run_id, mlflow_tracking_uri, save_artifacts_locally, artifact_output_dir
- `EvaluationResult` — full result: calibration, map50/precision/recall, TP/FP/FN counts,
  artifact paths, eval_time_s. Properties: `ece`, `mce`, `mean_confidence`.
  `to_dict()` for JSON serialisation.
- `ModelEvaluator.evaluate()` — 5-step pipeline:
  1. YOLO `.val()` for standard COCO metrics (mAP50, precision, recall)
  2. Per-image `.predict()` + ground truth label loading (YOLO txt format, de-normalised px coords)
  3. IoU matching via greedy best-confidence-first matching (COCO protocol, class-aware)
  4. ECE/MCE via `compute_calibration()` from `shared/metrics/calibration_metrics`
  5. Artifact save + MLflow logging
- `_compute_iou()` — standard IoU, [x1,y1,x2,y2] boxes, handles zero-area gracefully
- `_match_detections()` — greedy confidence-descending IoU matching, each GT matched ≤ once,
  class mismatch = FP, unmatched GT = FN (not added to calibration pairs)
- `ModelEvaluator._save_artifacts()` — saves `confidence_scores.npy`, `is_correct.npy`,
  `calibration.json` to `runs/eval/{run_id}/`
- `ModelEvaluator._log_to_mlflow()` — logs scalar metrics + numpy artifacts to MLflow run
  - Scalars: `eval/ece`, `eval/mce`, `eval/map50_val`, `eval/mean_confidence`, etc.
  - Artifacts: `predictions/confidence_scores.npy`, `predictions/is_correct.npy`,
    `calibration/calibration.json`

**Evaluate API Endpoint — backend/api/v1/training.py (APPENDED)**
- `POST /training/evaluate` — `EvaluateRequest` / `EvaluateResponse`
- Validates model_path and data_yaml existence (HTTP 404)
- Returns: all metrics + artifact paths + calibration_quality (excellent/good/moderate/poor)
- `_ece_quality()` helper for ECE tier classification
- This endpoint closes the full train→evaluate→analytics loop:
  train → POST /training/start → POST /training/evaluate → GET /analytics/calibration/run/{run_id}

**Inference Benchmark — benchmarks/inference/inference_benchmark.py (NEW)**
- `make_synthetic_image()` — gradient bg + circular blobs + Gaussian noise (realistic microscopy)
- `RunnerBenchmarkResult` + `BatchBenchmarkResult` + `InferenceBenchmarkReport` dataclasses
- `_benchmark_callable()` — warmup + timed loop + numpy percentile stats
- `_benchmark_preprocessing()` — cv2.resize + normalize + HWC→BCHW
- `_benchmark_onnx_runner()` — detects available providers, benchmarks preproc with ONNX env
- `_benchmark_nms()` — simulates YOLO NMS on 8400-anchor synthetic tensor
- `benchmark_batch_preprocessing()` — 1/2/4/8 batch comparison
- CLI: `--quick` (20 runs), `--n-runs`, `--imgsz`, `--output-dir`
- Results saved to `benchmarks/inference/results_YYYYMMDD_HHMMSS.json`

**Benchmark Results (RTX 4060 host, CPU-only pipeline components):**
- Image preprocessing 1280px: **84.3 FPS** (p50=11.5ms, p95=13.9ms)
- NMS post-process 8400 anchors: **6750 FPS** (p50=0.1ms)
- ECE computation 10k predictions 15 bins: **934 FPS** (p50=1.1ms)

**Tests — tests/unit/test_inference.py (NEW: 40 tests)**
- `TestLocalRunnerConfig` (2): defaults, custom values
- `TestLatencyStats` (9): update, mean, min/max, p95, to_dict, buffer capped at 50
- `TestLocalPyTorchRunnerGuards` (7): not-loaded RuntimeError, unload noop, double unload, repr, VRAM=None without CUDA
- `TestParseResults` (6): empty/single/multiple detections, class_id, unknown class fallback, float type
- `TestONNXRunnerConfig` (2): defaults, custom providers
- `TestONNXDetection` (2): creation, coordinate preservation
- `TestTensorRTAvailability` (2): returns bool, graceful False without TRT
- `TestIoUMatching` (10): perfect overlap, no overlap, partial, TP match, FP, class mismatch, GT dedup, empty preds, empty GTs, confidence ordering

### FULL SUITE
444 passed, 2 skipped — 0 failures.

### WHAT REMAINS (next priorities)
1. Backend: Token-based auth (single-user header token)
2. Backend: Alembic PostgreSQL migrations (replace SQLite for production)
3. Docker: `docker-compose.inference.yml`
4. Phase 14: ECE plots in PDF reports
5. Phase 18: Reliability diagram wired to real API data (already built as component)
6. Phase 12: MLflow + W&B integration tests

---

## 2026-05-25 — Analytics API (Phase 14) + Training Form Advanced Controls (Phase 18)

### IMPLEMENTED

**Analytics API — Phase 14**
- `analytics/api/__init__.py` — New package init
- `analytics/api/schemas.py` — Pydantic schemas:
  `CalibrationRequest`, `CalibrationResponse`, `BinStats`,
  `ConfidenceHistogramRequest`, `ConfidenceHistogramResponse`, `ConfidenceBin`
- `analytics/api/router.py` — 3 new endpoints:
  - `POST /analytics/calibration` — ECE/MCE from raw (confidences, is_correct) or MLflow run ID
  - `GET /analytics/calibration/run/{run_id}` — calibration for specific MLflow run
  - `POST /analytics/confidence/histogram` — confidence distribution with descriptive stats
  Key design decisions:
  - `_interpret_ece()` → 4-tier quality labels (excellent/good/moderate/poor) with actionable advice
  - `_build_bin_stats()` → per-bin calibration gap, overconfidence flag, weight
  - MLflow artifact loader (`predictions/confidence_scores.npy` + `predictions/is_correct.npy`)
  - BinStats `gap = conf − acc` (signed), `abs_gap = |conf − acc|`, `is_overconfident`, `is_empty`
- `backend/api/v1/router.py` — Analytics router registered (graceful try/except import)

**Training — Full Hyperparameter Exposure**
- `training/pipelines/yolo_trainer.py` — Added 3 fields to `TrainingConfig`:
  `scale: float = 0.5`, `cos_lr: bool = True`, `augment: bool = True`
  Updated `to_ultralytics_kwargs()` to include all three.
- `backend/api/v1/training.py` `TrainingStartRequest` — Extended from 10 → 28 fields:
  LR schedule: `lrf`, `warmup_epochs`, `cos_lr`
  Regularisation: `weight_decay`, `momentum`
  Early stopping: `patience`
  Augmentation: `augment`, `mosaic`, `close_mosaic`, `hsv_h`, `hsv_s`, `hsv_v`,
  `degrees`, `scale`, `flipud`, `fliplr`
  Training config instantiation updated to forward all new fields.
- `frontend/src/lib/types.ts` `TrainingStartRequest` — Mirrored all 28 fields.
- `frontend/src/app/training/page.tsx` — Full rewrite with advanced controls:
  - `Toggle`, `RangeInput`, `NumberInput`, `Section` sub-components
  - `DEFAULT_FORM` constant with all 28 fields and CTIP-tuned defaults
  - Collapsible sections: LR Schedule, Regularisation & Optimiser, Augmentation
  - Reset-to-defaults button (RotateCcw icon)
  - Augmentation section conditionally renders HSV/flip/scale controls when `augment=true`
  - TypeScript: zero errors (`npm run type-check` clean)

**Bug Fix — ECE test comment**
- `tests/unit/test_calibration_metrics.py::TestECEKnownCase` — Fixed incorrect docstring
  claiming ECE=0.8 (actual: 0.9). Updated assertion and scientific comment.
  (conf=0.9/acc=0.0 → gap=0.9, conf=0.1/acc=1.0 → gap=0.9; ECE = 0.9×0.5 + 0.9×0.5 = 0.9)

**Tests**
- `tests/unit/test_analytics_api.py` — NEW: 31 tests
  `TestCalibrationEndpoint`: 13 tests (200 response, ECE bounds, MCE≥ECE, bin counts,
  overconfident flag, histogram=bin_counts, interpretation string, source, gap math)
  `TestCalibrationValidation`: 6 tests (empty payload, length mismatch, out-of-range,
  num_bins too small/large)
  `TestConfidenceHistogram`: 6 tests (200, bin count, sum, mean range, high-conf fraction, 422)
  `TestInterpretEce`: 6 tests (4 quality tiers, overconfident/underconfident direction)
- `tests/unit/test_training_callbacks.py` — 20 tests (already passing from prior session)
- `tests/unit/test_calibration_metrics.py` — 27 tests (now all passing)

### FULL SUITE
404 passed, 2 skipped — 0 failures.

### WHAT REMAINS
- Reliability diagram frontend component (renders `bins` from `CalibrationResponse`)
- PDF report integration for ECE plots
- MLflow W&B integration tests (Phase 12)
- Token-based auth layer
- Alembic PostgreSQL migrations

---

## 2026-05-25 — Frontend Phase 18 + TDB Sprint (continued)

### IMPLEMENTED

**Frontend — ImageViewer migration (inference page)**
- `frontend/src/app/inference/page.tsx` — Fully migrated from manual SVG overlay + manual zoom
  state to shared `<ImageViewer>` component. Removed `DetectionOverlay`, `imageRef`, zoom state.
  Added `toAnnotationBoxes()` helper converting `DetectionBox[]` → `AnnotationBox[]`.
  Added `DropzonePlaceholder` sub-component. "Change image" shortcut added to bottom bar.

**Frontend — Datasets detail page (ImageViewer lightbox)**
- `frontend/src/app/datasets/[id]/page.tsx` — Replaced local `ImageViewer` modal with proper
  `SampleLightbox` wrapping the shared `ImageViewer`. Added keyboard navigation (←→ Esc),
  prev/next buttons, image position indicator, quality bar colour-coded green/amber/red.

**Frontend — Video page (FrameQualityTimeline)**
- `frontend/src/app/video/page.tsx` — Added `FrameQualityTimeline` SVG bar chart (no recharts
  dependency, pure SVG). Bars colour-coded: green=selected, blue=above gate, amber=duplicate,
  grey=rejected. Horizontal dashed quality-gate line. Added `AnalysisResults` wrapper component
  with 5-KPI grid (total, selected, selection rate, avg quality, process time). Frame list with
  per-frame quality mini-bar. Imports updated: added `BarChart2`, `Download`, removed `CheckCircle2`/`Clock`.

**TDB-003 — MaturityPipeline type assertion**
- `maturity/application/maturity_pipeline.py` — `analyze_crop()` now has explicit `isinstance(result, MaturityLabel)` guard
  raising `TypeError` with descriptive message on API contract breakage.

**TDB-001 — Stage micrometer auto-detection**
- `measurement/calibration/stage_micrometer.py` — New `detect_scale_bar_px()` function and
  `ScaleBarDetectionResult` dataclass. Algorithm: CLAHE → GaussianBlur → Canny → HoughLinesP
  → near-horizontal filter → Y-cluster → span → confidence heuristic.
- `measurement/api/router.py` — New `POST /measurement/profiles/calibrate/auto` endpoint.
  Accepts image upload + `scale_bar_um`, runs detection, saves profile on success.
  Returns `ScaleBarDetectionResponse` with detected flag, confidence, and optional profile schema.
- `tests/unit/test_measurement.py` — 6 new `TestScaleBarDetector` tests. Added `import numpy as np`
  to file-level imports. Total: 45 passing.

**TDB-002 confirmed already fixed** — `CalibrationScale.__post_init__` already validates `um_per_pixel > 0`.
Technical debt log updated.

### TEST RESULTS
- 328 unit tests passing, 2 skipped (GPU-only)
- Frontend TypeScript: 0 errors (npm run type-check clean)

---

## 2026-05-25 — Network Reconfiguration (3001-3010) + Public DDNS

### IMPLEMENTED

**Port Architecture Overhaul (all web services → 3001-3010)**

- `docker/docker-compose.yml` — Complete rewrite
  - Added `nginx` service: `ports: ["3001:80"]`, mounts `./nginx/nginx.conf:ro`
  - Backend: `ports: ["3002:8000"]`, CORS updated for your-domain.com
  - Frontend: `ports: ["3003:3000"]`, API_URL → `http://your-domain.com:3001/api/v1`
  - MLflow: `ports: ["3004:5000"]`
  - Label Studio: `ports: ["3005:8080"]` (annotation profile)
  - CVAT: `ports: ["3006:8080"]` (annotation profile)
  - All services on `trichome-net` bridge network
  - Header comment documents full port layout

- `docker/nginx/nginx.conf` — NEW
  - Path-based routing via nginx (not subdomain-based)
  - Upstreams: backend:8000, frontend:3000, mlflow:5000, label_studio:8080, cvat:8080
  - Rate limiting: api 30r/s (burst 60), inference 5r/s
  - WebSocket upgrade map for `/ws/` and `/api/v1/ws/` paths
  - `client_max_body_size 512M` for microscopy image uploads
  - `server_name your-domain.com localhost _`
  - HTTPS stub commented out (requires TLS provisioning)
  - `proxy_read_timeout 300s` for REST, `3600s` for WebSocket (training runs)

- `docker/docker-compose.training.yml` — Updated
  - MLflow port: `"5000:5000"` → `"3004:5000"`
  - Header comment updated with new port layout

- `docker/docker-compose.annotation.yml` — Updated
  - Label Studio port: `"${LABEL_STUDIO_PORT:-8080}:8080"` → `"3005:8080"`
  - PostgreSQL port: `"${POSTGRES_PORT:-5432}:5432"` → `"3007:5432"`
  - `LABEL_STUDIO_HOST` → `http://your-domain.com:3001/annotation`
  - Header comment updated with new port layout and DDNS access

- `backend/config.py` — Updated
  - `cors_origins` default: adds `http://your-domain.com:3001`, `http://your-domain.com`, `http://localhost:3001`
  - `mlflow_tracking_uri` default: `http://localhost:5000` → `http://localhost:3004`
  - `cvat_url` default: `http://localhost:8080` → `http://localhost:3006`
  - `label_studio_url` default: `http://localhost:8090` → `http://localhost:3005`

- `.env.example` — Rewritten
  - Port layout table in header comment
  - MLFLOW_TRACKING_URI → `http://localhost:3004`
  - CVAT_URL → `http://localhost:3006`
  - LABEL_STUDIO_URL → `http://localhost:3005`
  - CORS_ORIGINS includes your-domain.com entries
  - NEXT_PUBLIC_API_URL → `http://your-domain.com:3001/api/v1`
  - Removed LABEL_STUDIO_PORT and POSTGRES_PORT env vars (fixed in compose)

- `docs/deployment/network_setup.md` ← NEW
  - Full port layout table (3001-3010)
  - DDNS setup: ddclient + No-IP (3 methods)
  - Router port forwarding guide
  - Service access URL table (local / via nginx / via DDNS)
  - Health check commands
  - Nginx config management and reload
  - Rate limiting documentation
  - TLS/HTTPS migration path
  - Security hardening checklist

- `docs/deployment/annotation_stack.md` — Updated
  - All port references updated: 8080→3005, 5432→3007
  - Quick start updated to use `docker compose --profile annotation up -d`
  - CLI commands updated with correct ports (3005, 3002)
  - Access URLs updated to include DDNS variant

- `CLAUDE.md` — Updated Docker section with new port layout and commands

### WHY
- Consolidates all services to 3001-3010 range (single router forward rule)
- Nginx as single public entry point eliminates direct-service exposure
- DDNS via your-domain.com enables remote access without static IP
- Path-based routing through nginx provides centralized rate limiting, security headers, and timeout control
- 512M upload limit at nginx level eliminates per-service configuration drift

---

## 2026-05-25 — Benchmark Suite Sprint

### IMPLEMENTED

**Phase Benchmarks (all 5 modules)**

- `benchmarks/focus/focus_benchmark.py` — already existed; verified working
  - 16 metrics benchmarked: LVAR, MLAP, SLG, LEG, regional, tenengrad, TENGV, AGS,
    FFT ratio, DCT score, Brenner, Vollath F4, composite, composite_regional_4x4,
    heatmap_composite, heatmap_guidance
  - RTX 4060 results: composite 102.8 FPS, Brenner 3016 FPS (fastest), at 512×512px

- `benchmarks/maturity/maturity_benchmark.py` ← NEW
  - 14 functions benchmarked across color, texture, translucency, degradation, pipeline
  - Synthetic trichome crops: 4 HSV-based color stages (clear/cloudy/amber/degraded)
  - Fixed: BGR→RGB conversion for functions expecting RGB input (extract_color_features etc.)
  - Fixed: `use_cnn=False` → `use_analyzer=False` (correct field name)
  - Fixed: `analyze_batch` → `analyze()` (correct method name)
  - RTX 4060 results: pipeline_analyze_crop 94.7 FPS, LBP 122 FPS, detect_color_degradation 34544 FPS

- `benchmarks/morphology/morphology_benchmark.py` ← NEW
  - 10 functions benchmarked: contour, geometric descriptors, stalk/head, classification, density
  - Synthetic masks: 3 morphology types (bulbous/sessile/stalked) cycling
  - Fixed: `classify_morphology_geometric` takes `GeometricFeatures` not mask
  - Fixed: `compute_density_map(image_height=, image_width=)` not `field_height/field_width`
  - Fixed: `MorphologyClassifier.predict_geometric(features=feats)` requires keyword arg
  - Fixed: `Mask.from_uint8()` (not `from_binary()`) for Instance.mask construction
  - RTX 4060 results: pipeline_analyze_single 1189 FPS, density_map 213 FPS

- `benchmarks/measurement/measurement_benchmark.py` ← NEW
  - 9 functions benchmarked: combine_uncertainties, propagate_linear/area/ratio,
    focus_induced_uncertainty, Measurer.measure, estimate_scale_from_objective, pipeline
  - Fixed: propagation function signatures (positional not keyword for first 2 args)
  - Fixed: `propagate_ratio` takes `MeasurementWithUncertainty` objects not floats
  - Fixed: `focus_induced_uncertainty(pixel_size_um=)` not `um_per_pixel=`
  - Fixed: `estimate_scale_from_objective(sensor_pixel_size_um=)` not sensor dimensions
  - RTX 4060 results: all measurement ops sub-millisecond (GUM propagation negligible)

- `benchmarks/video/video_benchmark.py` ← NEW
  - 10 functions benchmarked: score_frame (fast + composite), perceptual_hash,
    hamming_distance, deduplicate_frames, estimate_motion, classify_motion_sequence,
    rank_top_n/diverse/adaptive
  - Pre-builds RankedFrame pool from scored frames for ranking benchmarks
  - RTX 4060 results: score_frame_fast 364 FPS, score_frame_composite 90.8 FPS (4× speedup)

- `docs/progress/benchmark_history.md` ← NEW
  - First RTX 4060 baseline for all 5 pipeline modules
  - Includes production latency analysis against 30 FPS real-time target
  - Notes: all pipelines meet real-time targets; maturity texture (LBP) is main bottleneck

### ALSO IMPLEMENTED

**`tests/integration/test_detect_segment_pipeline.py`** — 40 integration tests
- `TestDetectionDataFlow` (5 tests): Detection entity construction, to_dict contract, confidence flags, batch, BoundingBox area
- `TestSegmentationDataFlow` (5 tests): SegmentPipeline with mocked backend — zero/single/multi detections, result fields, raises-if-not-loaded
- `TestDetectToSegmentPipeline` (4 tests): 10-detection flow, geometry fields, confidence preservation, non-square images
- `TestMaturityIntegration` (5 tests): Instance→MaturityLabel, valid stages, confidence range, population sum-to-1, no cannabinoid claims
- `TestMorphologyIntegration` (3 tests): Instance→MorphologyType, type_distribution keys, batch processed count
- `TestMeasurementIntegration` (3 tests): no-profile fallback, custom 40× profile, population stats structure
- `TestFullPipelineIntegration` (3 tests): full chain produces results, handles mixed valid/invalid, determinism
- `TestScientificConstraints` (5 tests): optical-only stage values, calibrated confidence, no harvest claims, non-negative uncertainty, zero-um_per_pixel rejection
- `TestErrorHandling` (7 tests): empty lists, all-black crops, 1×1 crops, no-mask morphology, empty measurement, focus black image, video noise

### WHAT REMAINS

Priority order for next sprint:
1. **Docker annotation** — docker-compose.annotation.yml (Label Studio + PostgreSQL)
2. **Docker annotation** — docker-compose.annotation.yml (Label Studio + PostgreSQL)
3. **Research documentation** — literature_reviews, evaluation_methodology, calibration_protocols
4. **Frontend improvements** — ImageViewer with overlay, MetricsChart real-time
5. **Technical debt** — TDB-001 (stage micrometer auto-detect), TDB-009 (pytest.ini conflict)

---

## 2026-05-25 — CLI Refactor, Focus Tests, Maturity Pipeline Tests

### IMPLEMENTED

**Phase 17 — CLI Command Modules**
- `apps/cli/commands/detect.py` — Full YOLO detection CLI with progress bar, summary table, tiled mode
- `apps/cli/commands/segment.py` — SAM2 segmentation CLI, summary JSON output
- `apps/cli/commands/maturity.py` — Maturity analysis CLI: json/csv/table formats, stage distribution histogram
- `apps/cli/commands/calibrate.py` — Multi-sub-command calibration CLI: run (interactive+auto), list, show, estimate
  - `run`: interactive pixel measurement mode + Hough-line auto-detect mode
  - `list`: table of all saved profiles from `~/.trichome/profiles/`
  - `show`: detailed view of one profile
  - `estimate`: theoretical µm/pixel from objective + sensor specs
- `apps/cli/commands/benchmark.py` — Benchmark CLI: detection/focus/maturity/morphology/measurement/video/all
  - Uses actual domain modules where available, falls back to synthetic benchmarks
  - Writes per-module JSON to `./benchmarks/`
- `apps/cli/commands/train.py` — Training CLI: start/evaluate/export/list sub-commands
  - `start`: full YOLO training with config merge, MLflow, early stopping, Ctrl+C graceful stop
  - `evaluate`: mAP50/mAP50-95 with ultralytics model.val()
  - `export`: ONNX/TorchScript/TensorRT export
  - `list`: show recent training runs from results.json files
- `apps/cli/commands/video.py` — Video CLI: extract/info/score sub-commands
  - `extract`: full pipeline (sample → score → dedup → rank → save), saves metadata JSON
  - `info`: video metadata + quick quality preview
  - `score`: single image quality score with bar chart
- `apps/cli/commands/annotate.py` — VLM annotation CLI with HITL enforcement
  - `run`: batch auto-labeling, dry-run mode, Label Studio export
  - `review`: show pending review queue
  - `stats`: dataset annotation coverage statistics
- `apps/cli/commands/export.py` — Export CLI: run/list/preview sub-commands
  - `run`: PDF/CSV/JSON export with fallback implementations
  - `list`: list available sessions
  - `preview`: preview session data
- `apps/cli/main.py` — Refactored: thin entry point that imports sub-apps via `_add_subapp()` with graceful degradation
  - Reason: monolithic 450-line main.py → modular command files for maintainability

**Phase 4 — Focus Tests (tests/unit/test_focus.py)**
- 57 unit tests covering all focus metric functions and guidance modules
- TestLaplacianMetrics (8 tests): LVAR, MLAP, SLG, LEG, regional
- TestTenengradMetrics (6 tests): standard, variance, AGS, gradient map
- TestFFTMetrics (7 tests): FFT ratio, DCT score, Brenner, Vollath F4, power spectral slope
- TestCompositeFocusScore (14 tests): composite scorer, quality labels, regional, rank_frames
- TestFocusHeatmap (8 tests): FocusHeatmapResult, score_map, quality fractions
- TestAutofocusGuidance (11 tests): Z-stack analysis, FocusDriftDetector, select_best_frames
- TestFocusConsistency (3 tests): cross-module agreement, determinism, embedded patch

**Phase 5 — Maturity Pipeline Tests (tests/unit/test_maturity_pipeline.py)**
- 39 unit tests covering full maturity pipeline
- TestMaturityPipelineConfig (3 tests): defaults, field validation
- TestAnalyzeCrop (12 tests): MaturityLabel, stage enum, confidence, edge cases (black/white/tiny/large)
- TestAnalyzeBatch (10 tests): batch processing, Instance population, stage distribution, to_dict
- TestFeatureExtraction (8 tests): ColorFeatureVector, TextureFeatureVector, TranslucencyResult, DegradationResult
- TestScientificConstraints (4 tests): no THC/CBD references in any output path
- TestPopulationStats (3 tests): distribution fractions, valid stage keys

### FIXED

**maturity/application/maturity_pipeline.py**
- `from maturity.domain.degradation import detect_degradation` → `assess_degradation`
  (detect_degradation was never implemented — actual function is assess_degradation)
- `from maturity.domain.scientific_rules import classify_by_scientific_rules` → removed
  (function doesn't exist; replaced with `rule_based_maturity_estimate` from color_features)
- `_rule_classify()` now uses `rule_based_maturity_estimate(color)` for rule-based path
  
**focus/metrics/composite.py + focus/guidance/heatmap.py**
- `cv2.applyColorMap(..., cv2.COLORMAP_RdYlGn)` → `cv2.applyColorMap(..., getattr(cv2, "COLORMAP_RdYlGn", cv2.COLORMAP_JET))`
  (COLORMAP_RdYlGn not available in all OpenCV versions)

### WHAT REMAINS

Priority order for next sprint:
1. **Segmentation pipeline tests** — tests/unit/test_segmentation_pipeline.py
2. **Integration tests** — tests/integration/test_detect_segment_pipeline.py  
3. **Benchmarks** — benchmarks/focus/, benchmarks/maturity/, benchmarks/morphology/
4. **Docker annotation** — docker-compose.annotation.yml (Label Studio + PostgreSQL)
5. **Research documentation** — literature_reviews, evaluation methodology, calibration protocols
6. **Frontend improvements** — ImageViewer with overlay, MetricsChart real-time
7. **Technical debt** — TDB-001 (stage micrometer auto-detect), TDB-009 (pytest.ini conflict)

---

## 2026-05-25 — Core Science Module Completion Sprint

### IMPLEMENTED

**Morphology Module (Phase 6) — fully structured**
- `morphology/domain/geometric.py` — GeometricDescriptors dataclass + extract_geometric_descriptors()
  - Shape descriptors: area, perimeter, circularity, elongation, convexity, solidity, compactness
  - PCA-based major/minor axis and orientation
  - to_feature_vector() → 7-dim float32 array for ML classifiers
  - Reason: morphology classification requires structured geometric input; existing classifier.py
    had features defined inside classify_morphology_geometric() with no separate extraction layer
    
- `morphology/domain/stalk_detector.py` — StalkMeasurement + HeadMeasurement + detect_stalk_and_head()
  - Width-profile based stalk/head junction detection
  - Constriction point detection for sessile vs. stalked discrimination
  - Confidence scoring from constriction depth
  - Reason: stalk length is the primary feature distinguishing trichome morphological types;
    no standalone stalk detector existed
    
- `morphology/domain/density_map.py` — DensityMapResult + compute_density_map()
  - Grid-based discrete count map
  - Gaussian KDE surface (scipy or OpenCV fallback)
  - JET colormap heatmap for visualization
  - Physical density (trichomes/mm²) with calibration scale
  - Uniformity index (CV) for distribution quality assessment
  - Reason: population-level spatial analysis is essential for scientific reporting
    
- `morphology/application/morphology_pipeline.py` — MorphologyPipeline
  - Orchestrates: geometric extraction → stalk/head → classification → density
  - CNN + geometric fallback dispatch
  - Batch processing with error recovery
  - Reason: no application-layer pipeline existed; domain objects were isolated

- `morphology/api/router.py` — /morphology/instance, /density, /density/heatmap
- `morphology/schemas/schemas.py` — all Pydantic schemas
- `morphology/classification/classifier.py` — extended: added MorphologyClassifier class
  with has_model, predict_from_crop, predict_geometric methods

**Measurement Module (Phase 7) — fully structured**
- `measurement/domain/profile_manager.py` — MicroscopeProfile + ProfileManager
  - Built-in profiles: 10x/20x/40x/100x generic objectives
  - CRUD with JSON persistence
  - Stage micrometer calibration workflow
  - Uncertainty propagation from calibration precision
  - Reason: no profile management existed; stage_micrometer.py was isolated with no way
    to persist or reuse calibration results

- `measurement/domain/measurer.py` — Measurer + TrichomeMeasurements
  - Pixel → µm conversion with full uncertainty tracking
  - All dimensions: head diameter, stalk length, total height, areas, head/stalk ratio
  - morphology_hint property for sanity checking
  - Reason: the existing stage_micrometer.py only detected scale bars but didn't
    convert measurements to physical units

- `measurement/domain/propagation.py` — GUM-compliant uncertainty propagation
  - combine_uncertainties(): Pythagorean combination for independent sources
  - propagate_linear(): σ_calibration + σ_edge + σ_focus combined
  - propagate_area(): perimeter-based area uncertainty
  - propagate_ratio(): relative uncertainty for ratios (head/stalk)
  - focus_induced_uncertainty(): maps blur score to edge position uncertainty
  - Reference: JCGM 100:2008 (GUM)
  - Reason: scientific validity requires documented uncertainty; no propagation existed

- `measurement/application/measurement_pipeline.py` — MeasurementPipeline + PopulationStats
  - Batch measurement of Instance lists
  - Population statistics: mean, std, median, IQR for all dimensions
  - Write-back to Instance.head_diameter_um, stalk_length_um, total_height_um
  - Reason: full population statistics are required for scientific reporting

- `measurement/api/router.py` — /profiles CRUD + /profiles/calibrate + /measure/mask
- `measurement/schemas/schemas.py` — all Pydantic schemas

**Video Pipeline Module (Phase 8) — fully structured**
- `video_pipeline/domain/extractor.py` — VideoInfo + extract_frames_fixed_rate() + extract_frames_by_timestamps()
  - Generator pattern: one frame in memory at a time
  - Max dimension resize for 4K videos
  - Time range selection
  - Reason: existing video_pipeline.py was monolithic; extractor needed to be
    separate and reusable for streaming use cases

- `video_pipeline/domain/scorer.py` — FrameQualityScore + score_frame()
  - Focus (55%), exposure (25%), noise (20%) composite
  - Immerkaer (1996) noise estimator
  - Histogram-based exposure scoring
  - Reason: structured multi-dimensional quality scoring was embedded in video_pipeline.py;
    extracted to separate, testable module

- `video_pipeline/domain/hasher.py` — perceptual_hash() + hamming_distance() + deduplicate_frames()
  - DCT-based pHash (Zauner 2010)
  - Scene change detection
  - Greedy deduplication
  - Reason: near-duplicate detection is essential for static microscopy videos;
    no standalone module existed

- `video_pipeline/domain/ranker.py` — RankedFrame + rank_top_n() + rank_diverse_n() + rank_adaptive()
  - Three selection strategies with temporal diversity
  - Greedy adaptive selection (max-coverage style)
  - Reason: biological sample analysis requires temporal coverage, not just the best single frame

- `video_pipeline/domain/motion.py` — MotionEstimate + estimate_motion()
  - Lucas-Kanade sparse optical flow
  - RANSAC affine transform estimation
  - Motion sequence classification
  - Reason: motion estimation enables skipping dynamic frames during stage movement

- `video_pipeline/api/router.py` — /video/info, /video/best-frames, /video/analyze
- `video_pipeline/schemas/schemas.py` — all Pydantic schemas

**Maturity Module (Phase 5) — completed missing API/application layer**
- `maturity/application/maturity_pipeline.py` — MaturityPipeline + MaturityPipelineResult
  - Orchestrates: crop extraction → color → texture → translucency → degradation → classify
  - MaturityAnalyzer ensemble with rule-based fallback
  - Population stage distribution
  - Reason: domain components existed but no application pipeline connected them

- `maturity/api/router.py` — /maturity/analyze/crop, /maturity/analyze/population
  - Scientific caveat included in all responses
  - Reason: API router module was empty (1 line __init__.py)

- `maturity/schemas/schemas.py` — full Pydantic schema hierarchy
  - Reason: schemas __init__.py was empty

**Backend Router (Phase 2)**
- `backend/api/v1/router.py` — added focus, maturity, morphology, measurement, video_pipeline routers
  - Reason: new module routers were not registered

**Tests — new unit tests**
- `tests/unit/test_morphology.py` — 29 tests covering geometric, stalk, density, classifier
- `tests/unit/test_measurement.py` — 37 tests covering profile, calibration, uncertainty, measurer
- `tests/unit/test_video_pipeline.py` — 30 tests covering scorer, hasher, ranker, integration
- **Total test suite: 148 passing, 2 skipped (GPU only), 0 failing**

**Progress Tracking**
- `docs/progress/phase_status.md` — full phase status with completion percentages
- `docs/progress/implementation_log.md` — this file
- `CLAUDE.md` — repository guidance document (created at session start)

### FIXED

- `morphology/classification/classifier.py` — extended with MorphologyClassifier class
  (was: classify_morphology_geometric() only, no class wrapper)
  
- `.venv` — recreated pointing to Python 3.12.3 (was: pointing to Python 3.13 which
  was not installed on this system, causing all commands to fail)

### WHAT REMAINS

Priority order for next sprint:

1. **Focus tests** — add tests/unit/test_focus.py
2. **Maturity tests** — add tests/unit/test_maturity_pipeline.py  
3. **CLI commands** — implement detect.py, segment.py, maturity.py, calibrate.py, train.py
4. **Docker annotation compose** — docker-compose.annotation.yml (Label Studio)
5. **Research docs** — complete literature_reviews/key_papers_annotated.md
6. **Frontend components** — ImageViewer with overlay, MetricsChart real-time
7. **TensorRT** — full TensorRT runner implementation
8. **Benchmarks** — RTX 4060 timing benchmarks for all new modules

---

## 2026-05-25 — Docker Inference Stack + WSL2 Docs + TDB-007 GPU Semaphore Dependency

### IMPLEMENTED

**docker/docker-compose.inference.yml (NEW)**
- Inference-only deployment stack: FastAPI inference API + nginx reverse proxy + MLflow (read-only)
- Uses `docker/inference/Dockerfile.inference` (already existed)
- Separate Docker network `trichome-inference-net` to isolate from main stack
- External volumes: `trichome-models` (read-only `:ro`) + `trichome-mlflow` (read-only) — shared with main stack
- New volumes: `trichome-onnx-cache`, `trichome-inference-results`
- Container resource limits: `memory: 8G`, GPU reservation (RTX 4060)
- `ulimits.memlock=-1`: required for pinned GPU memory (CUDA)
- Health check on inference-api: `curl /health` every 30s, 40s start_period for model loading
- Port layout: 3001 (nginx) + 3002 (FastAPI at :8001) + 3004 (MLflow read-only)
- Usage: `docker compose -f docker-compose.inference.yml up -d`

**docker/nginx/nginx.inference.conf (NEW)**
- Lean nginx config for inference-only stack (no frontend, no annotation)
- Routes: `/health` → inference-api (unauthenticated, for LB probes), `/api/v1/` → inference-api,
  `/ws/` → WebSocket upgrade, `/mlflow/` → read-only MLflow UI
- Root `/` returns JSON status: `{"service":"trichome-inference","docs":"/api/v1/docs"}`
- Security headers: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection
- 512M client_max_body_size for large TIFF microscopy images
- 300s read/send timeout for heavy batch inference requests

**backend/dependencies/gpu.py (NEW) — TDB-007 fix**
- `_get_semaphore()` — lazy-init singleton `asyncio.Semaphore(1)`, safe across Python ≥ 3.10 (no
  deprecation from creating at import time without running loop)
- `acquire_gpu_slot(timeout=None)` — async context manager; acquires semaphore, releases on exit
  including exception. Optional timeout raises `asyncio.TimeoutError` if slot not acquired in time.
- `gpu_slot()` — FastAPI `Depends()`-compatible async generator dependency. Wraps
  `acquire_gpu_slot()` for injection into route signatures via `_slot: None = Depends(gpu_slot)`.
- `wire_task_router_semaphore()` — unifies the dependency semaphore with `task_router._gpu_semaphore`
  so REST inference + background training contend on the SAME single semaphore. Called from
  `backend/main.py` lifespan startup.
- `gpu_semaphore_status()` — dict of {available_slots, busy, max_concurrent} for health endpoint.
- Graceful import guard: `try: from backend.dependencies.gpu import gpu_slot except ImportError: yield`
  so maturity/morphology routers remain importable outside the full backend package (e.g. tests).

**backend/dependencies/__init__.py (NEW)** — package init

**maturity/api/router.py (UPDATED)**
- Added `_gpu_slot` import with ImportError fallback
- Added `Depends` to fastapi imports
- `analyze_crop` + `analyze_population` — added `_slot: None = Depends(_gpu_slot)` parameter
- Ensures future CNN/VLM maturity backends are automatically serialised

**morphology/api/router.py (UPDATED)**
- Added `_gpu_slot` import with ImportError fallback
- Added `Depends` to fastapi imports
- `classify_instance` — added `_slot: None = Depends(_gpu_slot)` parameter
- Density/heatmap endpoints (CPU-only) intentionally not guarded

**backend/api/v1/system.py (UPDATED)**
- `/queue` endpoint — added `gpu_semaphore` key to response using `gpu_semaphore_status()`
  → frontend can now show "GPU busy / available" status for both REST inference AND background tasks

**backend/main.py (UPDATED)**
- Lifespan startup: calls `wire_task_router_semaphore()` after DB init to unify semaphores

**docs/deployment/wsl2_setup.md (NEW)**
- Full setup guide: Windows 11, WSL2 Ubuntu 22.04, RTX 4060 GPU passthrough
- Sections: WSL2 prerequisites, CUDA toolkit install (inside WSL2 — NOT driver), uv setup,
  repository setup, WSL2 path considerations, Docker GPU config, NVIDIA Container Toolkit,
  `.wslconfig` tuning (memory=12GB, processors=12, swap=8GB, pageReporting=false),
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, Windows Defender exclusions
- Known issues table: 6 common WSL2 GPU/Docker problems with fixes
- Smoke test: nvidia-smi → pytest → uvicorn → curl health

**tests/unit/test_gpu_dependency.py (NEW — 12 tests)**
- `TestGetSemaphore` (3): lazy creation, singleton, starts with 1 slot
- `TestAcquireGpuSlot` (4): acquire/release, release on exception, serialises concurrent callers,
  timeout raises TimeoutError
- `TestGpuSlotDependency` (1): dependency yields (slot held) then releases (slot freed)
- `TestGpuSemaphoreStatus` (2): idle status {available=1, busy=False}, busy status while held
- `TestWireTaskRouterSemaphore` (2): no-raise when task_router absent, unifies objects

**CLAUDE.md (UPDATED)**
- Docker section: added `docker compose -f docker-compose.inference.yml up -d` with note about
  external volumes

### FULL SUITE
484 passed, 2 skipped — 0 failures.

### WHAT REMAINS
- docker-compose.inference.yml: nginx.inference.conf wired to `trichome-inference-api` container name
  (verify container naming when first running the stack)
- Rate limiting for GPU inference endpoints (429 when too many concurrent requests accumulate)
- Phase 14: ECE plots in PDF reports (ReliabilityDiagram data → analytics/export/pdf_exporter.py)
- Backend: token generation UI page (web page to display/generate API_TOKEN)
- Phase 12: MLflow + W&B integration tests for training callbacks
- Phase 13: TensorRT full implementation (requires NVIDIA TRT SDK)

---

## 2026-05-25 — Phase 14 Completion: ECE Calibration in PDF Reports

### IMPLEMENTED

**analytics/visualization/plotter.py (UPDATED)**
- `plot_reliability_diagram_from_bins(bins, ece, mce, total_samples, title, figsize)` (NEW function)
  - Takes pre-computed BinStats list (from CalibrationResponse) — no raw data needed
  - Left panel: reliability diagram with per-bin bars (orange=overconfident, blue=underconfident)
    with gap fill, diagonal reference, ECE/MCE/quality text annotations
  - Right panel: confidence count histogram (purple bars)
  - Returns matplotlib Figure (Agg backend, safe in server context)
  - Handles bins where `is_empty=True` gracefully (skips them)
  - Uses `matplotlib.patches.Patch` for colour-coded legend

**analytics/export/pdf_exporter.py (UPDATED)**
- `_build_calibration_section(calibration, styles, include_chart)` (NEW helper)
  - Takes CalibrationResponse-compatible dict
  - Returns list of ReportLab flowables: section heading, quality badge, summary table,
    interpretation paragraph, reliability diagram PNG (embedded via `io.BytesIO`), per-bin table
  - BytesIO approach avoids temp-file lifecycle issue where ReportLab reads images lazily
  - Per-bin table has colour-coded status column (overconfident=dark-orange, underconfident=dark-blue)
  - Handles missing optional fields (run_id, interpretation, bins) gracefully
- `export_calibration_pdf(calibration, output_path, model_id, run_id)` (NEW function)
  - Standalone PDF calibration report (no session data needed)
  - Cover page with model_id + optional run_id
  - Full calibration section + methodology + Guo et al. reference
- `export_session_pdf(..., calibration=None)` (UPDATED)
  - New optional `calibration` kwarg — if provided, inserts calibration section after maturity
  - Maturity chart also migrated to BytesIO (same fix — was using unlinked temp files)

**analytics/api/router.py (UPDATED)**
- `POST /analytics/calibration/report` (NEW endpoint)
  - Accepts `CalibrationResponse` JSON body + `?model_id=` query param
  - Returns `application/pdf` binary with `Content-Disposition: attachment`
  - Delegates to `export_calibration_pdf()` via temp dir
  - HTTP 503 if reportlab not installed; HTTP 500 on build failure
  - Enables full workflow: calibrate → download PDF in one API call

**tests/unit/test_pdf_calibration.py (NEW — 16 tests, all passing)**
- `TestPlotReliabilityDiagramFromBins` (5): returns Figure, 2 axes, empty bins OK,
  ECE annotation present, saves to PNG
- `TestBuildCalibrationSection` (5): non-empty list, empty bins OK, missing fields OK,
  ECE paragraph present, per-bin table included
- `TestExportCalibrationPdf` (4): creates PDF, has %PDF magic bytes, creates parent dirs,
  raises ImportError without reportlab
- `TestSessionPdfWithCalibration` (2): session PDF without calibration, with calibration
  (verified larger file size confirming section was included)

### BUGS FIXED
- **ReportLab lazy image loading** (`OSError: Cannot open resource`): `export_session_pdf`
  maturity chart was using `NamedTemporaryFile + os.unlink()` before `doc.build()` — fixed
  to use `io.BytesIO` (eager render, zero temp file lifecycle concerns).

### FULL SUITE
500 passed, 2 skipped — 0 failures.

### WHAT REMAINS
- API rate limiting for GPU inference endpoints (429 when semaphore held)
- Phase 12: MLflow + W&B integration tests
- Phase 13: TensorRT runner
- Frontend: video frame thumbnail serving endpoint
