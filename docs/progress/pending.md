# Pending Work

Last updated: 2026-05-29 (Full sprint: DDP trainer, Ollama LLM, temporal tracking, streaming datasets, CLI completions, TensorRT export; 1650 unit tests passing)

---

## HIGH PRIORITY (Science Core)

### Tests
- [x] `tests/unit/test_focus.py` — 57 tests ✅
- [x] `tests/unit/test_maturity_pipeline.py` — 39 tests ✅
- [x] `tests/unit/test_segmentation_pipeline.py` — 38 tests ✅
- [x] `tests/integration/test_detect_segment_pipeline.py` — 40 tests ✅
- [x] `tests/unit/test_active_learning.py` — 85 tests ✅ 2026-05-26
- [x] `tests/integration/test_al_pipeline_integration.py` — 32 tests ✅ 2026-05-26
- [x] `tests/unit/test_vlm_hallucination_filter.py` — 48 tests ✅ 2026-05-26
- [x] `tests/unit/test_annotation_stats.py` — 36 tests ✅ 2026-05-26
- [x] `tests/unit/test_vlm_schema_enforcer.py` — 63 tests ✅ 2026-05-26
- [x] `tests/unit/test_analytics_export.py` — 61 tests ✅ 2026-05-27
- [x] `tests/unit/test_inference_tiling.py` — 57 tests ✅ 2026-05-27

### CLI Commands
- [x] All 9 CLI commands implemented ✅

### Benchmarks
- [x] All 5 module benchmarks implemented ✅

---

## MEDIUM PRIORITY (Platform)

### Docker / Containers
- [x] All docker stacks implemented ✅
- [x] Container management API (16 endpoints) ✅ 2026-05-27
- [x] Background docker compose tasks + Browser Notification API ✅ 2026-05-27
- [x] "Reinstall all" (pull + force-recreate) + per-container pull ✅ 2026-05-27
- [x] Persistent background task store (SQLite-backed task_store.py + TaskRouter.restore_from_db) ✅ 2026-05-29
- [x] `POST /containers/{name}/rm` confirmation dialog in UI (Trash2 + modal in ProcessesTab) ✅ 2026-05-29

### Backend
- [x] PostgreSQL migrations (Alembic) ✅
- [x] Authentication layer ✅
- [x] GPU semaphore infrastructure ✅
- [x] MLflow callback fix ✅

### Research Docs
- [x] All research documents complete ✅ 2026-05-26

---

## LOWER PRIORITY (Future Platform)

### Frontend
- [x] ImageViewer with bounding box + mask overlay ✅
- [x] MetricsChart with real-time WS updates ✅
- [x] MetricsChart historical run overlay — RunSelector, comparison lines, delta badge ✅ 2026-05-26
- [x] ReliabilityDiagram.tsx ✅ 2026-05-26
- [x] Video player — FrameQualityTimeline SVG chart ✅
- [x] Frame-level thumbnail serving — `GET /video/thumbnail/{id}/{n}` + frontend img tag ✅ 2026-05-26
- [x] Calibration analytics page — `analytics/page.tsx` wired to ReliabilityDiagram ✅ 2026-05-26
- [x] Frontend TypeScript type-check pass ✅ 2026-05-27 (0 errors)
- [x] Setup wizard (`/setup`) — OS-style 7-step first-run assistant ✅ 2026-05-27
- [x] In-app wiki (`/wiki`) — 14 pages, EN/DE/ES, WikiRenderer, sidebar search ✅ 2026-05-27
- [x] Container management UI (Processes → Containers tab) ✅ 2026-05-27
  - Background task + polling + Browser Notification API
  - Reinstall all, per-container pull, live log stream
- [x] nginx.conf dynamic `PUBLIC_DOMAIN` env var injection (`nginx.conf.template` + envsubst in docker-compose) ✅ 2026-05-29
- [x] `tests/unit/test_setup_api.py` — 58 tests ✅ 2026-05-27
- [x] `tests/unit/test_containers_api.py` — 47 tests ✅ 2026-05-28
- [x] VLM Provider Switcher UI (`/processes` → VLM Providers tab) ✅ 2026-05-28
  - ProviderCard with tier badge, API key form (persisted to .env), model selector
  - VLMProvidersPanel: local vs remote sections, free-tier highlight, activate/configure
  - TDB-022 fixed: active provider persisted to .env across restarts

### Inference
- [x] TensorRT API (management endpoints, graceful degradation, 23 tests) ✅ 2026-05-29
- [x] NVIDIA Container Toolkit docs + compose file updates ✅ 2026-05-29
- [ ] TensorRT E2E engine build (requires YOLO11s .pt → ONNX export first)
- [x] Batch inference optimization — DetectionBatchQueue ✅ 2026-05-29
  - `backend/tasks/batch_queue.py`: 50ms collection window, max_size=8, EMA stats
  - `POST /inference/detect/queued`: queued endpoint (no per-request semaphore)
  - `GET /inference/batch_queue/stats`: monitoring endpoint
  - `_run_detection_batch()`: true YOLO batch forward pass with sequential fallback
  - Config: `BATCH_QUEUE_WINDOW_MS`, `BATCH_QUEUE_MAX_SIZE` in Settings
  - 41 tests, all passing

### Analytics
- [x] `tests/unit/test_analytics_export.py` — export module test coverage ✅
- [x] `tests/unit/test_model_tests_api.py` — 26 tests (POST/GET/PUT/DELETE + edge cases) ✅ 2026-05-29

### VLM / Annotation
- [x] VLM Ensemble API (`/vlm/ensemble/label`) ✅ 2026-05-29
  - Parallel provider calls via `asyncio.gather`, majority-vote consensus, agreement score
  - GPU semaphore per local provider; remote providers bypass it
  - Prompt registry: `GET /vlm/prompts`, `GET /vlm/prompts/{name}`, `POST /vlm/prompts/validate`
  - 42 tests passing (`tests/unit/test_vlm_ensemble.py`)
- [x] Annotation VLM Config Panel ✅ 2026-05-29
  - Provider + model dropdowns, prompt preset selector, custom system/user prompt textarea
  - Ensemble mode toggle with multi-provider selector
  - `AutoLabelRequest` extended with provider_id, model_id, prompt_name, custom prompts, ensemble fields

### Morphology
- [x] Morphology CNN Trainer (`morphology/training/cnn_trainer.py`) ✅ 2026-05-29
  - EfficientNet-B0 backbone, 4-class head (CAPITATE_STALKED/SESSILE/BULBOUS/NON_GLANDULAR)
  - FP16 mixed precision, early stopping (patience=10), ONNX export
  - REST API: `POST /morphology/training/start|evaluate|export`, `GET /morphology/training/status`
  - 50 tests passing (`tests/unit/test_morphology_cnn.py`)

### Security
- [x] API Token Management UI + API ✅ 2026-05-29
  - `GET /system/token/status`, `POST /system/token/generate`, `POST /system/token/clear`
  - `ApiSecuritySection` in `/settings`: status badge, one-time reveal, masked display, copy, confirm dialogs

### Inference UI
- [x] Batch Inference UI (`/inference/batch`) ✅ 2026-05-29
  - Drag & drop multi-image upload, conf_threshold slider, model selector, tiled toggle
  - Progress tracking, results table, JSON export, live queue stats cards

### Advanced
- [x] Multi-GPU distributed training (DDP + torchrun launcher) ✅ 2026-05-29
  - `training/distributed/`: DDPTrainer, DistributedLauncher, DistributedConfig
  - NCCL→gloo fallback, SyncBN, AMP, gradient accumulation, rank-0 checkpointing
  - REST API: GET/POST /training/distributed/status|start|jobs|stop — 64 tests
- [x] Streaming dataset support (zarr/HDF5) ✅ 2026-05-29
  - `shared/datasets/streaming/`: ZarrDataset, HDF5Dataset, DatasetConverter
  - YOLO→zarr, YOLO→hdf5, zarr↔hdf5 round-trip; thread-safe h5py file handles
  - REST API: POST /datasets/convert, GET /datasets/streaming/stats — 60 tests
- [x] Temporal trichome tracking across video frames ✅ 2026-05-29
  - `video_pipeline/tracking/`: SORT (Kalman+Hungarian), TrackingSession
  - TENTATIVE→CONFIRMED state machine; trajectory export for frontend overlay
  - REST API: 5 endpoints under /video/tracking/ — 87 tests
- [x] Ollama local LLM integration for report narrative generation ✅ 2026-05-29
  - `vlm_labeling/providers/local/ollama_provider.py`: narrative generation
  - THC safety: 15 forbidden cannabinoid keys stripped from all prompts
  - REST API: /ollama/status|models|pull|narrative|config — 65 tests
- [x] CLI shell completions (bash/zsh/fish) ✅ 2026-05-29
  - `trichome completions install|show|uninstall`; auto-detect shell
  - RC file patching; --dry-run; fish completions dir — 40 tests
- [x] TensorRT ONNX export pipeline ✅ 2026-05-29
  - `inference/tensorrt_engine/exporter.py`: .pt→ONNX→.engine full pipeline
  - Graceful degradation without ultralytics/TRT/onnxruntime
  - `trichome convert onnx|tensorrt|validate` CLI — 31 tests
- [ ] TensorRT E2E engine build (requires YOLO11s .pt weights — hardware gated)
