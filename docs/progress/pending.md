# Pending Work

Last updated: 2026-05-27 (container management + background docker tasks + wiki shipped; 960 tests passing)

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
- [ ] `tests/unit/test_analytics_export.py` — json_exporter, csv_exporter, pdf_generator (COCO format, benchmark export, session export)
- [ ] `tests/unit/test_inference_tiling.py` — tile stitching, NMS, overlap handling, ONNX fallback

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
- [ ] Persistent background task store (tasks lost on backend restart — in-memory only)
- [ ] `POST /containers/{name}/rm` confirmation dialog in UI

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
- [ ] nginx.conf dynamic `PUBLIC_DOMAIN` env var injection (currently hardcoded `server_name`)
- [ ] `tests/unit/test_setup_api.py` — pytest unit tests for config API endpoints
- [ ] `tests/unit/test_containers_api.py` — pytest tests for containers router

### Inference
- [ ] TensorRT full implementation (requires NVIDIA TRT SDK — not in venv)
- [ ] Batch inference optimization

### Analytics
- [ ] `tests/unit/test_analytics_export.py` — export module test coverage

### Advanced
- [ ] Multi-GPU distributed training support
- [ ] Streaming dataset support (zarr/HDF5)
- [ ] Temporal trichome tracking across video frames
- [ ] Ollama local LLM integration for report narrative generation
