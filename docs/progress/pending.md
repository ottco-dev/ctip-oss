# Pending Work

Last updated: 2026-05-26 (807 tests passing; annotation stats + schema enforcer tests complete)

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

### Docker
- [x] All docker stacks implemented ✅

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
- [ ] Frontend TypeScript type-check pass (`npm run type-check`)

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
