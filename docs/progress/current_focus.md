# Current Focus

**Updated: 2026-05-26 (TensorRT 10.x runner + builder + tiling tests; 960 tests passing)**

---

## Completed This Sprint

**TensorRT 10.x Stack — COMPLETE ✅**

Full CUDA + TRT stack operational on RTX 4060:
- TensorRT `10.14.1.48+cuda13.0` (18 apt packages) installed + verified
- pycuda `2026.1` compiled and installed in venv (required nvcc in PATH)
- CUDA toolkit 12.6 (`/usr/local/cuda-12.6/bin/nvcc`)
- `system_trt.pth` added to venv site-packages → `import tensorrt` works in venv
- CUDA path persisted in `.venv/bin/activate`

**`inference/tensorrt_engine/runner.py` — REWRITTEN ✅**

Full TRT 10.x API migration:
- `num_io_tensors` + `get_tensor_name/mode/shape/dtype` (replaces deprecated `num_bindings`)
- `set_tensor_address` + `execute_async_v3` (replaces `execute_async_v2`)
- `_postprocess`: smarter transpose detection (shape[0] < 100 guard)
- Empty output guard (size == 0 check before argmax)
- Context manager (`__enter__` / `__exit__`)
- Full timing breakdown: `preprocess_ms`, `inference_ms`, `postprocess_ms`
- `fp16` field in `TRTRunnerConfig`

**`inference/tensorrt_engine/builder.py` — NEW ✅**

ONNX → TRT engine builder:
- `TRTBuildConfig` dataclass (onnx_path, engine_path, imgsz, fp16, workspace_gb, batch sizes)
- `build_engine_from_onnx(config, overwrite=False)` — IBuilderConfig.set_memory_pool_limit (TRT 10 API)
- Dynamic shape optimization profile with min/opt/max batch
- `inspect_engine(path)` — returns I/O tensor summary dict
- `overwrite=False` guard avoids redundant rebuilds

**`tests/unit/test_tensorrt_runner.py` — NEW ✅**

35 tests + 2 GPU-skipped:
- `tensorrt_available()` True/False paths
- `TRTRunnerConfig` defaults + custom
- `TRTDetection` / `TRTResult` field validation
- `TRICHOME_CLASSES` map
- `TensorRTRunner.__repr__`, `.load()` error paths (missing engine, no TRT)
- `.unload()` idempotent, `.__del__()` safe
- Context manager protocol
- `_postprocess()`: empty output, confidence filter, transposed format, coordinate clamp, pad offset
- `TRTBuildConfig` defaults + custom
- `build_engine_from_onnx()` error paths
- `inspect_engine()` error paths
- GPU integration tests (2 tests, skipped unless `@pytest.mark.gpu`)

**`tests/unit/test_inference_tiling.py` — NEW ✅**

57 tests across `detection/domain/tiled_inference.py`:
- `TileConfig` validation (overlap, min_tile_size)
- `TileInfo` width/height/to_global_bbox
- `compute_tiles` — small images, 2×2 grid, last-tile boundary, 4K images, wide/portrait
- `extract_tile` — normal slice + boundary padding
- `is_tile_empty` — variance threshold + skip_empty_tiles=False bypass
- `_cluster_by_iou` — high-IoU merge, separate clusters, chain of 3
- `_fuse_cluster` — confidence-weighted bbox, single-det passthrough
- `_standard_nms_merge` — greedy suppression
- `get_tile_coverage_map` — corners, overlap ≥2, shape match, full coverage
- `detect_tiled` — empty tile skip, local→global translation, diagnostics keys, clipped coords

**Bug Fixed: `tiled_inference.py` detection mutation** ✅

`detect_tiled()` was mutating `det.bounding_box` in-place, causing coordinate accumulation
when the same mock Detection was returned by multiple tiles. Fixed by creating a new
`Detection` object for each tile's coordinate-shifted result.

---

## Previous Sprints

**Analytics Export Tests — COMPLETE ✅**
`tests/unit/test_analytics_export.py` — 61 tests (2 skipped when reportlab installed)

**Annotation Statistics Tests — COMPLETE ✅**
`tests/unit/test_annotation_stats.py` — 36 tests, all passing

**VLM Schema Enforcer Tests — COMPLETE ✅**
`tests/unit/test_vlm_schema_enforcer.py` — 63 tests, all passing

**README.md — UPDATED ✅**
All stale numbers corrected (EN/DE/ES): 50→868 tests, 72+→128+ endpoints

---

## Test Status

- Total: **960 passed**, 4 skipped (GPU-only + reportlab guard), 0 failing
- New this sprint: +92 tests (35 TRT runner + 57 tiling)

---

## Next Priority (in order)

1. **Frontend TypeScript type-check** — `npm run type-check` in `frontend/`
   - Verify MetricsChart refactor + video page type fixes compile clean

2. **ONNX Runtime tests** — `tests/unit/test_onnx_runner.py`
   - ONNX session mock, provider selection, postprocess

3. **TRT engine integration test** — export YOLO11s to ONNX → build .engine → infer
   - Requires: `yolo11s.pt` model weights + Ultralytics export
   - Full E2E: `build_engine_from_onnx` → `TensorRTRunner.infer(synthetic_image)`

4. **README.md update** — reflect 960 tests + TRT runner complete

---

## Blocked

- TRT engine E2E test (requires YOLO11s .pt weights for export)
- SAM2 model inference (requires model weights download, ~2.4 GB)
- PostgreSQL production migration (running on SQLite in dev)

---

## Architecture Invariants (never violate)

- **VLM outputs → NEVER directly to training data** (HITL mandatory)
- **GPU semaphore = 1** (RTX 4060, 8 GB VRAM) — REST endpoints + background tasks share one slot
- **No THC/cannabinoid concentration claims** (optical maturity only)
- **File deletion only inside `/path/to/trichome-analysis/`**
