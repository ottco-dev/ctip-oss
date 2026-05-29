# Current Focus

**Updated: 2026-05-29 (VLM auto-label + Label Studio integration — fully operational)**

---

## Completed This Sprint

### VLM Auto-Label + Label Studio Integration ✅ 2026-05-29

- Moondream2 FP16 (bitsandbytes incompatible, custom F.linear bypasses bnb interception)
- accelerate downgraded 1.13.0 → 0.26.1 (dispatch_model→model.to() conflict with bnb)
- Label Studio auto-connect: env credentials read from `get_settings()` not `os.environ`
- Annotation jobs: job_uuid NOT NULL fix, _fmt_ts datetime/float, UUID lookup
- Frontend: ReviewItem.id string, correct auto-label field names
- **End-to-end verified**: 3-image auto-label → queue populated; LS connected (4 projects)

### CTIP Rebrand + Dark/Light Theme ✅ 2026-05-29

- Renamed "TrichomeLab" → **CTIP** (Cannabis Trichome Intelligence Platform) across layout
- `CtipLogo` SVG: trichome stalk+bulb + microscope body+objective, all using `var(--accent)`
- Dual-theme CSS variable system in `globals.css`: `[data-theme="dark"]` (default) + `[data-theme="light"]`
- Sage green accent palette: `--accent: #4a7c45`, hover, subtle, text variants
- No-flash inline theme script in `layout.tsx` reads `ctip-ui-store` from localStorage before hydration
- `ThemeProvider` component syncs Zustand `theme` state → `document.documentElement` attribute
- Theme toggle (Sun/Moon) in Sidebar bottom section; `tailwind.config.ts` `darkMode: 'selector'`

### System Dashboard Rewrite ✅ 2026-05-29

`SystemStatusTab.tsx` — complete rewrite (~580 lines):
- `RingGauge` + `Sparkline` (gradient fill) components
- `GpuPanel`: 4 rings (VRAM/Compute/Temp/Power), sparkline history, SM count, compute capability
- `QueuePanel`: wired to `gpu_semaphore.busy`, `gpu_queue_depth`, `jobs.pending/running/completed/failed`
- `ServicesPanel`: calls `GET /system/services` every 10s, TCP health check per service
- `ConfigPanel`: hostname, OS, Python, CUDA device, VRAM limit
- `LogTerminal`: `GET /system/logs?limit=20`, color-codes ERROR/WARNING/INFO
- Backend: `GET /system/services` + `GET /system/logs` endpoints added to `system.py`

### nginx Dynamic PUBLIC_DOMAIN ✅ 2026-05-29

- `docker/nginx/nginx.conf.template` with `server_name ${PUBLIC_DOMAIN} localhost _;`
- `docker-compose.yml` nginx service: mounts template, runs `envsubst '$$PUBLIC_DOMAIN'` before nginx
- All `your-domain.com` hardcoded strings replaced with `${PUBLIC_DOMAIN:-localhost}`
- `PUBLIC_DOMAIN` env var passed via `environment:` block, defaults to `localhost`

### Container rm Confirmation Dialog ✅ 2026-05-29

- Trash2 button added to each container card in `ProcessesTab.tsx`
- Confirm modal: backdrop blur, container name display, warning text, Cancel + Remove buttons
- `confirmRm` state gates the `api.delete('/containers/${confirmRm}')` call

### TaskRouter Restart Recovery ✅ 2026-05-29

- `TaskRouter.restore_from_db(db_session)` loads last 24h of `BackgroundJob` records
- Marks pending/running jobs as `failed` (can't recover mid-job after restart)
- Called in `main.py` lifespan startup after `create_all_tables()`

### model-tests API Tests ✅ 2026-05-29

`tests/unit/test_model_tests_api.py` — 26 tests, all passing:
- `TestCreate`: 9 tests (201, UUID, defaults, timestamps, empty graph, nested, unicode, missing graph 422, duplicate names)
- `TestList`: 3 tests (200, newest-first order, no graph field in list)
- `TestLoadDetail`: 4 tests (graph preserved, 404, uuid match)
- `TestUpdate`: 5 tests (name/desc, graph roundtrip, updated_at increment, created_at unchanged, 404)
- `TestDelete`: 5 tests (returns uuid, gone from list, 404 on load, double-delete 404, unknown 404)
- Uses `StaticPool` in-memory SQLite for session isolation

### VLM Provider Switcher UI ✅ 2026-05-28

New "VLM Providers" tab in the Processes page (`/processes`):
- `VLMProvidersPanel` fetches all 8 providers from `GET /vlm/providers`
- Local section (Moondream, Qwen2VL) + Remote API section (OpenAI, Anthropic, Google, Together, Groq, HuggingFace)
- `ProviderCard`: tier badge, status dot, cost/rate info, model selector, Activate button
- Collapsible API key form: persists key to `.env` via `POST /vlm/providers/{id}/configure`
- TDB-022 fixed: `POST /vlm/providers/active` now persists selection to `.env`

### Port-Conflict Dialog + Container test suite ✅ 2026-05-28

- Docker reinstall failure → `status: "port_conflict"` → orange modal dialog in UI
- User picks new port → wired to `.env` + derived vars (MLFLOW_TRACKING_URI etc.) → retry
- `docker compose pull --ignore-buildable` prevents pull failure on locally-built images
- `backend/utils/env_file.py`: shared `.env` read/write (preserves comments/ordering)
- `tests/unit/test_containers_api.py`: 47 tests — all endpoints covered

### Container Management + Background Docker Tasks ✅ 2026-05-27

**Problem solved**: SSE streams kept the HTTP connection open for the full docker compose duration (3–10+ min). If the user navigated away, the operation appeared to fail.

**Solution**: Fire-and-forget asyncio background tasks with polling:
- `POST /containers/compose/up/background` → `{task_id}` returned immediately
- `POST /containers/compose/reinstall/background` → pull + force-recreate (also background)
- `GET /containers/compose/task/{task_id}` → poll every 3s for `status`, `log`, `elapsed_seconds`
- Browser Notification API: `new Notification(...)` fires when task completes, works while navigated away
- `ComposeToast` in-app fallback (bottom-right, 12s auto-dismiss)

Additional endpoints: `POST /containers/{name}/pull`, SSE live log per container, compose config + `.env` reader

**UI additions (processes page)**:
- "Start + Notify" replaces SSE-blocking start
- "Reinstall all" (purple) — full pull + recreate
- Per-container: Download button, pull result banner in log panel
- Notification permission indicator in header

### .env crash fixes ✅ 2026-05-27
- `VRAM_INFERENCE_BUDGET_GB=""` → `"2.0"` — empty string → Pydantic `float_parsing` → backend crash on startup
- `DATA_ROOT="/mnt/data/trichome"` → `"./data"` — `/mnt/data` not mounted → `PermissionError` on `ensure_dirs()`
- `REPO_ROOT parents[4]` → `parents[3]` in `containers.py` — was resolving to the home directory, not repo root

### In-app Wiki (Next.js) ✅ 2026-05-27
14-page multilingual documentation (EN/DE/ES), WikiRenderer, sidebar with search + language switcher, `docs/github-wiki/` for GitHub export.

---

## Previous Sprints

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

- Total: **1053 passed**, 6 skipped (GPU-only + reportlab guard), 0 failing
- This sprint: +26 tests (test_model_tests_api)
- Cumulative: +1027 from previous sprints

---

## Next Priority (in order)

1. **TDB-023: Modal training submission** — stub → real implementation
   - Dataset upload to Modal Volumes, job submission, status polling

2. **TensorRT full implementation** — requires NVIDIA TRT SDK + YOLO11s .pt weights for export

3. **Batch inference optimization** — queue + aggregate requests for GPU throughput

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
