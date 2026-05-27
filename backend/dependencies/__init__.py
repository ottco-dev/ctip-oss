"""
backend.dependencies — FastAPI dependency providers.

Available:
  gpu.acquire_gpu_slot        — context manager + Depends() for GPU semaphore
  gpu.gpu_slot                — FastAPI Depends() generator (waits for slot)
  gpu.gpu_slot_or_429         — FastAPI Depends() generator (429 if queue full)
  gpu.configure_gpu_rate_limit — set module-level max_queue_depth at startup
  gpu.gpu_semaphore_status    — dict of slot status for /system/queue health
"""
