"""
tests/conftest.py — Shared pytest configuration and fixtures.

Marks:
  - gpu: tests requiring a CUDA GPU
  - slow: tests that take > 30 seconds
  - integration: tests requiring external services (CVAT, MLflow, etc.)
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires NVIDIA GPU with CUDA")
    config.addinivalue_line("markers", "slow: runs > 30 seconds")
    config.addinivalue_line("markers", "integration: requires external services")


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if --no-gpu flag is passed or GPU is unavailable."""
    skip_gpu = pytest.mark.skip(reason="No GPU available or --no-gpu flag set")
    no_gpu = config.getoption("--no-gpu", default=False)

    if no_gpu:
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
        return

    try:
        import torch
        if not torch.cuda.is_available():
            for item in items:
                if "gpu" in item.keywords:
                    item.add_marker(skip_gpu)
    except ImportError:
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


def pytest_addoption(parser):
    parser.addoption("--no-gpu", action="store_true", default=False, help="Skip GPU tests")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_bgr_image():
    """256×256 BGR test image (solid color with noise)."""
    import numpy as np
    np.random.seed(42)
    img = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    return img


@pytest.fixture(scope="session")
def sample_square_mask():
    """Binary mask with a 100×100 square."""
    import numpy as np
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[78:178, 78:178] = 255
    return mask
