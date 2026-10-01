# Contributing to CTIP

Thanks for helping! CTIP is research software — correctness and honesty matter more than features.

## What helps most

- **Annotated microscope images** under an open licence (CC BY 4.0 or similar), with the microscope, magnification,
  lighting and a session id per image. Quality beats quantity: sharp, consistently lit, carefully labelled.
- **Evaluation results** on your own data or hardware, including failure cases.
- **Bug reports** with steps to reproduce, logs and your environment (`trichome status`).
- Code: fixes, tests, documentation.

## Ground rules

- **No cannabinoid claims.** CTIP measures optical properties. Do not add THC/CBD prediction or wording that implies it.
- **Humans stay in the loop.** VLM or model output must never go into training data without review.
- **No leakage.** Dataset splits stay grouped by imaging session.
- **Reproducible.** Fixed seeds (`GLOBAL_SEED = 42`), deterministic splits, benchmarks with the exact command used.
- **Numbers need evidence.** Accuracy figures in docs must come from a described, reproducible evaluation.

## Development

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install uv && uv pip install -e ".[dev]"
pytest -m "not gpu and not integration"          # must pass
cd frontend && npm ci && npx tsc --noEmit && npm run build
```

- Python: `ruff check .` and type hints; modules follow `domain/ application/ infrastructure/ api/`.
- Every change to logic comes with tests; GPU-only tests are marked `@pytest.mark.gpu`.
- Keep commits focused and describe *why*.

## Pull requests

1. Fork, branch from `main`.
2. Make the change with tests and docs.
3. Open a PR describing the problem, the change and how you verified it. CI must be green.

By contributing you agree that your contribution is licensed under the AGPL-3.0-or-later.
