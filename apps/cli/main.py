"""
apps.cli.main — TrichomeLab CLI entry point.

All sub-commands are implemented in apps/cli/commands/:
    detect.py   — trichome detect    Run YOLO trichome detection
    segment.py  — trichome segment   SAM2 instance segmentation
    maturity.py — trichome maturity  Optical maturity classification
    calibrate.py — trichome calibrate  Microscope pixel→µm calibration
    benchmark.py — trichome benchmark  Pipeline performance benchmarks
    train.py    — trichome train     YOLO training + evaluation
    video.py    — trichome video     Video frame extraction + ranking
    annotate.py — trichome annotate  VLM auto-labeling (HITL enforced)
    export.py   — trichome export    PDF/CSV/JSON report generation

Plus built-in commands:
    serve       — Start FastAPI backend server
    status      — Show GPU / API status
    version     — Print version info

Usage:
    trichome detect image.jpg --output /tmp/results/
    trichome segment image.jpg --model sam2-base
    trichome maturity crops/ --batch --format table
    trichome calibrate run stage_mic.tif --objective 40 --spacing 10
    trichome benchmark focus
    trichome train start --data /data/trichome.yaml --epochs 100
    trichome video extract video.mp4 --top-n 100
    trichome annotate run crops/ --vlm moondream --task maturity
    trichome export run session_id --formats pdf,csv
    trichome serve --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# ── Main app ──────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="trichome",
    help="[bold]TrichomeLab CLI[/bold] — Cannabis Trichome Analysis Platform",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

VERSION = "0.2.0"

# ── Sub-command groups (each from its own module) ─────────────────────────────

def _add_subapp(name: str, module_path: str, help_text: str) -> None:
    """Register a sub-app from a commands module, with graceful degradation."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        sub_app = getattr(mod, "app", None)
        if sub_app is not None:
            app.add_typer(sub_app, name=name, help=help_text)
    except Exception as e:
        # Register a stub command instead of failing silently
        @app.command(name)
        def _stub(ctx: typer.Context) -> None:  # type: ignore[misc]
            console.print(f"[red]Command '{name}' failed to load:[/red] {e}")
            console.print(f"[dim]Run: uv pip install -e '.[dev]'[/dim]")
            raise typer.Exit(code=1)
        _stub.__doc__ = f"{help_text} [unavailable: {e}]"


# Register all sub-command groups
_add_subapp("detect",    "apps.cli.commands.detect",    "Run YOLO trichome detection on images.")
_add_subapp("segment",   "apps.cli.commands.segment",   "Instance segmentation with SAM2.")
_add_subapp("maturity",  "apps.cli.commands.maturity",  "Optical maturity classification (HSV/LAB/LBP/GLCM).")
_add_subapp("calibrate", "apps.cli.commands.calibrate", "Microscope pixel-to-µm calibration.")
_add_subapp("benchmark", "apps.cli.commands.benchmark", "Pipeline performance benchmarks (FPS, mAP, VRAM).")
_add_subapp("train",     "apps.cli.commands.train",     "YOLO training, evaluation, and export.")
_add_subapp("video",     "apps.cli.commands.video",     "Extract and rank best frames from microscopy video.")
_add_subapp("annotate",  "apps.cli.commands.annotate",  "VLM auto-labeling with mandatory human review.")
_add_subapp("export",    "apps.cli.commands.export",    "Export session results to PDF, CSV, JSON reports.")

# ── Built-in commands ─────────────────────────────────────────────────────────

@app.command()
def version() -> None:
    """Print TrichomeLab CLI version and GPU info."""
    rprint(f"[bold]TrichomeLab CLI[/bold]  v{VERSION}")
    rprint("[dim]Cannabis Trichome Intelligence Platform — detection · maturity · morphology · measurement[/dim]")

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            rprint(f"GPU:   [green]{name}[/green]  ({vram:.1f} GB VRAM)")
        else:
            rprint("GPU:   [yellow]CUDA not available — CPU mode[/yellow]")
    except ImportError:
        rprint("GPU:   [red]PyTorch not installed[/red]")

    try:
        import cv2
        rprint(f"OpenCV: {cv2.__version__}")
    except ImportError:
        rprint("OpenCV: [red]not installed[/red]")


@app.command()
def status() -> None:
    """Show system status: GPU VRAM, loaded models, API availability."""
    console.print("\n[bold]TrichomeLab System Status[/bold]\n")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_total = props.total_memory / 1e9
            vram_used = torch.cuda.memory_allocated(0) / 1e9
            vram_pct = vram_used / vram_total * 100
            bar = "█" * int(vram_pct / 5) + "░" * (20 - int(vram_pct / 5))
            console.print(f"  GPU:   [green]{props.name}[/green]")
            console.print(f"  VRAM:  {vram_used:.1f}/{vram_total:.1f} GB  [{bar}] {vram_pct:.0f}%")
        else:
            console.print("  GPU:   [yellow]CUDA not available — running CPU mode[/yellow]")
    except ImportError:
        console.print("  GPU:   [red]PyTorch not installed[/red]")

    console.print()

    # Backend API
    _check_service("Backend API", "http://localhost:8000/api/v1/system/health")

    # Label Studio
    _check_service("Label Studio", "http://localhost:8080/health")

    # MLflow
    _check_service("MLflow", "http://localhost:5000/health")

    console.print()
    console.print("[dim]Start backend: trichome serve[/dim]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev only)"),
    workers: int = typer.Option(1, "--workers", "-w", help="Uvicorn workers (reload=True forces 1)"),
    log_level: str = typer.Option("info", "--log-level", help="Log level: debug | info | warning | error"),
) -> None:
    """
    Start the TrichomeLab FastAPI backend server.

    \b
    Examples:
        trichome serve
        trichome serve --port 8080 --reload
        trichome serve --workers 2 --log-level debug
    """
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red]  Run: uv pip install uvicorn")
        raise typer.Exit(code=1)

    console.print(f"[bold]Starting TrichomeLab API[/bold]")
    console.print(f"  URL:    [cyan]http://{host}:{port}[/cyan]")
    console.print(f"  Docs:   [cyan]http://{host}:{port}/docs[/cyan]")
    console.print(f"  Reload: {reload}")
    console.print(f"  Workers: {workers if not reload else 1}")
    console.print()

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=log_level,
    )


# ── Utilities ─────────────────────────────────────────────────────────────────

def _check_service(name: str, url: str) -> None:
    """Check if a service is running by hitting its health endpoint."""
    try:
        import urllib.request
        urllib.request.urlopen(url, timeout=2)
        console.print(f"  {name}:  [green]Running[/green]  ({url})")
    except Exception:
        console.print(f"  {name}:  [yellow]Not running[/yellow]  ({url})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point (registered via pyproject.toml scripts)."""
    app()


if __name__ == "__main__":
    main()
