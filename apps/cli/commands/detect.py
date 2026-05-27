"""
apps.cli.commands.detect — Trichome detection CLI command.

Runs YOLO-based trichome detection on images or directories.
Supports tiled inference for large microscopy images (> 1280px).

Usage:
    trichome detect image.jpg --output /tmp/results/
    trichome detect /data/microscopy/ --model yolo11m --tiled
    trichome detect image.tif --conf 0.3 --crops --device cpu
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import print as rprint

console = Console()

app = typer.Typer(
    help="Run trichome detection on images or video.",
    add_help_option=True,
)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def _collect_images(input_path: Path) -> list[Path]:
    """Collect image files from a path (single file or directory)."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
    return []


@app.command()
def run(
    input_path: Path = typer.Argument(..., help="Input image, video, or directory", metavar="INPUT"),
    output_dir: Path = typer.Option(Path("./output"), "--output", "-o", help="Output directory for results"),
    model: str = typer.Option("yolo11s", "--model", "-m", help="Model variant: yolo11n | yolo11s | yolo11m | yolo11l"),
    confidence: float = typer.Option(0.25, "--conf", "-c", help="Confidence threshold (0–1)", min=0.0, max=1.0),
    iou: float = typer.Option(0.45, "--iou", help="NMS IoU threshold", min=0.0, max=1.0),
    device: str = typer.Option("cuda:0", "--device", "-d", help="Compute device: cuda:0 | cpu"),
    tiled: bool = typer.Option(True, "--tiled/--no-tiled", help="Use sliding-window tiled inference"),
    tile_size: int = typer.Option(1280, "--tile-size", help="Tile size in pixels (tiled mode only)"),
    overlap: float = typer.Option(0.2, "--overlap", help="Tile overlap fraction (tiled mode only)"),
    save_crops: bool = typer.Option(False, "--crops", help="Save per-detection image crops"),
    save_json: bool = typer.Option(True, "--json/--no-json", help="Save detections as JSON"),
    save_viz: bool = typer.Option(True, "--viz/--no-viz", help="Save visualizations"),
    calibrate_confidence: bool = typer.Option(False, "--calibrate", help="Apply Platt scaling confidence calibration"),
) -> None:
    """
    Run trichome detection on images or video.

    Supports YOLO v11 with optional tiled inference for large microscopy images.
    Outputs include JSON detections, visualization overlays, and optional image crops.

    \b
    Examples:
        trichome detect image.jpg
        trichome detect /data/microscopy/ --model yolo11m --output /data/results/
        trichome detect image.tif --no-tiled --conf 0.3 --device cpu
    """
    from shared.logging.logger import configure_logging
    configure_logging(log_level="INFO")

    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input not found: {input_path}")
        raise typer.Exit(code=1)

    images = _collect_images(input_path)
    if not images:
        console.print(f"[red]Error:[/red] No images found at {input_path}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    console.print("\n[bold cyan]Trichome Detection[/bold cyan]")
    console.print(f"  Input:     {input_path}")
    console.print(f"  Output:    {output_dir}")
    console.print(f"  Model:     {model}  (conf={confidence}, iou={iou})")
    console.print(f"  Tiled:     {tiled}" + (f"  ({tile_size}px, {overlap:.0%} overlap)" if tiled else ""))
    console.print(f"  Device:    {device}")
    console.print(f"  Images:    {len(images)}")
    console.print()

    try:
        from detection.infrastructure.yolo_backend import YOLODetector
        from detection.domain.detector import DetectionConfig
        from detection.application.detect_pipeline import DetectionPipeline, PipelineConfig
        from shared.utils.image_utils import load_image
        import json

        det_config = DetectionConfig(
            confidence_threshold=confidence,
            iou_threshold=iou,
            device=device,
            tiled=tiled,
            tile_size=tile_size,
            overlap_ratio=overlap,
        )
        detector = YOLODetector(model_id=model, config=det_config)
        detector.load()
        console.print(f"[dim]Model loaded[/dim]")

        pipeline_cfg = PipelineConfig(
            export_crops=save_crops,
            export_json=save_json,
            export_visualization=save_viz,
            output_dir=str(output_dir),
            apply_calibration=calibrate_confidence,
        )
        pipeline = DetectionPipeline(detector=detector, config=pipeline_cfg)

        # Run detection with progress tracking
        total_detections = 0
        timing_ms: list[float] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Detecting…", total=len(images))

            for img_path in images:
                image = load_image(str(img_path))
                result = pipeline.run(
                    image=image,
                    image_id=img_path.stem,
                    image_path=img_path,
                )
                total_detections += result.num_detections
                timing_ms.append(result.timing.total_ms)
                progress.advance(task, 1)
                progress.update(
                    task,
                    description=f"{img_path.name}: {result.num_detections} detected",
                )

        detector.unload()

        # Summary table
        avg_ms = sum(timing_ms) / len(timing_ms) if timing_ms else 0.0
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        table = Table(title="Detection Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Images processed", str(len(images)))
        table.add_row("Total detections", str(total_detections))
        table.add_row("Avg per image", f"{total_detections / max(len(images), 1):.1f}")
        table.add_row("Avg latency", f"{avg_ms:.1f} ms")
        table.add_row("Throughput", f"{fps:.1f} FPS")
        table.add_row("Output dir", str(output_dir))
        console.print(table)

    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        console.print("[dim]Run: uv pip install -e '.[dev]' to install dependencies[/dim]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Detection failed:[/red] {e}")
        raise typer.Exit(code=1)


# Register as default when invoked directly
@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    input_path: Optional[Path] = typer.Argument(None, metavar="INPUT"),
) -> None:
    """Trichome detect command group."""
    if ctx.invoked_subcommand is None and input_path is not None:
        ctx.invoke(run, input_path=input_path)


if __name__ == "__main__":
    app()
