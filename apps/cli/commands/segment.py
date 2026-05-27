"""
apps.cli.commands.segment — Trichome instance segmentation CLI command.

Two-stage pipeline:
  1. YOLO detection → bounding boxes as SAM2 prompts
  2. SAM2-tiny or MobileSAM → pixel-precise instance masks

Outputs: mask PNGs, JSON with polygon vertices, optional visualization overlay.

Usage:
    trichome segment image.jpg --output /tmp/masks/
    trichome segment image.jpg --model sam2-base --conf 0.3
    trichome segment /data/images/ --device cpu --no-viz
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()

app = typer.Typer(
    help="Instance segmentation of trichomes using SAM2.",
    add_help_option=True,
)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS)


@app.command()
def run(
    input_path: Path = typer.Argument(..., help="Input image or directory", metavar="INPUT"),
    output_dir: Path = typer.Option(Path("./output/segments"), "--output", "-o"),
    detection_model: str = typer.Option("yolo11s", "--det-model", help="YOLO model for box prompts"),
    sam_model: str = typer.Option("sam2-tiny", "--model", "-m", help="SAM model: sam2-tiny | sam2-base | mobile-sam"),
    conf: float = typer.Option(0.25, "--conf", help="Detection confidence threshold"),
    device: str = typer.Option("cuda", "--device", "-d", help="cuda | cpu"),
    min_mask_area: int = typer.Option(100, "--min-area", help="Minimum mask area in pixels (filter noise)"),
    fill_holes: bool = typer.Option(True, "--fill-holes/--no-fill-holes", help="Fill holes in masks"),
    save_masks: bool = typer.Option(True, "--masks/--no-masks", help="Save individual mask PNGs"),
    save_polygons: bool = typer.Option(True, "--polygons/--no-polygons", help="Save polygon JSON"),
    save_viz: bool = typer.Option(True, "--viz/--no-viz", help="Save visualization overlay"),
) -> None:
    """
    Segment trichomes with SAM2 prompted by YOLO detections.

    The two-stage pipeline first detects bounding boxes with YOLO, then uses
    SAM2 to generate pixel-precise instance masks within each box.

    \b
    Examples:
        trichome segment image.jpg
        trichome segment image.jpg --model sam2-base --output /data/masks/
        trichome segment /data/images/ --device cpu --no-viz
    """
    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input not found: {input_path}")
        raise typer.Exit(code=1)

    images = _collect_images(input_path)
    if not images:
        console.print(f"[red]Error:[/red] No images found at {input_path}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_masks:
        (output_dir / "masks").mkdir(exist_ok=True)
    if save_viz:
        (output_dir / "viz").mkdir(exist_ok=True)

    console.print("\n[bold cyan]Trichome Segmentation[/bold cyan]")
    console.print(f"  Input:     {input_path}")
    console.print(f"  Output:    {output_dir}")
    console.print(f"  Detection: {detection_model}  (conf={conf})")
    console.print(f"  SAM model: {sam_model}")
    console.print(f"  Device:    {device}")
    console.print(f"  Images:    {len(images)}")
    console.print()

    try:
        from segmentation.application.segment_pipeline import SegmentationPipeline, SegmentationConfig

        config = SegmentationConfig(
            segmentor_backend=sam_model,
            device=device,
            conf_threshold=conf,
            min_mask_area_px=min_mask_area,
            fill_holes=fill_holes,
            save_masks=save_masks,
            save_polygons=save_polygons,
            save_visualization=save_viz,
        )
        pipeline = SegmentationPipeline(config)

        all_results: list[dict] = []
        total_instances = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Segmenting…", total=len(images))

            for img_path in images:
                result = pipeline.run(
                    str(img_path),
                    output_dir=str(output_dir),
                    visualize=save_viz,
                )
                n = result.get("num_instances", 0)
                total_instances += n
                all_results.append({"image": img_path.name, "num_instances": n})
                progress.advance(task, 1)
                progress.update(task, description=f"{img_path.name}: {n} instances")

        # Write summary JSON
        summary_path = output_dir / "segmentation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "total_images": len(images),
                    "total_instances": total_instances,
                    "per_image": all_results,
                    "config": {
                        "sam_model": sam_model,
                        "conf": conf,
                        "device": device,
                        "min_mask_area_px": min_mask_area,
                    },
                },
                f, indent=2,
            )

        table = Table(title="Segmentation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Images processed", str(len(images)))
        table.add_row("Total instances", str(total_instances))
        table.add_row("Avg per image", f"{total_instances / max(len(images), 1):.1f}")
        table.add_row("Output dir", str(output_dir))
        console.print(table)

    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        console.print("[dim]SAM2 weights may need to be downloaded. See README.[/dim]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Segmentation failed:[/red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
