"""
apps.cli.commands.maturity — Trichome maturity analysis CLI command.

Classifies optical maturity stage (Clear → Cloudy → Amber) from color and texture
features. Uses HSV/LAB color analysis, LBP/GLCM/Gabor texture descriptors,
translucency proxy, and oxidation/degradation detection.

SCIENTIFIC CAVEAT:
    Maturity = optical color state ONLY.
    This is NOT a measurement of THC, CBD, or any cannabinoid concentration.
    Harvest recommendations are indicative, not predictive.

Usage:
    trichome maturity crop.jpg
    trichome maturity crops/ --batch --format table
    trichome maturity image.jpg --format csv --output /tmp/results/
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

console = Console()

app = typer.Typer(
    help="Analyze trichome maturity from optical image features.",
    add_help_option=True,
)

SCIENTIFIC_CAVEAT = (
    "SCIENTIFIC CAVEAT: This analysis classifies the OPTICAL COLOR STATE of trichome heads "
    "only. Clear → Cloudy → Amber transitions reflect optical appearance. "
    "This is NOT a measurement of THC, CBD, or any cannabinoid concentration. "
    "Harvest recommendations are indicative only."
)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS)


def _format_stage(stage: str) -> str:
    """Color-code stage for terminal output."""
    stage_map = {
        "clear": "[bright_white]CLEAR[/bright_white]",
        "mostly_clear": "[white]MOSTLY CLEAR[/white]",
        "mixed": "[yellow]MIXED[/yellow]",
        "mostly_cloudy": "[bright_yellow]MOSTLY CLOUDY[/bright_yellow]",
        "cloudy": "[orange1]CLOUDY[/orange1]",
        "mixed_amber": "[dark_orange]MIXED AMBER[/dark_orange]",
        "mostly_amber": "[orange_red1]MOSTLY AMBER[/orange_red1]",
        "full_amber": "[red]FULL AMBER[/red]",
        "degraded": "[bright_red]DEGRADED[/bright_red]",
        "unknown": "[dim]UNKNOWN[/dim]",
    }
    return stage_map.get(stage.lower(), stage.upper())


@app.command()
def run(
    input_path: Path = typer.Argument(..., help="Input image or directory of trichome crops", metavar="INPUT"),
    output_dir: Path = typer.Option(Path("./output/maturity"), "--output", "-o"),
    batch: bool = typer.Option(False, "--batch", "-b", help="Process all images in directory"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json | csv | table"),
    min_confidence: float = typer.Option(0.0, "--min-conf", help="Minimum confidence to report (0=all)"),
    show_features: bool = typer.Option(False, "--features", help="Show extracted feature values"),
    scientific_mode: bool = typer.Option(True, "--scientific/--no-scientific", help="Show scientific caveats"),
) -> None:
    """
    Analyze trichome maturity from optical image features.

    Input should be cropped trichome head images. For full images, run detection
    first to extract crops.

    Features used: HSV color statistics, LAB color statistics, LBP texture,
    GLCM texture (contrast, correlation, homogeneity), Gabor filters,
    translucency proxy (light scattering), oxidation/degradation detection.

    \b
    Examples:
        trichome maturity crop.jpg
        trichome maturity crops/ --batch --format table
        trichome maturity crops/ --batch --format csv --output /tmp/results/
    """
    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input not found: {input_path}")
        raise typer.Exit(code=1)

    images = _collect_images(input_path) if (batch or input_path.is_dir()) else [input_path]
    if not images:
        console.print(f"[red]Error:[/red] No images found at {input_path}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if scientific_mode:
        console.print(Panel(SCIENTIFIC_CAVEAT, title="[yellow]⚠ Scientific Caveat[/yellow]", border_style="yellow"))

    console.print(f"\n[bold cyan]Trichome Maturity Analysis[/bold cyan]")
    console.print(f"  Images:    {len(images)}")
    console.print(f"  Format:    {format}")
    console.print()

    try:
        from maturity.application.maturity_pipeline import MaturityPipeline, MaturityPipelineConfig

        config = MaturityPipelineConfig()
        pipeline = MaturityPipeline(config)

        results: list[dict] = []

        for img_path in track(images, description="Analyzing maturity…", console=console):
            try:
                import cv2
                import numpy as np

                crop_bgr = cv2.imread(str(img_path))
                if crop_bgr is None:
                    console.print(f"[yellow]  ⚠ Cannot read: {img_path.name}[/yellow]")
                    continue
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

                label = pipeline.analyze_crop(crop_rgb)
                stage = label.stage.value if hasattr(label.stage, "value") else str(label.stage)
                confidence = float(getattr(label, "confidence", 0.0))

                if confidence < min_confidence:
                    continue

                record: dict = {
                    "image": img_path.name,
                    "stage": stage,
                    "confidence": round(confidence, 4),
                    "scientific_caveat": SCIENTIFIC_CAVEAT if scientific_mode else "",
                }

                if show_features:
                    record["color_features"] = getattr(label, "color_features", {})
                    record["texture_features"] = getattr(label, "texture_features", {})

                results.append(record)

            except Exception as e:
                console.print(f"[yellow]  ⚠ Failed {img_path.name}: {e}[/yellow]")

        if not results:
            console.print("[yellow]No results above confidence threshold.[/yellow]")
            raise typer.Exit(code=0)

        # Stage distribution
        from collections import Counter
        stage_counts = Counter(r["stage"] for r in results)

        # Output in requested format
        if format == "table":
            table = Table(title="Maturity Results", show_lines=False)
            table.add_column("Image", style="dim", max_width=30)
            table.add_column("Stage")
            table.add_column("Confidence", justify="right")
            for r in results:
                table.add_row(
                    r["image"],
                    _format_stage(r["stage"]),
                    f"{r['confidence']:.1%}",
                )
            console.print(table)

            # Distribution summary
            dist_table = Table(title="Stage Distribution")
            dist_table.add_column("Stage")
            dist_table.add_column("Count", justify="right")
            dist_table.add_column("Fraction", justify="right")
            total = len(results)
            for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
                dist_table.add_row(
                    _format_stage(stage),
                    str(count),
                    f"{count / total:.1%}",
                )
            console.print(dist_table)

        elif format == "csv":
            csv_path = output_dir / "maturity_results.csv"
            fieldnames = ["image", "stage", "confidence"]
            if show_features:
                fieldnames += ["color_features", "texture_features"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
            console.print(f"\n[bold green]CSV saved:[/bold green] {csv_path}")

        else:  # json
            out_data = {
                "total": len(results),
                "stage_distribution": dict(stage_counts),
                "results": results,
                "scientific_caveat": SCIENTIFIC_CAVEAT,
            }
            json_path = output_dir / "maturity_results.json"
            with open(json_path, "w") as f:
                json.dump(out_data, f, indent=2)
            console.print(f"\n[bold green]Maturity analysis complete[/bold green]")
            console.print(f"  Analyzed: {len(results)} images")
            console.print(f"  Results:  {json_path}")

        # Distribution summary for non-table formats
        if format != "table":
            console.print("\n[bold]Stage Distribution:[/bold]")
            for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
                pct = count / len(results) * 100
                bar = "█" * int(pct / 5)
                console.print(f"  {stage:20s}  {count:3d}  {pct:5.1f}%  {bar}")

    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Maturity analysis failed:[/red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
