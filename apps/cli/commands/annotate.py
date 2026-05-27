"""
apps.cli.commands.annotate — VLM auto-labeling CLI command.

Auto-labels trichome images using Vision Language Models (VLMs).
All proposals are queued for mandatory human review before entering
training data (human-in-the-loop architectural invariant).

Available VLMs (4-bit quantized for RTX 4060 8 GB VRAM):
  moondream    — Moondream-2B (fastest, ~1.5 GB VRAM)
  florence2    — Florence-2-base (balanced)
  qwen2vl      — Qwen2-VL-2B (best accuracy)

Label tasks:
  maturity     — Stage classification: clear | cloudy | amber | degraded
  quality      — Image quality assessment: sharp | acceptable | blurry
  morphology   — Type classification: bulbous | sessile | stalked

Usage:
    trichome annotate crops/ --vlm moondream --task maturity
    trichome annotate image.jpg --task quality
    trichome annotate crops/ --vlm qwen2vl --task morphology --batch-size 8
    trichome annotate review   List pending review queue
"""

from __future__ import annotations

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
    help="Auto-label trichome images using VLMs (human review enforced).",
    add_help_option=True,
)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
_HITL_NOTICE = (
    "Human-in-the-loop is REQUIRED for VLM labeling. "
    "All proposals are queued for review at http://localhost:3000/annotation/review. "
    "VLM proposals are NEVER written directly to training data."
)


def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS)


@app.command("run")
def run_annotation(
    input_path: Path = typer.Argument(..., help="Image or directory to auto-label", metavar="INPUT"),
    output_dir: Path = typer.Option(Path("./output/annotations"), "--output", "-o"),
    vlm: str = typer.Option("moondream", "--vlm", help="VLM: moondream | florence2 | qwen2vl"),
    task: str = typer.Option("maturity", "--task", "-t", help="Task: maturity | quality | morphology"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Images per batch"),
    device: str = typer.Option("cuda", "--device"),
    confidence_threshold: float = typer.Option(0.5, "--conf", help="Minimum confidence to keep proposal"),
    queue_review: bool = typer.Option(True, "--queue/--no-queue", help="Queue for human review (default: on)"),
    export_label_studio: bool = typer.Option(False, "--ls", help="Export proposals to Label Studio format"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without running VLM"),
) -> None:
    """
    Auto-label trichome images using a Vision Language Model.

    Proposals are always queued for human review. This is a mandatory
    architectural constraint — VLM outputs are never written directly to
    training datasets.

    \b
    Examples:
        trichome annotate run crops/ --vlm moondream --task maturity
        trichome annotate run crops/ --vlm qwen2vl --task morphology --batch-size 8
        trichome annotate run image.jpg --task quality --no-queue
    """
    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input not found: {input_path}")
        raise typer.Exit(code=1)

    images = _collect_images(input_path)
    if not images:
        console.print(f"[red]Error:[/red] No images found at {input_path}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Always show HITL notice
    console.print(Panel(_HITL_NOTICE, title="[yellow]⚠ Human Review Required[/yellow]", border_style="yellow"))

    console.print(f"\n[bold cyan]VLM Auto-Labeling[/bold cyan]")
    console.print(f"  VLM:       {vlm}")
    console.print(f"  Task:      {task}")
    console.print(f"  Images:    {len(images)}")
    console.print(f"  Batch:     {batch_size}")
    console.print(f"  Device:    {device}")
    console.print(f"  Min conf:  {confidence_threshold:.0%}")
    console.print()

    if dry_run:
        console.print("[yellow]Dry run — simulating auto-labeling…[/yellow]")
        _simulate_labeling(images, task, output_dir)
        return

    try:
        from vlm_labeling.application.auto_label_pipeline import AutoLabelPipeline, AutoLabelConfig

        config = AutoLabelConfig(
            vlm_model=vlm,
            label_task=task,
            batch_size=batch_size,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        pipeline = AutoLabelPipeline(config)

        results: list[dict] = []
        failed: list[str] = []

        # Process in batches
        image_strs = [str(p) for p in images]
        batches = [image_strs[i:i + batch_size] for i in range(0, len(image_strs), batch_size)]

        for batch in track(batches, description="Labeling…", console=console):
            try:
                batch_results = list(pipeline.run_batch(batch))
                results.extend(batch_results)
            except Exception as e:
                console.print(f"[yellow]  Batch failed: {e}[/yellow]")
                failed.extend(batch)

        # Filter by confidence
        high_conf = [r for r in results if r.get("confidence", 1.0) >= confidence_threshold]
        low_conf = [r for r in results if r.get("confidence", 1.0) < confidence_threshold]

        console.print(f"\n  Labeled:      {len(results)}")
        console.print(f"  High conf:    {len(high_conf)} (≥{confidence_threshold:.0%})")
        console.print(f"  Low conf:     {len(low_conf)} (rejected)")
        console.print(f"  Failed:       {len(failed)}")

        # Save proposals
        proposals_path = output_dir / f"proposals_{task}_{vlm}.json"
        with open(proposals_path, "w") as f:
            json.dump({
                "vlm": vlm,
                "task": task,
                "total": len(results),
                "accepted": len(high_conf),
                "confidence_threshold": confidence_threshold,
                "proposals": high_conf,
                "rejected": low_conf,
                "failed": failed,
                "review_required": True,
                "hitl_notice": _HITL_NOTICE,
            }, f, indent=2)

        console.print(f"\n  [green]Proposals saved:[/green] {proposals_path}")

        if export_label_studio:
            ls_path = output_dir / f"label_studio_{task}.json"
            _export_label_studio_format(high_conf, task, ls_path)
            console.print(f"  [green]Label Studio:[/green] {ls_path}")

        if queue_review:
            console.print(f"\n[cyan]→ Review queue: http://localhost:3000/annotation/review[/cyan]")
            console.print(f"[dim]  {len(high_conf)} proposals awaiting human approval[/dim]")

    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        console.print("[dim]Run: uv pip install -e '.[vlm]' to install VLM dependencies[/dim]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Auto-labeling failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("review")
def show_review_queue(
    queue_dir: Path = typer.Option(Path("./output/annotations"), "--dir"),
    task: Optional[str] = typer.Option(None, "--task"),
) -> None:
    """Show pending human review queue."""
    proposal_files = sorted(queue_dir.glob("proposals_*.json"))
    if task:
        proposal_files = [f for f in proposal_files if f"_{task}_" in f.name]

    if not proposal_files:
        console.print(f"[yellow]No proposals found in {queue_dir}[/yellow]")
        return

    table = Table(title="Review Queue", show_lines=True)
    table.add_column("File", style="cyan")
    table.add_column("Task")
    table.add_column("VLM")
    table.add_column("Pending", justify="right")
    table.add_column("Status")

    total_pending = 0
    for f in proposal_files:
        try:
            data = json.loads(f.read_text())
            n = data.get("accepted", 0)
            total_pending += n
            table.add_row(
                f.name,
                data.get("task", "?"),
                data.get("vlm", "?"),
                str(n),
                "[yellow]awaiting review[/yellow]",
            )
        except Exception:
            table.add_row(f.name, "?", "?", "?", "[red]error[/red]")

    console.print(table)
    console.print(f"\n  Total pending: [bold]{total_pending}[/bold] proposals")
    console.print(f"  Review UI: [cyan]http://localhost:3000/annotation/review[/cyan]")


@app.command("stats")
def annotation_stats(
    dataset_dir: Path = typer.Argument(..., help="Annotation dataset directory"),
) -> None:
    """Show annotation statistics for a labeled dataset."""
    if not dataset_dir.exists():
        console.print(f"[red]Error:[/red] Directory not found: {dataset_dir}")
        raise typer.Exit(code=1)

    # Count files
    images = list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.png"))
    labels = list(dataset_dir.rglob("*.txt")) + list(dataset_dir.rglob("*.json"))

    table = Table(title=f"Annotation Stats: {dataset_dir.name}", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_row("Total images", str(len(images)))
    table.add_row("Label files", str(len(labels)))
    table.add_row("Coverage", f"{len(labels) / max(len(images), 1):.1%}")
    console.print(table)


def _simulate_labeling(images: list[Path], task: str, output_dir: Path) -> None:
    """Simulate labeling output for dry-run."""
    import random
    rng = random.Random(42)
    stage_map = {
        "maturity": ["clear", "mostly_clear", "cloudy", "mostly_amber", "amber"],
        "quality": ["sharp", "acceptable", "blurry"],
        "morphology": ["bulbous", "sessile", "stalked"],
    }
    labels = stage_map.get(task, ["unknown"])

    results = [
        {
            "image": str(img),
            "label": rng.choice(labels),
            "confidence": round(rng.uniform(0.55, 0.99), 3),
            "vlm": "simulated",
            "dry_run": True,
        }
        for img in images
    ]
    path = output_dir / f"proposals_dry_run_{task}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Dry run complete. {len(results)} simulated proposals → {path}[/dim]")


def _export_label_studio_format(proposals: list[dict], task: str, output_path: Path) -> None:
    """Convert proposals to Label Studio import format."""
    ls_tasks = []
    for p in proposals:
        ls_tasks.append({
            "data": {"image": p.get("image", "")},
            "predictions": [{
                "model_version": p.get("vlm", "vlm"),
                "result": [{
                    "type": "choices",
                    "from_name": "label",
                    "to_name": "image",
                    "value": {"choices": [p.get("label", "unknown")]},
                }],
                "score": p.get("confidence", 0.0),
            }],
        })
    with open(output_path, "w") as f:
        json.dump(ls_tasks, f, indent=2)


if __name__ == "__main__":
    app()
