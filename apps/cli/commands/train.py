"""
apps.cli.commands.train — Model training CLI command.

Manages YOLO training runs with full experiment tracking support.
Integrates with MLflow and W&B if configured.

Sub-commands:
  start     — Start a new training run
  resume    — Resume a stopped training run
  stop      — Request graceful stop of running training
  status    — Show training status (requires running API)
  export    — Export trained model to ONNX/TensorRT
  evaluate  — Evaluate a trained model on a dataset

Usage:
    trichome train start --data /data/trichome.yaml --epochs 100
    trichome train start --config configs/training/yolo11s_detection.yaml
    trichome train export --model runs/train/exp/best.pt --format onnx
    trichome train evaluate --model best.pt --dataset /data/test/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

app = typer.Typer(
    help="Manage YOLO trichome detection training runs.",
    add_help_option=True,
)


@app.command("start")
def start_training(
    data_yaml: str = typer.Option("", "--data", "-d", help="Dataset YAML path"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Training config YAML"),
    model: str = typer.Option("yolo11s", "--model", "-m", help="Base model: yolo11n | yolo11s | yolo11m | yolo11l"),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Training epochs"),
    batch: int = typer.Option(16, "--batch", "-b", help="Batch size"),
    imgsz: int = typer.Option(640, "--imgsz", help="Input image size"),
    device: str = typer.Option("cuda:0", "--device", help="Training device"),
    workers: int = typer.Option(4, "--workers", help="DataLoader workers"),
    project: str = typer.Option("trichome_runs", "--project", help="MLflow/W&B project name"),
    run_name: str = typer.Option("", "--run-name", help="Experiment run name"),
    resume: str = typer.Option("", "--resume", help="Resume from checkpoint path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without training"),
    patience: int = typer.Option(50, "--patience", help="Early stopping patience (0=disabled)"),
    save_period: int = typer.Option(10, "--save-period", help="Save checkpoint every N epochs"),
) -> None:
    """
    Start a YOLO training run for trichome detection.

    Uses RTX-4060-optimized defaults: mixed precision, cache images,
    GradScaler for AMP, cosine LR decay.

    \b
    Examples:
        trichome train start --data /data/trichome.yaml
        trichome train start --data /data/trichome.yaml --model yolo11m --epochs 200
        trichome train start --config configs/training/yolo11s_detection.yaml
    """
    try:
        from training.pipelines.yolo_trainer import YOLOTrainer, TrainingConfig
    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        raise typer.Exit(code=1)

    # Build config
    cfg = TrainingConfig()
    if data_yaml:
        cfg.data_yaml = data_yaml
    if config_path and config_path.exists():
        _merge_yaml_config(cfg, config_path)
    if epochs != 100:
        cfg.epochs = epochs
    if batch != 16:
        cfg.batch_size = batch
    if imgsz != 640:
        cfg.imgsz = imgsz
    if device != "cuda:0":
        cfg.device = device
    if model != "yolo11s":
        cfg.model_variant = model
    if project:
        cfg.project = project
    if run_name:
        cfg.name = run_name
    if resume:
        cfg.resume = resume
    if patience != 50:
        cfg.patience = patience
    cfg.workers = workers
    cfg.save_period = save_period

    if not cfg.data_yaml:
        console.print("[red]Error:[/red] --data is required (dataset YAML path)")
        console.print("[dim]Example: trichome train start --data /data/trichome.yaml[/dim]")
        raise typer.Exit(code=1)

    # Print config summary
    table = Table(title="Training Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")
    table.add_row("Model", cfg.model_variant)
    table.add_row("Dataset", cfg.data_yaml)
    table.add_row("Epochs", str(cfg.epochs))
    table.add_row("Batch size", f"{cfg.batch_size} (eff: {cfg.effective_batch_size})")
    table.add_row("Image size", str(cfg.imgsz))
    table.add_row("Device", cfg.device)
    table.add_row("Workers", str(cfg.workers))
    table.add_row("Patience", str(cfg.patience) if cfg.patience else "disabled")
    table.add_row("Project", cfg.project)
    console.print(table)

    if dry_run:
        console.print("\n[yellow]Dry run — configuration valid, not starting training[/yellow]")
        return

    console.print(f"\n[bold]Starting training…[/bold]")
    console.print(f"[dim]Monitor in MLflow: http://localhost:5000[/dim]")
    console.print(f"[dim]Press Ctrl+C to request graceful stop[/dim]\n")

    try:
        trainer = YOLOTrainer(cfg)
        result = trainer.train()

        console.print(f"\n[bold green]Training Complete![/bold green]")
        _print_training_result(result)

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user (Ctrl+C)[/yellow]")
        console.print("[dim]Training state may be resumable with --resume[/dim]")
        raise typer.Exit(code=0)
    except Exception as e:
        console.print(f"\n[red]Training failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate_model(
    model_path: Path = typer.Argument(..., help="Path to trained model (.pt or .onnx)"),
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Dataset directory or YAML"),
    conf: float = typer.Option(0.25, "--conf"),
    iou: float = typer.Option(0.5, "--iou"),
    device: str = typer.Option("cuda:0", "--device"),
    output: Path = typer.Option(Path("./eval_results"), "--output", "-o"),
    split: str = typer.Option("val", "--split", help="Dataset split: train | val | test"),
) -> None:
    """
    Evaluate a trained model on a labeled dataset.

    Computes mAP50, mAP50-95, precision, recall, F1 per class.
    """
    if not model_path.exists():
        console.print(f"[red]Error:[/red] Model not found: {model_path}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]Model Evaluation[/bold cyan]")
    console.print(f"  Model:   {model_path}")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Split:   {split}")

    try:
        from ultralytics import YOLO
        import json

        model = YOLO(str(model_path))
        with console.status("Running evaluation…"):
            metrics = model.val(
                data=str(dataset),
                conf=conf,
                iou=iou,
                device=device,
                split=split,
                verbose=False,
            )

        results = {
            "model": str(model_path),
            "dataset": str(dataset),
            "split": split,
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        }

        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        for k, v in results.items():
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
        console.print(table)

        output.mkdir(parents=True, exist_ok=True)
        result_path = output / f"eval_{model_path.stem}_{split}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n  [green]Saved:[/green] {result_path}")

    except ImportError:
        console.print("[red]ultralytics not installed.[/red] Run: pip install ultralytics")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Evaluation failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("export")
def export_model(
    model_path: Path = typer.Argument(..., help="Path to trained .pt model"),
    format: str = typer.Option("onnx", "--format", "-f", help="Export format: onnx | torchscript | tflite | engine"),
    imgsz: int = typer.Option(640, "--imgsz", help="Input image size for export"),
    device: str = typer.Option("cpu", "--device", help="Export device (cpu | cuda:0)"),
    half: bool = typer.Option(False, "--half", help="FP16 quantization (ONNX/TensorRT only)"),
    simplify: bool = typer.Option(True, "--simplify/--no-simplify", help="ONNX simplification"),
    dynamic: bool = typer.Option(False, "--dynamic", help="Dynamic axes for ONNX"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path (default: model dir)"),
) -> None:
    """
    Export trained YOLO model to ONNX, TorchScript, or TensorRT.

    \b
    Examples:
        trichome train export best.pt --format onnx
        trichome train export best.pt --format engine --half --device cuda:0
    """
    if not model_path.exists():
        console.print(f"[red]Error:[/red] Model not found: {model_path}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]Model Export[/bold cyan]")
    console.print(f"  Model:   {model_path}")
    console.print(f"  Format:  {format}")
    console.print(f"  imgsz:   {imgsz}")
    console.print(f"  Half:    {half}")

    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        with console.status(f"Exporting to {format}…"):
            exported = model.export(
                format=format,
                imgsz=imgsz,
                device=device,
                half=half,
                simplify=simplify if format == "onnx" else False,
                dynamic=dynamic,
            )

        export_path = Path(exported) if exported else model_path.with_suffix(f".{format}")
        if output:
            import shutil
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(export_path, output)
            export_path = output

        console.print(f"\n[bold green]Export complete:[/bold green] {export_path}")

    except ImportError:
        console.print("[red]ultralytics not installed.[/red] Run: pip install ultralytics")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Export failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("list")
def list_runs(
    runs_dir: Path = typer.Option(Path("./trichome_runs"), "--dir", help="Runs directory"),
    limit: int = typer.Option(20, "--limit", help="Max runs to show"),
) -> None:
    """List recent training runs with metrics."""
    if not runs_dir.exists():
        console.print(f"[yellow]No runs found at {runs_dir}[/yellow]")
        return

    runs = sorted(runs_dir.rglob("results.json"), reverse=True)[:limit]
    if not runs:
        console.print(f"[yellow]No results.json files found in {runs_dir}[/yellow]")
        return

    table = Table(title=f"Training Runs (last {limit})")
    table.add_column("Run", style="cyan")
    table.add_column("mAP50", justify="right")
    table.add_column("mAP50-95", justify="right")
    table.add_column("Epochs")
    table.add_column("Status")

    for result_path in runs:
        try:
            data = json.loads(result_path.read_text())
            run_name = result_path.parent.name
            table.add_row(
                run_name,
                f"{data.get('best_map50', 0):.4f}",
                f"{data.get('best_map50_95', 0):.4f}",
                str(data.get("best_epoch", "?")),
                "[green]done[/green]",
            )
        except Exception:
            table.add_row(result_path.parent.name, "—", "—", "—", "[yellow]error[/yellow]")

    console.print(table)


def _merge_yaml_config(cfg: object, config_path: Path) -> None:
    """Merge YAML config file values into TrainingConfig."""
    try:
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        if not data:
            return
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    except Exception as e:
        console.print(f"[yellow]Config merge warning: {e}[/yellow]")


def _print_training_result(result: object) -> None:
    table = Table(title="Training Results", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Best mAP50", f"{getattr(result, 'best_map50', 0):.4f}")
    table.add_row("Best mAP50-95", f"{getattr(result, 'best_map50_95', 0):.4f}")
    table.add_row("Best epoch", str(getattr(result, 'best_epoch', '?')))
    model_path = getattr(result, 'best_model_path', None)
    if model_path:
        table.add_row("Best model", str(model_path))
    console.print(table)


if __name__ == "__main__":
    app()
