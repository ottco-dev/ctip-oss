"""
apps.cli.commands.convert — YOLO model conversion CLI command.

Provides three subcommands for the full ONNX/TensorRT conversion pipeline:

  trichome convert onnx      model.pt --output-dir ./models/
  trichome convert tensorrt  model.pt --output-dir ./models/ --fp16
  trichome convert validate  model.onnx

Usage:
    trichome convert onnx      best.pt --output-dir /models/
    trichome convert onnx      best.pt --imgsz 640 --opset 17
    trichome convert tensorrt  best.pt --output-dir /models/ --fp16 --workspace-gb 4
    trichome convert validate  best.onnx
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

app = typer.Typer(
    name="convert",
    help="Convert YOLO model weights to ONNX and/or TensorRT formats.",
    add_help_option=True,
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_file(path: Path, label: str = "File") -> None:
    """Exit with an error message if *path* does not exist."""
    if not path.exists():
        console.print(f"[red]{label} not found:[/red] {path}")
        raise typer.Exit(code=1)


def _print_result_table(result: dict, title: str = "Export Result") -> None:
    """Render the export result dict as a Rich table."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for k, v in result.items():
        val = str(v) if v is not None else "[dim]n/a[/dim]"
        table.add_row(k.replace("_", " ").title(), val)
    console.print(table)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("onnx")
def export_onnx(
    model: Path = typer.Argument(
        ..., help="Path to YOLO .pt weights file.", metavar="MODEL_PT"
    ),
    output_dir: Path = typer.Option(
        Path("./models/onnx"),
        "--output-dir", "-o",
        help="Directory to write the .onnx file.",
    ),
    imgsz: int = typer.Option(
        1280, "--imgsz", help="Square input resolution for export (pixels)."
    ),
    opset: int = typer.Option(
        17, "--opset", help="ONNX opset version (17 recommended for TRT 10)."
    ),
    fp16: bool = typer.Option(
        True, "--fp16/--no-fp16", help="Export with FP16 weights."
    ),
    simplify: bool = typer.Option(
        True, "--simplify/--no-simplify", help="Run onnx-simplifier after export."
    ),
    dynamic: bool = typer.Option(
        False, "--dynamic/--no-dynamic",
        help="Enable dynamic batch axis (not recommended for RTX 4060)."
    ),
) -> None:
    """
    Export YOLO .pt weights to ONNX format.

    \b
    Examples:
        trichome convert onnx best.pt
        trichome convert onnx best.pt --output-dir /models/ --imgsz 640
        trichome convert onnx best.pt --no-simplify --no-fp16
    """
    _require_file(model, "Model weights")

    from inference.tensorrt_engine.exporter import YOLOExportConfig, YOLOToTensorRT, ExportError

    try:
        cfg = YOLOExportConfig(
            model_path=str(model),
            output_dir=str(output_dir),
            imgsz=imgsz,
            fp16=fp16,
            opset=opset,
            simplify=simplify,
            dynamic_batch=dynamic,
        )
    except ValueError as exc:
        console.print(f"[red]Invalid configuration:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]YOLO → ONNX Export[/bold cyan]")
    console.print(f"  Model:      {model}")
    console.print(f"  Output dir: {output_dir}")
    console.print(f"  imgsz:      {imgsz}  opset={opset}  fp16={fp16}  simplify={simplify}")
    console.print()

    exporter = YOLOToTensorRT(cfg)
    t0 = time.perf_counter()
    try:
        onnx_path = exporter.export_onnx_only()
    except FileNotFoundError as exc:
        console.print(f"[red]File not found:[/red] {exc}")
        raise typer.Exit(code=1)
    except ExportError as exc:
        console.print(f"[red]Export error:[/red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {exc}")
        raise typer.Exit(code=1)

    elapsed = round(time.perf_counter() - t0, 2)
    onnx_file = Path(onnx_path)
    size_mb = onnx_file.stat().st_size / (1024 ** 2) if onnx_file.exists() else 0.0

    _print_result_table({
        "onnx_path":    onnx_path,
        "size_mb":      f"{size_mb:.1f} MB",
        "export_time":  f"{elapsed} s",
    }, title="ONNX Export Result")

    console.print(f"\n[bold green]ONNX export complete:[/bold green] {onnx_path}")


@app.command("tensorrt")
def export_tensorrt(
    model: Path = typer.Argument(
        ..., help="Path to YOLO .pt weights file.", metavar="MODEL_PT"
    ),
    output_dir: Path = typer.Option(
        Path("./models/tensorrt"),
        "--output-dir", "-o",
        help="Directory to write the .onnx and .engine files.",
    ),
    imgsz: int = typer.Option(
        1280, "--imgsz", help="Square input resolution for export (pixels)."
    ),
    fp16: bool = typer.Option(
        True, "--fp16/--no-fp16", help="Enable FP16 precision in TRT engine."
    ),
    workspace_gb: float = typer.Option(
        4.0, "--workspace-gb", help="TensorRT builder GPU workspace budget (GB)."
    ),
    opset: int = typer.Option(
        17, "--opset", help="ONNX opset version."
    ),
    simplify: bool = typer.Option(
        True, "--simplify/--no-simplify", help="Run onnx-simplifier before TRT build."
    ),
) -> None:
    """
    Full pipeline: YOLO .pt → ONNX → TensorRT .engine

    Requires ultralytics for the YOLO export step and TensorRT for engine
    building. If TensorRT is not installed, the pipeline stops after ONNX
    export and prints a warning.

    \b
    Examples:
        trichome convert tensorrt best.pt
        trichome convert tensorrt best.pt --output-dir /models/ --no-fp16
        trichome convert tensorrt best.pt --workspace-gb 6
    """
    _require_file(model, "Model weights")

    from inference.tensorrt_engine.exporter import YOLOExportConfig, YOLOToTensorRT, ExportError

    try:
        cfg = YOLOExportConfig(
            model_path=str(model),
            output_dir=str(output_dir),
            imgsz=imgsz,
            fp16=fp16,
            opset=opset,
            simplify=simplify,
            workspace_gb=workspace_gb,
        )
    except ValueError as exc:
        console.print(f"[red]Invalid configuration:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]YOLO → ONNX → TensorRT Pipeline[/bold cyan]")
    console.print(f"  Model:        {model}")
    console.print(f"  Output dir:   {output_dir}")
    console.print(f"  imgsz:        {imgsz}")
    console.print(f"  FP16:         {fp16}")
    console.print(f"  Workspace:    {workspace_gb} GB")
    console.print(f"  ONNX opset:   {opset}")
    console.print()

    exporter = YOLOToTensorRT(cfg)
    try:
        result = exporter.export()
    except FileNotFoundError as exc:
        console.print(f"[red]File not found:[/red] {exc}")
        raise typer.Exit(code=1)
    except ExportError as exc:
        console.print(f"[red]Export error:[/red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Pipeline error:[/red] {exc}")
        raise typer.Exit(code=1)

    _print_result_table(result, title="Export Pipeline Result")

    if result.get("engine_path") is None:
        console.print(
            "\n[yellow]Warning:[/yellow] TensorRT engine was not built "
            "(TensorRT not available). ONNX file is ready for ONNX Runtime inference."
        )
    else:
        console.print(f"\n[bold green]Pipeline complete.[/bold green] Engine: {result['engine_path']}")


@app.command("validate")
def validate_onnx(
    onnx_path: Path = typer.Argument(
        ..., help="Path to the ONNX model file to validate.", metavar="ONNX_FILE"
    ),
) -> None:
    """
    Validate an ONNX model with a random test inference via ONNXRuntime.

    Runs a dummy forward pass and reports output shapes. Useful for confirming
    that the exported model loads correctly before attempting a TensorRT build.

    \b
    Examples:
        trichome convert validate best.onnx
        trichome convert validate /models/yolo11s_trichome.onnx
    """
    _require_file(onnx_path, "ONNX file")

    from inference.tensorrt_engine.exporter import YOLOExportConfig, YOLOToTensorRT

    # We only need the validate_onnx() method — model_path / output_dir
    # are required by config but irrelevant here; use dummy non-empty values.
    cfg = YOLOExportConfig(
        model_path=str(onnx_path),   # ignored by validate_onnx
        output_dir=str(onnx_path.parent),
    )
    exporter = YOLOToTensorRT(cfg)

    console.print(f"\n[bold cyan]ONNX Validation[/bold cyan]")
    console.print(f"  File: {onnx_path}")
    console.print()

    result = exporter.validate_onnx(str(onnx_path))

    if result["ok"]:
        table = Table(title="Validation Result", show_header=True, header_style="bold green")
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        table.add_row("Status",       "[bold green]PASS[/bold green]")
        table.add_row("Input shape",  str(result["input_shape"]))
        table.add_row("Output count", str(len(result["output_shapes"])))
        for i, shape in enumerate(result["output_shapes"]):
            table.add_row(f"Output {i} shape", str(shape))
        console.print(table)
        console.print("\n[bold green]ONNX model is valid.[/bold green]")
    else:
        console.print(Panel(
            f"[red]{result['error']}[/red]",
            title="[bold red]Validation FAILED[/bold red]",
        ))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
