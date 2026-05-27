"""
apps.cli.commands.export — Analysis session export CLI command.

Exports trichome analysis results in multiple formats:
  pdf    — Scientific report (detection summary, maturity distribution chart,
            morphology breakdown, measurement data, scientific caveats)
  csv    — Flat tabular export of all measurements and classifications
  json   — Full structured export with all metadata and provenance

Usage:
    trichome export session_abc123 --formats pdf,csv,json
    trichome export /path/to/session.json --formats pdf
    trichome export session_id --formats csv --output /data/reports/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

app = typer.Typer(
    help="Export analysis results to PDF, CSV, and/or JSON reports.",
    add_help_option=True,
)

_VALID_FORMATS = {"pdf", "csv", "json"}


def _parse_formats(formats_str: str) -> list[str]:
    fmts = [f.strip().lower() for f in formats_str.split(",")]
    invalid = [f for f in fmts if f not in _VALID_FORMATS]
    if invalid:
        console.print(f"[red]Unknown formats:[/red] {', '.join(invalid)}")
        console.print(f"[dim]Valid formats: {', '.join(sorted(_VALID_FORMATS))}[/dim]")
        raise typer.Exit(code=1)
    return fmts


@app.command("run")
def run_export(
    session: str = typer.Argument(..., help="Session ID or path to session result JSON"),
    output_dir: Path = typer.Option(Path("./reports"), "--output", "-o", help="Output directory"),
    formats: str = typer.Option("pdf,csv,json", "--formats", "-f", help="Comma-separated: pdf | csv | json"),
    title: str = typer.Option("Trichome Analysis Report", "--title", help="Report title"),
    author: str = typer.Option("", "--author", help="Author name for PDF report"),
    include_crops: bool = typer.Option(False, "--crops", help="Include detection crop images in PDF"),
    include_heatmaps: bool = typer.Option(False, "--heatmaps", help="Include density heatmaps in PDF"),
    language: str = typer.Option("en", "--lang", help="Report language: en | de (PDF only)"),
) -> None:
    """
    Export analysis session results to scientific reports.

    The PDF report includes:
      - Trichome detection summary (count, confidence distribution)
      - Maturity stage distribution (pie chart + table)
      - Morphology type breakdown
      - Measurement statistics (head diameter, stalk length, area)
      - Scientific caveats and methodology disclosure

    \b
    Examples:
        trichome export run session_abc123
        trichome export run /tmp/results/session.json --formats pdf,csv
        trichome export run session_id --title "Strain XYZ — Week 8"
    """
    fmt_list = _parse_formats(formats)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load session data
    session_data = _load_session(session)
    session_id = session_data.get("session_id", Path(session).stem if Path(session).exists() else session)

    console.print(f"\n[bold cyan]Export Analysis Results[/bold cyan]")
    console.print(f"  Session:  {session_id}")
    console.print(f"  Formats:  {', '.join(fmt_list)}")
    console.print(f"  Output:   {output_dir}")
    console.print()

    exported: list[tuple[str, Path]] = []
    errors: list[str] = []

    for fmt in fmt_list:
        out_path = output_dir / f"{session_id}_report.{fmt}"
        try:
            if fmt == "json":
                _export_json(session_data, out_path)
            elif fmt == "csv":
                _export_csv(session_data, out_path)
            elif fmt == "pdf":
                _export_pdf(
                    session_data, out_path,
                    title=title, author=author,
                    include_crops=include_crops,
                    include_heatmaps=include_heatmaps,
                    language=language,
                )
            exported.append((fmt.upper(), out_path))
            console.print(f"  [green]✓[/green] {fmt.upper()}: {out_path}")
        except Exception as e:
            errors.append(f"{fmt}: {e}")
            console.print(f"  [red]✗[/red] {fmt.upper()}: {e}")

    console.print()
    if exported:
        console.print(f"[bold green]Export complete[/bold green]  ({len(exported)}/{len(fmt_list)} formats)")
    if errors:
        console.print(f"[red]Errors:[/red] {'; '.join(errors)}")
        raise typer.Exit(code=1)


@app.command("list")
def list_sessions(
    sessions_dir: Path = typer.Option(Path("./output"), "--dir", help="Sessions directory"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List available analysis sessions that can be exported."""
    sessions = sorted(
        sessions_dir.rglob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    if not sessions:
        console.print(f"[yellow]No sessions found in {sessions_dir}[/yellow]")
        return

    table = Table(title=f"Available Sessions (last {limit})")
    table.add_column("ID", style="cyan")
    table.add_column("Date")
    table.add_column("Detections", justify="right")
    table.add_column("Path", style="dim")

    for s in sessions:
        try:
            data = json.loads(s.read_text())
            from datetime import datetime
            mtime = datetime.fromtimestamp(s.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            table.add_row(
                data.get("session_id", s.stem),
                mtime,
                str(data.get("total_detections", "?")),
                str(s),
            )
        except Exception:
            table.add_row(s.stem, "?", "?", str(s))

    console.print(table)


@app.command("preview")
def preview_session(
    session: str = typer.Argument(..., help="Session ID or JSON path"),
) -> None:
    """Preview session data before exporting."""
    data = _load_session(session)
    if not data:
        console.print(f"[yellow]No data for session: {session}[/yellow]")
        return

    console.print(Panel(
        json.dumps({k: v for k, v in data.items() if k != "detections"}, indent=2),
        title=f"Session: {data.get('session_id', session)}",
    ))


# ── Internal export functions ─────────────────────────────────────────────────

def _load_session(session: str) -> dict:
    """Load session data from path or return skeleton."""
    p = Path(session)
    if p.exists() and p.suffix == ".json":
        try:
            return json.loads(p.read_text())
        except Exception as e:
            console.print(f"[yellow]Warning: could not parse {p}: {e}[/yellow]")

    # Return skeleton with session_id for downstream processing
    return {
        "session_id": session,
        "total_detections": 0,
        "maturity_distribution": {},
        "morphology_distribution": {},
        "measurements": [],
        "metadata": {},
    }


def _export_json(data: dict, output: Path) -> None:
    """Write full JSON export."""
    try:
        from analytics.export.json_exporter import JsonExporter
        exporter = JsonExporter()
        exporter.export(data, output)
    except ImportError:
        # Fallback: write raw JSON
        with open(output, "w") as f:
            json.dump(data, f, indent=2, default=str)


def _export_csv(data: dict, output: Path) -> None:
    """Write flat CSV export."""
    try:
        from analytics.export.csv_exporter import CsvExporter
        exporter = CsvExporter()
        exporter.export(data, output)
    except ImportError:
        # Fallback: write measurements CSV
        import csv
        measurements = data.get("measurements", [])
        if not measurements:
            # Create minimal CSV with summary
            with open(output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "total_detections"])
                writer.writerow([data.get("session_id", ""), data.get("total_detections", 0)])
        else:
            with open(output, "w", newline="") as f:
                if measurements:
                    writer = csv.DictWriter(f, fieldnames=measurements[0].keys())
                    writer.writeheader()
                    writer.writerows(measurements)


def _export_pdf(
    data: dict,
    output: Path,
    title: str = "Trichome Analysis Report",
    author: str = "",
    include_crops: bool = False,
    include_heatmaps: bool = False,
    language: str = "en",
) -> None:
    """Generate scientific PDF report."""
    try:
        from analytics.export.pdf_exporter import PdfExporter
        exporter = PdfExporter(
            title=title,
            author=author or "TrichomeLab",
            include_crops=include_crops,
            include_heatmaps=include_heatmaps,
            language=language,
        )
        exporter.export(data, output)
    except ImportError:
        # Fallback: generate minimal text-based "PDF" using reportlab if available
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas

            c = canvas.Canvas(str(output), pagesize=A4)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(72, 750, title)
            c.setFont("Helvetica", 12)
            c.drawString(72, 710, f"Session: {data.get('session_id', 'unknown')}")
            c.drawString(72, 690, f"Total detections: {data.get('total_detections', 0)}")
            c.setFont("Helvetica-Oblique", 9)
            c.drawString(72, 100, "SCIENTIFIC CAVEAT: Maturity classification reflects optical color state only.")
            c.save()
        except ImportError:
            raise ImportError("PDF generation requires 'analytics' module or 'reportlab' package")


if __name__ == "__main__":
    app()
