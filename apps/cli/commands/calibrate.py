"""
apps.cli.commands.calibrate — Microscope pixel-to-µm calibration CLI.

Two calibration workflows:
  1. Interactive measurement: enter pixel distances measured from a stage
     micrometer image using ImageJ/Fiji, then compute µm/pixel scale.
  2. Auto-detect: Hough-line based automatic scale bar detection from image.

Outputs a MicroscopeProfile JSON saved to the specified path, importable
by the measurement pipeline.

Usage:
    trichome calibrate stage_micrometer.tif --spacing 10.0 --objective 40
    trichome calibrate image.tif --auto --spacing 10.0 --name "40x_microscope_A"
    trichome calibrate --list          List saved profiles
    trichome calibrate --show profile  Show profile details
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(
    help="Microscope calibration: pixel-to-µm scale from stage micrometer.",
    add_help_option=True,
)

_DEFAULT_PROFILE_DIR = Path.home() / ".trichome" / "profiles"


@app.command("run")
def run_calibration(
    image: Path = typer.Argument(..., help="Stage micrometer image path"),
    spacing_um: float = typer.Option(10.0, "--spacing", help="Known line spacing in µm"),
    objective: float = typer.Option(40.0, "--objective", help="Objective magnification (10, 20, 40, 100)"),
    auto: bool = typer.Option(False, "--auto", help="Auto-detect scale bar via Hough-line detection"),
    name: str = typer.Option("", "--name", help="Profile name (default: auto-generated)"),
    output: Path = typer.Option(Path("./calibration_profile.json"), "--output", "-o", help="Output profile JSON path"),
    save_to_library: bool = typer.Option(True, "--library/--no-library", help="Also save to ~/.trichome/profiles/"),
) -> None:
    """
    Calibrate microscope pixel-to-µm scale from a stage micrometer image.

    Interactive mode: enter pixel distances measured in ImageJ/Fiji.
    Auto mode:       Hough-line detection finds scale bar automatically.

    \b
    Examples:
        trichome calibrate run stage_micrometer.tif --spacing 10.0 --objective 40
        trichome calibrate run image.tif --auto --spacing 10 --name "lab_scope_A"
    """
    if not image.exists():
        console.print(f"[red]Error:[/red] Image not found: {image}")
        raise typer.Exit(code=1)

    console.print("\n[bold cyan]Microscope Calibration[/bold cyan]")
    console.print(f"  Image:       {image}")
    console.print(f"  Spacing:     {spacing_um} µm/division")
    console.print(f"  Objective:   {objective}×")
    console.print(f"  Mode:        {'Automatic (Hough-line)' if auto else 'Interactive'}")
    console.print()

    try:
        from measurement.calibration.stage_micrometer import StageMicrometerCalibrator

        calibrator = StageMicrometerCalibrator(known_spacing_um=spacing_um)

        if auto:
            # Auto-detect scale bar
            import cv2
            import numpy as np

            console.print("[dim]Running Hough-line scale bar detection…[/dim]")
            gray = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2GRAY)

            # Use built-in auto-detect if available
            auto_measurements = _auto_detect_scale_bar(gray, spacing_um=spacing_um)
            if not auto_measurements:
                console.print("[yellow]Auto-detection found no scale bars. Falling back to interactive mode.[/yellow]")
                auto = False
            else:
                for px_dist in auto_measurements:
                    calibrator.add_measurement(px_dist)
                    console.print(f"  [dim]Auto-detected: {px_dist:.1f} px[/dim]")

        if not auto:
            # Interactive mode
            console.print(
                Panel(
                    "Open the image in [bold]ImageJ/Fiji[/bold] and measure pixel distances "
                    "between adjacent division lines on the stage micrometer.\n\n"
                    "Enter each measurement as a pixel count, then type [bold]done[/bold] when finished.\n"
                    "[yellow]Minimum 3 measurements recommended for reliable calibration.[/yellow]",
                    title="Instructions",
                    border_style="blue",
                )
            )

            while True:
                try:
                    raw = typer.prompt("\nPixel distance (or 'done')")
                    if raw.strip().lower() in ("done", "d", "q", "exit"):
                        break
                    px = float(raw)
                    if px <= 0:
                        console.print("[yellow]Value must be positive.[/yellow]")
                        continue
                    calibrator.add_measurement(px)
                    n = len(calibrator._measurements)
                    console.print(f"  [green]✓[/green] Measurement #{n}: {px:.1f} px")
                except ValueError:
                    console.print("[yellow]Enter a number or 'done'[/yellow]")

        if not calibrator._measurements:
            console.print("[red]No measurements recorded. Calibration cancelled.[/red]")
            raise typer.Exit(code=1)

        n_meas = len(calibrator._measurements)
        if n_meas < 3:
            console.print(
                f"[yellow]Warning: Only {n_meas} measurements. "
                "3+ recommended for reliable uncertainty estimation.[/yellow]"
            )

        # Compute calibration
        profile_name = name or f"{int(objective)}x_{image.stem}"
        profile = calibrator.compute_calibration(
            profile_name=profile_name,
            objective_magnification=objective,
        )

        # Display results
        console.print()
        _print_calibration_result(profile)

        if profile.measurement_cv and profile.measurement_cv > 5.0:
            console.print(
                "[yellow]  ⚠ CV > 5% — measurement variability is high. "
                "Consider recalibrating with more measurements.[/yellow]"
            )

        # Save profile
        output.parent.mkdir(parents=True, exist_ok=True)
        profile.save(output)
        console.print(f"\n  [green]Saved:[/green] {output}")

        if save_to_library:
            _DEFAULT_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            lib_path = _DEFAULT_PROFILE_DIR / f"{profile_name}.json"
            profile.save(lib_path)
            console.print(f"  [green]Library:[/green] {lib_path}")

    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Calibration failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("list")
def list_profiles(
    profile_dir: Path = typer.Option(_DEFAULT_PROFILE_DIR, "--dir", help="Profile library directory"),
) -> None:
    """List all saved microscope calibration profiles."""
    if not profile_dir.exists():
        console.print(f"[yellow]No profiles found at {profile_dir}[/yellow]")
        console.print("[dim]Run 'trichome calibrate run' to create a profile.[/dim]")
        return

    profiles = sorted(profile_dir.glob("*.json"))
    if not profiles:
        console.print(f"[yellow]No profiles in {profile_dir}[/yellow]")
        return

    table = Table(title=f"Microscope Profiles ({len(profiles)})", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("µm/pixel", justify="right")
    table.add_column("Objective")
    table.add_column("Uncertainty", justify="right")
    table.add_column("Method")
    table.add_column("Date")

    for p in profiles:
        try:
            data = json.loads(p.read_text())
            table.add_row(
                data.get("profile_name", p.stem),
                f"{data.get('um_per_pixel', 0):.4f}",
                f"{data.get('objective_magnification', '?')}×",
                f"±{data.get('relative_uncertainty_percent', 0):.1f}%",
                data.get("calibration_method", "unknown"),
                data.get("calibration_date", "")[:10],
            )
        except Exception:
            table.add_row(p.stem, "—", "—", "—", "error", "")

    console.print(table)


@app.command("show")
def show_profile(
    name: str = typer.Argument(..., help="Profile name or path"),
    profile_dir: Path = typer.Option(_DEFAULT_PROFILE_DIR, "--dir"),
) -> None:
    """Show detailed information for a calibration profile."""
    # Try as path first, then as library name
    profile_path = Path(name) if Path(name).exists() else profile_dir / f"{name}.json"
    if not profile_path.exists():
        console.print(f"[red]Profile not found:[/red] {name}")
        raise typer.Exit(code=1)

    try:
        from measurement.calibration.stage_micrometer import MicroscopeProfile
        profile = MicroscopeProfile.load(profile_path)
        _print_calibration_result(profile)
    except Exception as e:
        console.print(f"[red]Error loading profile:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("estimate")
def estimate_scale(
    objective: float = typer.Argument(..., help="Objective magnification (10, 20, 40, 100)"),
    sensor_pixel_size: float = typer.Option(3.45, "--pixel-size", help="Camera sensor pixel size in µm"),
    camera_adapter: float = typer.Option(0.5, "--adapter", help="C-mount adapter magnification"),
) -> None:
    """
    Estimate µm/pixel from objective magnification and camera sensor specs.

    Use this when you don't have a stage micrometer available.
    NOTE: This is an approximation — always prefer physical calibration.

    \b
    Formula:
        µm/pixel = sensor_pixel_size / (objective × camera_adapter)
    """
    try:
        from measurement.calibration.stage_micrometer import estimate_scale_from_objective
        um_per_px = estimate_scale_from_objective(
            objective_magnification=objective,
            sensor_pixel_size_um=sensor_pixel_size,
            adapter_magnification=camera_adapter,
        )
        console.print(f"\n[bold]Estimated Scale[/bold]")
        console.print(f"  Objective:    {objective}×")
        console.print(f"  Sensor pixel: {sensor_pixel_size} µm")
        console.print(f"  Adapter:      {camera_adapter}×")
        console.print(f"  µm/pixel:     [bold green]{um_per_px:.4f}[/bold green]")
        console.print()
        console.print("[yellow]⚠ This is a theoretical estimate. Physical calibration is preferred.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


def _print_calibration_result(profile: object) -> None:
    """Print calibration result table."""
    table = Table(title="Calibration Result", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    um_per_px = getattr(profile, "um_per_pixel", None) or getattr(profile, "um_per_px", None)
    table.add_row("µm/pixel", f"[bold green]{um_per_px:.4f}[/bold green]")
    table.add_row("Objective", f"{getattr(profile, 'objective_magnification', '?')}×")

    unc = getattr(profile, "relative_uncertainty_percent", None)
    if unc is not None:
        table.add_row("Relative uncertainty", f"±{unc:.1f}%")

    cv = getattr(profile, "measurement_cv", None)
    if cv is not None:
        table.add_row("CV (repeatability)", f"{cv:.1f}%")

    n = getattr(profile, "n_measurements", None)
    if n is not None:
        table.add_row("N measurements", str(n))

    table.add_row("Method", getattr(profile, "calibration_method", "stage_micrometer"))
    table.add_row("Name", getattr(profile, "profile_name", "unnamed"))
    console.print(table)


def _auto_detect_scale_bar(gray, spacing_um: float = 10.0) -> list[float]:
    """
    Attempt Hough-line scale bar detection.
    Returns list of detected pixel spacings.
    """
    try:
        import cv2
        import numpy as np

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Probabilistic Hough lines
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=80, minLineLength=30, maxLineGap=5,
        )
        if lines is None:
            return []

        # Collect horizontal line y-coordinates
        h_lines: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 5 or angle > 175:  # Near-horizontal
                h_lines.append((y1 + y2) / 2.0)

        if len(h_lines) < 2:
            return []

        h_lines_sorted = sorted(set(round(y) for y in h_lines))
        spacings = [abs(h_lines_sorted[i + 1] - h_lines_sorted[i]) for i in range(len(h_lines_sorted) - 1)]

        # Filter plausible spacings (10–2000 px)
        plausible = [s for s in spacings if 10 < s < 2000]
        return plausible[:10]  # Return top-10 candidates

    except Exception:
        return []


if __name__ == "__main__":
    app()
