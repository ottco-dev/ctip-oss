"""
apps.cli.commands.video — Video processing CLI command.

Extracts and ranks the best-quality frames from trichome microscopy videos.

Pipeline:
  1. Sample frames at target FPS
  2. Score each frame: composite(focus=55%, exposure=25%, noise=20%)
  3. Deduplication: perceptual hash (pHash, Zauner 2010)
  4. Ranking: top-N, diverse-N, or adaptive coverage
  5. Extract + save selected frames as PNG/JPEG

Sub-commands:
  extract   — Extract best frames from a video
  info      — Show video file metadata
  score     — Score a single frame (focus, exposure, noise)
  analyze   — Full analysis: extract + score + rank + report

Usage:
    trichome video extract microscopy.mp4 --top-n 100
    trichome video info video.mov
    trichome video analyze video.mp4 --output /data/frames/ --top-n 50
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import track, Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

console = Console()

app = typer.Typer(
    help="Extract and rank best frames from trichome microscopy video.",
    add_help_option=True,
)


@app.command("extract")
def extract_frames(
    video_path: Path = typer.Argument(..., help="Input video file"),
    output_dir: Path = typer.Option(Path("./output/frames"), "--output", "-o"),
    top_n: int = typer.Option(50, "--top-n", "-n", help="Number of best frames to extract"),
    fps: float = typer.Option(2.0, "--fps", help="Sampling rate (frames/second to sample)"),
    strategy: str = typer.Option("adaptive", "--strategy", "-s", help="Ranking strategy: top | diverse | adaptive"),
    min_quality: float = typer.Option(0.3, "--min-quality", help="Minimum composite quality score (0–1)"),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Perceptual hash deduplication"),
    dedup_threshold: int = typer.Option(8, "--dedup-threshold", help="Hamming distance for near-duplicate detection"),
    max_dimension: int = typer.Option(1920, "--max-dim", help="Resize frames if larger (0=no resize)"),
    format: str = typer.Option("png", "--format", help="Output format: png | jpg"),
    quality: int = typer.Option(95, "--quality", help="JPEG quality (jpg only)"),
) -> None:
    """
    Extract the best-quality frames from a microscopy video.

    Frames are scored on focus (Laplacian variance), exposure (histogram),
    and noise (Immerkaer estimator). Near-duplicates are removed using
    perceptual hashing (DCT-based pHash).

    \b
    Ranking strategies:
        top      — Best N frames by composite score
        diverse  — N frames uniformly distributed across timeline
        adaptive — Greedy max-coverage (quality + diversity)

    \b
    Examples:
        trichome video extract microscopy.mp4 --top-n 100
        trichome video extract video.mov --top-n 50 --strategy diverse
        trichome video extract video.mp4 --fps 1.0 --no-dedup --format jpg
    """
    if not video_path.exists():
        console.print(f"[red]Error:[/red] Video not found: {video_path}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Video Frame Extraction[/bold cyan]")
    console.print(f"  Input:     {video_path}")
    console.print(f"  Output:    {output_dir}")
    console.print(f"  Top-N:     {top_n}")
    console.print(f"  Sample:    {fps} fps")
    console.print(f"  Strategy:  {strategy}")
    console.print(f"  Min quality: {min_quality:.0%}")
    console.print(f"  Dedup:     {dedup} (threshold={dedup_threshold})")
    console.print()

    try:
        from video_pipeline.domain.extractor import get_video_info, extract_frames_fixed_rate
        from video_pipeline.domain.scorer import score_frame, FrameQualityScore
        from video_pipeline.domain.hasher import perceptual_hash, deduplicate_frames
        from video_pipeline.domain.ranker import RankedFrame, rank_top_n, rank_diverse_n, rank_adaptive
        import numpy as np
        import cv2

        # Get video info
        info = get_video_info(str(video_path))
        console.print(f"  Duration:  {info.duration_s:.1f}s  ({info.total_frames} frames @ {info.fps:.1f} fps)")
        console.print(f"  Size:      {info.width}×{info.height}")
        console.print()

        # Compute every_n from fps
        every_n = max(1, int(info.fps / fps))

        # Sample + score frames
        frames_data: list[tuple[np.ndarray, object]] = []
        frame_hashes: list[int] = []

        console.print("[dim]Sampling and scoring frames…[/dim]")
        max_dim = max_dimension if max_dimension > 0 else None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scoring…", total=None)

            for frame_rgb, frame_info in extract_frames_fixed_rate(
                str(video_path),
                every_n_frames=every_n,
                max_dimension=max_dim,
            ):
                score = score_frame(frame_rgb)
                phash = perceptual_hash(frame_rgb) if dedup else 0
                frame_hashes.append(phash)
                frames_data.append((frame_rgb, RankedFrame(
                    frame_index=frame_info.frame_index,
                    timestamp_s=frame_info.timestamp_s,
                    quality=score,
                    phash=phash,
                )))
                progress.update(task, description=f"Frame {frame_info.frame_index}: q={score.composite:.2f}")

        console.print(f"  Sampled:   {len(frames_data)} frames")

        # Deduplication
        if dedup and len(frames_data) > 1:
            keep_indices = deduplicate_frames(frame_hashes, threshold=dedup_threshold)
            frames_data = [frames_data[i] for i in keep_indices]
            console.print(f"  After dedup: {len(frames_data)} frames")

        # Filter by quality
        before = len(frames_data)
        frames_data = [(f, r) for f, r in frames_data if r.quality.composite >= min_quality]
        if len(frames_data) < before:
            console.print(f"  After quality filter: {len(frames_data)} frames")

        if not frames_data:
            console.print("[yellow]No frames passed quality filter. Lower --min-quality.[/yellow]")
            raise typer.Exit(code=0)

        # Rank
        ranked_frames = [r for _, r in frames_data]
        if strategy == "top":
            selected = rank_top_n(ranked_frames, top_n)
        elif strategy == "diverse":
            selected = rank_diverse_n(ranked_frames, top_n)
        else:
            selected = rank_adaptive(ranked_frames, top_n)

        selected_indices = {r.frame_index for r in selected}

        # Save selected frames
        saved_count = 0
        metadata: list[dict] = []

        console.print(f"\n[dim]Saving {len(selected)} selected frames…[/dim]")
        for frame_rgb, ranked in frames_data:
            if ranked.frame_index not in selected_indices:
                continue
            fname = f"frame_{ranked.frame_index:06d}_q{ranked.quality.composite:.2f}.{format}"
            fpath = output_dir / fname
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if format == "jpg":
                cv2.imwrite(str(fpath), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(fpath), frame_bgr)
            saved_count += 1
            metadata.append({
                "filename": fname,
                "frame_index": ranked.frame_index,
                "timestamp_s": round(ranked.timestamp_s, 3),
                "composite": round(ranked.quality.composite, 4),
                "focus": round(ranked.quality.focus, 4),
                "exposure": round(ranked.quality.exposure, 4),
                "noise": round(ranked.quality.noise, 4),
            })

        # Save metadata
        meta_path = output_dir / "frames_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "video": str(video_path),
                "strategy": strategy,
                "total_sampled": len(frames_data),
                "selected": saved_count,
                "frames": metadata,
            }, f, indent=2)

        # Summary
        if metadata:
            avg_q = sum(m["composite"] for m in metadata) / len(metadata)
            console.print(f"\n[bold green]Extraction complete[/bold green]")
            console.print(f"  Saved:     {saved_count} frames")
            console.print(f"  Avg quality: {avg_q:.3f}")
            console.print(f"  Output:    {output_dir}")
            console.print(f"  Metadata:  {meta_path}")

    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Extraction failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("info")
def video_info(
    video_path: Path = typer.Argument(..., help="Video file to inspect"),
) -> None:
    """Show video file metadata: duration, resolution, fps, codec."""
    if not video_path.exists():
        console.print(f"[red]Error:[/red] Video not found: {video_path}")
        raise typer.Exit(code=1)

    try:
        from video_pipeline.domain.extractor import get_video_info

        info = get_video_info(str(video_path))

        table = Table(title=f"Video Info: {video_path.name}", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        table.add_row("Path", str(video_path))
        table.add_row("Duration", f"{info.duration_s:.2f}s  ({info.duration_s / 60:.1f} min)")
        table.add_row("Resolution", f"{info.width}×{info.height}" + (" [4K]" if info.is_4k else " [HD]" if info.is_hd else ""))
        table.add_row("FPS", f"{info.fps:.3f}")
        table.add_row("Total frames", f"{info.total_frames:,}")
        table.add_row("Codec", info.codec)
        table.add_row("File size", f"{video_path.stat().st_size / 1e6:.1f} MB")
        console.print(table)

        # Quick sampling preview
        sample_n = min(5, info.total_frames)
        console.print(f"\n[dim]Sample {sample_n} quality scores…[/dim]")

        from video_pipeline.domain.extractor import extract_frames_fixed_rate
        from video_pipeline.domain.scorer import score_frame

        every_n = max(1, info.total_frames // sample_n)
        scores = []
        for frame_rgb, finfo in extract_frames_fixed_rate(str(video_path), every_n_frames=every_n):
            scores.append(score_frame(frame_rgb).composite)
            if len(scores) >= sample_n:
                break

        if scores:
            avg = sum(scores) / len(scores)
            mn, mx = min(scores), max(scores)
            console.print(f"  Quality range: [{mn:.2f} – {mx:.2f}]  avg={avg:.2f}")
            if avg < 0.4:
                console.print("[yellow]  ⚠ Low average quality — video may be out of focus[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("score")
def score_image(
    image_path: Path = typer.Argument(..., help="Image to score"),
    show_breakdown: bool = typer.Option(True, "--breakdown/--no-breakdown"),
) -> None:
    """Score a single image on focus, exposure, and noise quality."""
    if not image_path.exists():
        console.print(f"[red]Error:[/red] Image not found: {image_path}")
        raise typer.Exit(code=1)

    try:
        import cv2
        import numpy as np
        from video_pipeline.domain.scorer import score_frame

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            console.print(f"[red]Cannot read image:[/red] {image_path}")
            raise typer.Exit(code=1)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        score = score_frame(rgb)

        table = Table(title=f"Quality Score: {image_path.name}", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Label")

        def _bar(v: float) -> str:
            n = int(v * 20)
            return "█" * n + "░" * (20 - n)

        table.add_row("Composite", f"{score.composite:.3f}", f"{_bar(score.composite)}  {score.quality_label}")
        if show_breakdown:
            table.add_row("Focus (55%)", f"{score.focus:.3f}", _bar(score.focus))
            table.add_row("Exposure (25%)", f"{score.exposure:.3f}", _bar(score.exposure))
            table.add_row("Noise (20%)", f"{score.noise:.3f}", _bar(score.noise))

        console.print(table)

        if not score.is_usable:
            console.print("[red]  ✗ Below usable threshold (0.2)[/red]")
        elif score.is_excellent:
            console.print("[green]  ✓ Excellent quality[/green]")
        else:
            console.print("[yellow]  ~ Acceptable quality[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
