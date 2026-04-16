"""CLI entry point using Typer."""

from __future__ import annotations

import os
import sys

# Force UTF-8 mode on Windows to avoid cp1252 encoding errors with Rich
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from .config import Config

app = typer.Typer(
    name="slide-extractor",
    help="Lecture slide extraction pipeline using Meta SAM 3.1",
)
console = Console(force_terminal=True, legacy_windows=False)


@app.command()
def process(
    video_path: Annotated[
        Path, typer.Argument(help="Path to a video file or directory of videos")
    ],
    fps: Annotated[float, typer.Option(help="Frame sampling rate in fps")] = 1.0,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory")
    ] = None,
    confidence: Annotated[
        Optional[float], typer.Option(help="Detection confidence threshold")
    ] = None,
    ssim_threshold: Annotated[
        Optional[float], typer.Option(help="SSIM change detection threshold")
    ] = None,
    debug: Annotated[
        bool, typer.Option(help="Save all intermediate outputs")
    ] = False,
    phase: Annotated[
        Optional[str],
        typer.Option(help="Run single phase: detect, geometry, track, output"),
    ] = None,
    device: Annotated[
        Optional[str], typer.Option(help="Device: cuda or cpu")
    ] = None,
) -> None:
    """Process video(s) to extract lecture slides."""
    from .pipeline import run_pipeline

    cfg = Config.from_env()

    # CLI overrides
    cfg.fps = fps
    if output_dir:
        cfg.output_dir = output_dir
    if confidence is not None:
        cfg.detection_confidence = confidence
    if ssim_threshold is not None:
        cfg.ssim_threshold = ssim_threshold
    if debug:
        cfg.save_debug_frames = True
    if device:
        cfg.device = device

    # Collect video files
    videos: list[Path] = []
    if video_path.is_dir():
        for ext in ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm"):
            videos.extend(video_path.glob(ext))
        videos.sort()
    elif video_path.is_file():
        videos = [video_path]
    else:
        console.print(f"[red]Error: {video_path} not found[/red]")
        raise typer.Exit(1)

    if not videos:
        console.print("[red]No video files found[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Processing {len(videos)} video(s) at {fps} fps[/bold]\n")

    for i, vp in enumerate(videos, 1):
        console.print(
            f"[bold yellow]--- Video {i}/{len(videos)}: {vp.name} ---[/bold yellow]"
        )
        run_pipeline(vp, cfg, phase_filter=phase)


@app.command()
def download_models() -> None:
    """Download SAM 3.1 model checkpoints."""
    from .models import download_sam3_weights

    cfg = Config.from_env()
    console.print("[bold]Downloading SAM 3.1 checkpoints...[/bold]")
    download_sam3_weights(cfg)
    console.print("[green]Done![/green]")


@app.command()
def info(
    video_path: Annotated[Path, typer.Argument(help="Path to video file")],
) -> None:
    """Show video metadata (duration, fps, resolution)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        console.print(f"[red]Cannot open {video_path}[/red]")
        raise typer.Exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    from rich.table import Table

    table = Table(title=f"Video Info: {video_path.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_row("Resolution", f"{width}x{height}")
    table.add_row("FPS", f"{fps:.2f}")
    table.add_row("Frames", str(frame_count))
    table.add_row("Duration", f"{duration:.1f}s ({duration / 60:.1f}m)")
    table.add_row("File size", f"{video_path.stat().st_size / 1024 / 1024:.1f} MB")
    console.print(table)


if __name__ == "__main__":
    app()
