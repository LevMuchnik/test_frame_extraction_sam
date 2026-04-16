"""Pipeline orchestrator — runs all 4 phases in sequence."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console

from .config import Config
from .metrics import PipelineMetrics
from .phase1_detection import run_phase1
from .phase2_geometry import run_phase2
from .phase3_tracking import run_phase3
from .phase4_output import run_phase4

console = Console()

PHASES = {"detect", "geometry", "track", "output"}


def run_pipeline(
    video_path: Path,
    cfg: Config,
    phase_filter: str | None = None,
) -> None:
    """Run the full slide extraction pipeline on a single video."""
    if phase_filter and phase_filter not in PHASES:
        console.print(
            f"[red]Unknown phase '{phase_filter}'. "
            f"Choose from: {', '.join(sorted(PHASES))}[/red]"
        )
        return

    dirs = cfg.ensure_dirs(video_path)
    metrics = PipelineMetrics(video=video_path.name)
    metrics.start_time = time.time()

    detections = []
    geometry_records = []
    transitions = []

    # Phase 1: Detection
    if not phase_filter or phase_filter == "detect":
        with metrics.phase("1_detection") as pm:
            detections = run_phase1(video_path, cfg, dirs, pm)

    # If running a later phase standalone, load from JSON
    if phase_filter and phase_filter != "detect" and not detections:
        import json

        det_path = dirs["base"] / "detections.json"
        if det_path.exists():
            detections = json.loads(det_path.read_text())
            console.print(f"  [dim]Loaded {len(detections)} detections from cache[/dim]")
        else:
            console.print("[red]No detections.json found. Run 'detect' phase first.[/red]")
            return

    # Phase 2: Geometry
    if not phase_filter or phase_filter == "geometry":
        with metrics.phase("2_geometry") as pm:
            geometry_records = run_phase2(video_path, detections, cfg, dirs, pm)

    if phase_filter and phase_filter not in ("detect", "geometry") and not geometry_records:
        import json

        geo_path = dirs["base"] / "geometry.json"
        if geo_path.exists():
            geometry_records = json.loads(geo_path.read_text())
            console.print(
                f"  [dim]Loaded {len(geometry_records)} geometry records from cache[/dim]"
            )
        else:
            console.print("[red]No geometry.json found. Run 'geometry' phase first.[/red]")
            return

    # Phase 3: Tracking
    if not phase_filter or phase_filter == "track":
        with metrics.phase("3_tracking") as pm:
            transitions = run_phase3(geometry_records, cfg, dirs, pm)

    if phase_filter == "output" and not transitions:
        import json

        trans_path = dirs["base"] / "transitions.json"
        if trans_path.exists():
            data = json.loads(trans_path.read_text())
            transitions = data.get("transitions", [])
            console.print(
                f"  [dim]Loaded {len(transitions)} transitions from cache[/dim]"
            )
        else:
            console.print("[red]No transitions.json found. Run 'track' phase first.[/red]")
            return

    # Phase 4: Output
    if not phase_filter or phase_filter == "output":
        with metrics.phase("4_output") as pm:
            run_phase4(transitions, geometry_records, cfg, dirs, pm)

    metrics.end_time = time.time()
    metrics.save(dirs["base"] / "metrics.json")
    metrics.print_summary()
