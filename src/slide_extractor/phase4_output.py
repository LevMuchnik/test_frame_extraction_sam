"""Phase 4: Timestamping & Output — best-frame selection and final export."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console

from .config import Config
from .metrics import PhaseMetrics

console = Console(force_terminal=True, legacy_windows=False)


def _compute_sharpness(image: np.ndarray) -> float:
    """Compute Laplacian variance as a sharpness measure."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def run_phase4(
    transitions: list[dict],
    geometry_records: list[dict],
    cfg: Config,
    dirs: dict[str, Path],
    pm: PhaseMetrics,
) -> None:
    """Select best frame per slide and produce final outputs.

    For each slide (transition), picks the frame with the highest
    combined score (detection confidence x sharpness).
    """
    if not transitions:
        console.print("  [yellow]No slides to output[/yellow]")
        return

    # Index geometry records by sample_idx for fast lookup
    geo_by_idx = {r["sample_idx"]: r for r in geometry_records if r.get("deskewed_file")}

    slides_csv_rows: list[dict] = []

    for slide in transitions:
        slide_num = slide["slide_num"]
        start_idx = slide["start_sample_idx"]
        end_idx = slide["end_sample_idx"]

        # Find all deskewed frames in this slide's range
        candidates = [
            geo_by_idx[idx]
            for idx in range(start_idx, end_idx + 1)
            if idx in geo_by_idx
        ]

        if not candidates:
            pm.inc("slides_no_frames")
            continue

        # Score each candidate: detection_score * sharpness
        best_candidate = None
        best_combined = -1.0

        for cand in candidates:
            img_path = dirs["deskewed"] / cand["deskewed_file"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            sharpness = _compute_sharpness(img)
            det_score = cand.get("score", 0.5)
            combined = det_score * sharpness

            if combined > best_combined:
                best_combined = combined
                best_candidate = {
                    **cand,
                    "sharpness": round(sharpness, 2),
                    "combined_score": round(combined, 2),
                }

        if best_candidate is None:
            pm.inc("slides_no_frames")
            continue

        # Copy best frame to slides directory
        src = dirs["deskewed"] / best_candidate["deskewed_file"]
        dst = dirs["slides"] / f"slide_{slide_num:03d}.jpg"
        shutil.copy2(str(src), str(dst))
        pm.inc("slides_saved")

        slides_csv_rows.append({
            "slide_num": slide_num,
            "start_time": slide["start_timestamp"],
            "end_time": slide["end_timestamp"],
            "duration": slide.get("duration", 0),
            "file": dst.name,
            "confidence": best_candidate.get("score", 0),
            "sharpness": best_candidate["sharpness"],
            "combined_score": best_candidate["combined_score"],
        })

    # Write slides.csv
    csv_path = dirs["base"] / "slides.csv"
    if slides_csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=slides_csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(slides_csv_rows)

    # Summary stats
    if slides_csv_rows:
        durations = [r["duration"] for r in slides_csv_rows]
        pm.set("avg_slide_duration", sum(durations) / len(durations))
        pm.set("total_slides", len(slides_csv_rows))

    console.print(
        f"  Saved {pm.counters.get('slides_saved', 0)} slides to {dirs['slides']}"
    )
    console.print(f"  Index: {csv_path}")
