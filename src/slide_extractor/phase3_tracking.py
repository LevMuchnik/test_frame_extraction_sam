"""Phase 3: Content Tracking — SSIM-based slide transition detection."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from skimage.metrics import structural_similarity as ssim

from .config import Config
from .metrics import PhaseMetrics

console = Console()


def run_phase3(
    geometry_records: list[dict],
    cfg: Config,
    dirs: dict[str, Path],
    pm: PhaseMetrics,
) -> list[dict]:
    """Compare consecutive deskewed slides via SSIM to find transitions.

    Returns list of transition records with SSIM scores.
    """
    # Filter to successfully deskewed frames
    deskewed = [r for r in geometry_records if r.get("deskewed_file")]
    if not deskewed:
        console.print("  [yellow]No deskewed frames to compare[/yellow]")
        return []

    transitions: list[dict] = []
    ssim_scores: list[dict] = []
    prev_gray: np.ndarray | None = None
    current_slide_start = deskewed[0]

    for i, rec in enumerate(deskewed):
        img_path = dirs["deskewed"] / rec["deskewed_file"]
        img = cv2.imread(str(img_path))
        if img is None:
            pm.inc("read_errors")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            # First frame — start of first slide
            prev_gray = gray
            transitions.append({
                "slide_num": 0,
                "start_timestamp": rec["timestamp"],
                "start_sample_idx": rec["sample_idx"],
                "start_frame_idx": rec["frame_idx"],
            })
            pm.inc("slides_found")
            continue

        # Compute SSIM
        score = ssim(prev_gray, gray)
        ssim_scores.append({
            "sample_idx": rec["sample_idx"],
            "timestamp": rec["timestamp"],
            "ssim": round(score, 4),
        })

        if score < cfg.ssim_threshold:
            # Slide transition detected
            pm.inc("transitions")

            # Close previous slide
            transitions[-1]["end_timestamp"] = rec["timestamp"]
            transitions[-1]["end_sample_idx"] = rec["sample_idx"]
            transitions[-1]["end_frame_idx"] = rec["frame_idx"]
            transitions[-1]["duration"] = round(
                rec["timestamp"] - transitions[-1]["start_timestamp"], 3
            )

            # Open new slide
            transitions.append({
                "slide_num": len(transitions),
                "start_timestamp": rec["timestamp"],
                "start_sample_idx": rec["sample_idx"],
                "start_frame_idx": rec["frame_idx"],
            })
            pm.inc("slides_found")

        prev_gray = gray

    # Close the last slide (end = last frame)
    if transitions and "end_timestamp" not in transitions[-1]:
        last = deskewed[-1]
        transitions[-1]["end_timestamp"] = last["timestamp"]
        transitions[-1]["end_sample_idx"] = last["sample_idx"]
        transitions[-1]["end_frame_idx"] = last["frame_idx"]
        transitions[-1]["duration"] = round(
            last["timestamp"] - transitions[-1]["start_timestamp"], 3
        )

    # Compute stats
    if ssim_scores:
        all_ssim = [s["ssim"] for s in ssim_scores]
        pm.set("avg_ssim", sum(all_ssim) / len(all_ssim))
        pm.set("min_ssim", min(all_ssim))

    # Save outputs
    output = {
        "transitions": transitions,
        "ssim_timeline": ssim_scores,
    }
    output_path = dirs["base"] / "transitions.json"
    output_path.write_text(json.dumps(output, indent=2))

    console.print(
        f"  Found {pm.counters.get('slides_found', 0)} unique slides, "
        f"{pm.counters.get('transitions', 0)} transitions"
    )

    return transitions
