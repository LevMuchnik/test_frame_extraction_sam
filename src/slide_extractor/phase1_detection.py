"""Phase 1: Smart Sampling + SAM 3 text-prompted detection."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress, TimeElapsedColumn

from .config import Config
from .metrics import PhaseMetrics
from .models import detect_slide, load_sam3

console = Console(force_terminal=True, legacy_windows=False)


def _mask_to_rle(mask: np.ndarray) -> dict:
    """Encode a binary mask as run-length encoding for JSON storage."""
    pixels = mask.flatten()
    runs = []
    prev = False
    count = 0
    for p in pixels:
        if p == prev:
            count += 1
        else:
            if count > 0:
                runs.append(count)
            count = 1
            prev = p
    runs.append(count)
    return {
        "shape": list(mask.shape),
        "starts_with": bool(pixels[0]) if len(pixels) > 0 else False,
        "runs": runs,
    }


def run_phase1(
    video_path: Path,
    cfg: Config,
    dirs: dict[str, Path],
    pm: PhaseMetrics,
) -> list[dict]:
    """Extract frames and run SAM 3 detection.

    Returns list of detection records (one per sampled frame).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(native_fps / cfg.fps))
    expected_samples = total_frames // frame_skip

    pm.set("native_fps", native_fps)
    pm.set("target_fps", cfg.fps)
    pm.set("frame_skip", frame_skip)

    console.print(
        f"  Video: {native_fps:.1f} fps, {total_frames} frames, "
        f"sampling every {frame_skip} frames (~{expected_samples} samples)"
    )

    # Load SAM 3
    _, processor = load_sam3(cfg)

    detections: list[dict] = []
    frame_idx = 0
    sample_idx = 0

    with Progress(
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Detecting slides...", total=expected_samples)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / native_fps
            pm.inc("frames_sampled")

            # Save debug frame
            if cfg.save_debug_frames:
                cv2.imwrite(
                    str(dirs["frames"] / f"frame_{sample_idx:05d}.jpg"), frame
                )

            # Run SAM 3 detection
            result = detect_slide(
                image=frame,
                processor=processor,
                prompt=cfg.positive_prompt,
                confidence_threshold=cfg.detection_confidence,
            )

            record: dict = {
                "sample_idx": sample_idx,
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 3),
                "detected": result is not None,
            }

            if result is not None:
                pm.inc("frames_detected")
                record["score"] = result["score"]
                record["box"] = result["box"]
                record["mask_rle"] = _mask_to_rle(result["mask"])

                # Save debug mask overlay
                if cfg.save_debug_frames:
                    overlay = frame.copy()
                    overlay[result["mask"]] = (
                        overlay[result["mask"]] * 0.5
                        + np.array([0, 255, 0]) * 0.5
                    ).astype(np.uint8)
                    x1, y1, x2, y2 = [int(c) for c in result["box"]]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        overlay,
                        f"{result['score']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imwrite(
                        str(dirs["masks"] / f"mask_{sample_idx:05d}.jpg"), overlay
                    )
            else:
                pm.inc("frames_absent")

            detections.append(record)
            sample_idx += 1
            frame_idx += 1
            progress.advance(task)

    cap.release()

    # Compute summary stats
    detected_scores = [d["score"] for d in detections if d["detected"]]
    if detected_scores:
        pm.set("detection_rate", len(detected_scores) / len(detections))
        pm.set("avg_score", sum(detected_scores) / len(detected_scores))
        pm.set("min_score", min(detected_scores))
        pm.set("max_score", max(detected_scores))

    # Save detections manifest
    output_path = dirs["base"] / "detections.json"
    output_path.write_text(json.dumps(detections, indent=2))
    console.print(
        f"  Detected slides in {pm.counters.get('frames_detected', 0)}"
        f"/{pm.counters.get('frames_sampled', 0)} frames"
    )

    return detections
