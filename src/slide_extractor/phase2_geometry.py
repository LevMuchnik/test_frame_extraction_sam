"""Phase 2: Geometric Refinement — contour approximation + homography deskewing."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from .config import Config
from .metrics import PhaseMetrics

console = Console()


def _rle_to_mask(rle: dict) -> np.ndarray:
    """Decode RLE back to a binary mask."""
    shape = rle["shape"]
    starts_with = rle["starts_with"]
    runs = rle["runs"]

    pixels = []
    current = starts_with
    for run in runs:
        pixels.extend([current] * run)
        current = not current

    return np.array(pixels, dtype=bool).reshape(shape)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(d)]  # bottom-left has largest difference
    return rect


def process_frame_geometry(
    frame: np.ndarray,
    mask: np.ndarray,
    epsilon_factor: float,
    deskew_width: int,
    deskew_height: int,
) -> tuple[np.ndarray | None, dict]:
    """Apply contour approximation and homography to a single frame.

    Returns:
        (deskewed_image or None, geometry_info dict)
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, {"status": "no_contours"}

    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(largest, epsilon, True)

    info = {
        "contour_area": float(cv2.contourArea(largest)),
        "perimeter": float(perimeter),
        "num_corners": len(approx),
    }

    if len(approx) != 4:
        info["status"] = "not_4_corners"
        return None, info

    corners = _order_points(approx.reshape(4, 2))
    info["corners"] = corners.tolist()
    info["status"] = "ok"

    # Destination points for the flat slide
    dst = np.array(
        [
            [0, 0],
            [deskew_width - 1, 0],
            [deskew_width - 1, deskew_height - 1],
            [0, deskew_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    deskewed = cv2.warpPerspective(frame, M, (deskew_width, deskew_height))

    return deskewed, info


def run_phase2(
    video_path: Path,
    detections: list[dict],
    cfg: Config,
    dirs: dict[str, Path],
    pm: PhaseMetrics,
) -> list[dict]:
    """Process detected frames: contour approx + homography deskew.

    Returns list of geometry records with deskewed image paths.
    """
    # Filter to detected frames only
    detected = [d for d in detections if d["detected"]]
    if not detected:
        console.print("  [yellow]No detections to process[/yellow]")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    geometry_records: list[dict] = []
    corner_counts: dict[int, int] = {}

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Deskewing slides...", total=len(detected))

        for det in detected:
            frame_idx = det["frame_idx"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                pm.inc("read_errors")
                progress.advance(task)
                continue

            mask = _rle_to_mask(det["mask_rle"])
            deskewed, info = process_frame_geometry(
                frame=frame,
                mask=mask,
                epsilon_factor=cfg.contour_epsilon,
                deskew_width=cfg.deskew_width,
                deskew_height=cfg.deskew_height,
            )

            # Track corner count distribution
            nc = info.get("num_corners", 0)
            corner_counts[nc] = corner_counts.get(nc, 0) + 1

            record = {
                "sample_idx": det["sample_idx"],
                "frame_idx": frame_idx,
                "timestamp": det["timestamp"],
                "score": det["score"],
                **info,
            }

            if deskewed is not None:
                pm.inc("deskewed_ok")
                fname = f"deskewed_{det['sample_idx']:05d}.jpg"
                cv2.imwrite(str(dirs["deskewed"] / fname), deskewed)
                record["deskewed_file"] = fname
            else:
                pm.inc("deskewed_skipped")

            geometry_records.append(record)
            progress.advance(task)

    cap.release()

    # Log corner distribution
    console.print(f"  Corner distribution: {corner_counts}")
    pm.set(
        "geometry_success_rate",
        pm.counters.get("deskewed_ok", 0) / max(len(detected), 1),
    )

    # Save geometry manifest
    output_path = dirs["base"] / "geometry.json"
    output_path.write_text(json.dumps(geometry_records, indent=2))
    console.print(
        f"  Deskewed {pm.counters.get('deskewed_ok', 0)}/{len(detected)} frames"
    )

    return geometry_records
