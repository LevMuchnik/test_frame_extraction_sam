"""Debug visualization helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def draw_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw a semi-transparent mask overlay on a frame."""
    overlay = frame.copy()
    overlay[mask] = (
        overlay[mask] * (1 - alpha) + np.array(color) * alpha
    ).astype(np.uint8)
    return overlay


def draw_corners(
    frame: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    radius: int = 8,
) -> np.ndarray:
    """Draw corner points on a frame."""
    result = frame.copy()
    for i, pt in enumerate(corners):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(result, (x, y), radius, color, -1)
        cv2.putText(
            result, str(i), (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return result


def create_comparison(
    original: np.ndarray,
    deskewed: np.ndarray,
    max_height: int = 540,
) -> np.ndarray:
    """Create a side-by-side comparison image."""
    h1, w1 = original.shape[:2]
    scale = max_height / h1
    resized_orig = cv2.resize(original, (int(w1 * scale), max_height))

    h2, w2 = deskewed.shape[:2]
    scale2 = max_height / h2
    resized_desk = cv2.resize(deskewed, (int(w2 * scale2), max_height))

    gap = np.ones((max_height, 10, 3), dtype=np.uint8) * 128
    return np.hstack([resized_orig, gap, resized_desk])


def save_slide_gallery(
    slides_dir: Path,
    output_path: Path,
    cols: int = 4,
    thumb_width: int = 480,
) -> None:
    """Create a contact sheet / gallery of all extracted slides."""
    slide_files = sorted(slides_dir.glob("slide_*.jpg"))
    if not slide_files:
        return

    thumbs = []
    for f in slide_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_width / w
        thumb = cv2.resize(img, (thumb_width, int(h * scale)))
        thumbs.append(thumb)

    if not thumbs:
        return

    # Pad to fill last row
    thumb_h = thumbs[0].shape[0]
    while len(thumbs) % cols != 0:
        thumbs.append(np.zeros((thumb_h, thumb_width, 3), dtype=np.uint8))

    rows = []
    for i in range(0, len(thumbs), cols):
        rows.append(np.hstack(thumbs[i : i + cols]))

    gallery = np.vstack(rows)
    cv2.imwrite(str(output_path), gallery)
