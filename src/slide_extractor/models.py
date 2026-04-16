"""SAM 3.1 model loading, checkpoint management, and inference helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rich.console import Console

from .config import Config

console = Console(force_terminal=True, legacy_windows=False)

# Lazy-loaded model references
_image_model = None
_processor = None


def download_sam3_weights(cfg: Config) -> Path:
    """Download SAM 3.1 checkpoints via huggingface_hub."""
    from huggingface_hub import snapshot_download

    cache_dir = cfg.model_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]Downloading SAM 3.1 weights to {cache_dir}...[/dim]")

    kwargs = {}
    if cfg.hf_token:
        kwargs["token"] = cfg.hf_token

    path = snapshot_download(
        repo_id="facebook/sam3",
        cache_dir=str(cache_dir),
        **kwargs,
    )
    console.print(f"[green]Weights cached at {path}[/green]")
    return Path(path)


def load_sam3(cfg: Config) -> tuple:
    """Load SAM 3 image model and processor. Returns (model, processor).

    Uses lazy loading — subsequent calls return cached references.
    """
    global _image_model, _processor

    if _image_model is not None and _processor is not None:
        return _image_model, _processor

    console.print(f"[dim]Loading SAM 3.1 on {cfg.device}...[/dim]")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    _image_model = build_sam3_image_model(device=cfg.device)
    _processor = Sam3Processor(
        _image_model,
        device=cfg.device,
        confidence_threshold=0.05,  # Low threshold; we filter ourselves in detect_slide
    )

    console.print("[green]SAM 3.1 loaded[/green]")
    return _image_model, _processor


def detect_slide(
    image: np.ndarray,
    processor,
    prompt: str,
    confidence_threshold: float,
) -> dict | None:
    """Run SAM 3 text-prompted detection on a single frame.

    Args:
        image: BGR numpy array from OpenCV.
        processor: Sam3Processor instance.
        prompt: Text prompt (e.g., "projector screen . presentation slide").
        confidence_threshold: Minimum score to accept a detection.

    Returns:
        Dict with keys: mask (np.ndarray bool), score (float), box (np.ndarray)
        or None if no detection passes the threshold.
    """
    # Convert BGR to RGB PIL image
    pil_image = Image.fromarray(image[:, :, ::-1])

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(pil_image)
        output = processor.set_text_prompt(prompt, inference_state)

    masks = output["masks"]  # (N, 1, H, W) bool tensor
    scores = output["scores"]  # (N,) confidence scores
    boxes = output["boxes"]  # (N, 4) bounding boxes [x0, y0, x1, y1]

    # Convert to numpy (cast from bfloat16 to float32 first)
    if isinstance(scores, torch.Tensor):
        scores = scores.float().cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.float().cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.float().cpu().numpy()

    if len(scores) == 0:
        return None

    # Find best detection above threshold
    best_idx = int(np.argmax(scores))
    if scores[best_idx] < confidence_threshold:
        return None

    # masks shape: (N, 1, H, W) -> squeeze to (H, W)
    mask = masks[best_idx]
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    return {
        "mask": mask.astype(bool),
        "score": float(scores[best_idx]),
        "box": boxes[best_idx].tolist(),
    }
