"""Configuration management — loads from .env, with CLI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _bool(val: str) -> bool:
    return val.lower() in ("true", "1", "yes")


@dataclass
class Config:
    # Model
    sam3_model: str = "sam3.1_hiera_large"
    model_cache_dir: Path = field(default_factory=lambda: Path("./models"))
    hf_token: str = ""

    # Detection
    positive_prompt: str = "slide . presentation"
    detection_confidence: float = 0.3

    # Geometry
    deskew_width: int = 1920
    deskew_height: int = 1080
    contour_epsilon: float = 0.02

    # Tracking
    ssim_threshold: float = 0.85

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    save_debug_frames: bool = False

    # Performance
    device: str = "cuda"

    # Sampling
    fps: float = 1.0

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> Config:
        """Load config from .env file, falling back to defaults."""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        # Use string defaults for fields that use default_factory
        return cls(
            sam3_model=os.getenv("SAM3_MODEL", "sam3.1_hiera_large"),
            model_cache_dir=Path(os.getenv("MODEL_CACHE_DIR", "./models")),
            hf_token=os.getenv("HF_TOKEN", ""),
            positive_prompt=os.getenv("POSITIVE_PROMPT", "slide . presentation"),
            detection_confidence=float(os.getenv("DETECTION_CONFIDENCE", "0.3")),
            deskew_width=int(os.getenv("DESKEW_WIDTH", "1920")),
            deskew_height=int(os.getenv("DESKEW_HEIGHT", "1080")),
            contour_epsilon=float(os.getenv("CONTOUR_EPSILON", "0.02")),
            ssim_threshold=float(os.getenv("SSIM_THRESHOLD", "0.85")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "./outputs")),
            save_debug_frames=_bool(os.getenv("SAVE_DEBUG_FRAMES", "false")),
            device=os.getenv("DEVICE", "cuda"),
        )

    def video_output_dir(self, video_path: Path) -> Path:
        """Return the output directory for a specific video."""
        name = video_path.stem
        return self.output_dir / name

    def ensure_dirs(self, video_path: Path) -> dict[str, Path]:
        """Create and return all output subdirectories for a video."""
        base = self.video_output_dir(video_path)
        dirs = {
            "base": base,
            "frames": base / "frames",
            "masks": base / "masks",
            "deskewed": base / "deskewed",
            "slides": base / "slides",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs
