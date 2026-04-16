# Lecture Slide Extraction Pipeline (SAM 3.1)

Automated lecture slide extraction from videos using Meta's **Segment Anything Model 3.1** (SAM 3.1).

The pipeline detects presentation slides in video frames using text-prompted segmentation, deskews them via homography, tracks content changes via SSIM, and outputs clean, timestamped slide images.

## Setup

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA 12.6+ (recommended)
- [uv](https://docs.astral.sh/uv/) package manager
- HuggingFace account with access to [facebook/sam3](https://huggingface.co/facebook/sam3)

### Installation

```bash
# Clone the repository
git clone https://github.com/LevMuchnik/test_frame_extraction_sam.git
cd test_frame_extraction_sam

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your HuggingFace token

# Login to HuggingFace (for model weights)
uv run huggingface-cli login

# Download SAM 3.1 model weights
uv run slide-extractor download-models
```

## Usage

### Process a single video

```bash
uv run slide-extractor process sample_videos/lecture.mp4
```

### Process all videos in a directory

```bash
uv run slide-extractor process sample_videos/
```

### Key options

```bash
# Custom frame rate (default: 1 fps)
uv run slide-extractor process video.mp4 --fps 0.5

# Save debug frames and mask overlays
uv run slide-extractor process video.mp4 --debug

# Adjust detection confidence
uv run slide-extractor process video.mp4 --confidence 0.6

# Adjust slide change sensitivity
uv run slide-extractor process video.mp4 --ssim-threshold 0.90

# Run on CPU
uv run slide-extractor process video.mp4 --device cpu

# Run a single phase
uv run slide-extractor process video.mp4 --phase detect
```

### Video info

```bash
uv run slide-extractor info sample_videos/lecture.mp4
```

## Pipeline Phases

| Phase | Description | Output |
|-------|-------------|--------|
| **1. Detection** | Sample frames at target FPS, run SAM 3.1 text-prompted segmentation | `detections.json`, debug frames/masks |
| **2. Geometry** | Douglas-Peucker contour -> 4 corners -> homography deskew | `deskewed/*.jpg`, `geometry.json` |
| **3. Tracking** | SSIM comparison to detect slide transitions | `transitions.json` |
| **4. Output** | Best-frame selection per slide, timestamps | `slides/*.jpg`, `slides.csv`, `metrics.json` |

Each phase produces its own JSON manifest, so you can re-run later phases independently with different thresholds.

## Output Structure

```
outputs/{video_name}/
├── frames/           # Sampled frames (--debug only)
├── masks/            # Annotated detection masks (--debug only)
├── deskewed/         # All deskewed slide frames
├── slides/           # Final best-frame per unique slide
├── detections.json   # Phase 1 results
├── geometry.json     # Phase 2 results
├── transitions.json  # Phase 3 results + SSIM timeline
├── slides.csv        # Slide index with timestamps
└── metrics.json      # Aggregate stats and phase timings
```

## Configuration

All settings can be configured via `.env` file or CLI flags. CLI flags take precedence.

See `.env.example` for all available settings.
