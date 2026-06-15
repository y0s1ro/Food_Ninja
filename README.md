# School Canteen Food-Waste Prototype

An MVP system for measuring plate leftovers by food category using a regular USB camera and FoodSAM segmentation.

## What this project does

After a student places a plate under the camera:
1. The system detects a plate circle in the frame.
2. Capture is allowed only when the operator arms it (button/key-based arming).
3. The captured plate image is segmented with FoodSAM.
4. Waste percentages are computed per allowed menu category.
5. Results and artifacts are stored in SQLite for analytics and future portion personalization.

## Why this approach

- No depth camera required.
- Uses existing FoodSAM checkpoints and category mapping.
- Supports daily menu constraints to improve practical accuracy.
- Stores structured records for future recommendation logic.

## Main features

- Automatic plate detection (Hough circle based).
- One-shot button-armed auto-capture.
- Daily menu category filtering from JSON.
- FoodSAM semantic + enhanced mask analysis.
- SQLite logging of each capture and per-category metrics.
- Debug logging for denominator/ROI validation.

## Repository components

- `canteen_auto_capture.py`: capture loop, FoodSAM call, waste calculation, SQLite persistence.
- `visualize_segmentation.py`: visualization and category reporting utilities.
- `configs/daily_menu.json`: daily menu configuration.
- `FoodSAM/semantic.py`: FoodSAM semantic pipeline entrypoint.
- `FoodSAM/FoodSAM_tools/category_id_files/foodseg103_category_id.txt`: valid class list.

## Requirements

- Python environment with FoodSAM dependencies.
- OpenCV-compatible USB camera.
- FoodSAM checkpoints in `ckpts/`.

Minimum critical packages (in addition to project requirements):
- `torch`
- `mmcv-full`
- `segment-anything`
- `opencv-python`

See `installation.md` for the original FoodSAM setup guide.

## Setup

### 1) Create and activate environment

Use your preferred tool (`venv` or `conda`) and install dependencies.

### 2) Install dependencies

Install FoodSAM dependencies first (PyTorch, MMCV, SAM), then:

```bash
pip install -r requirement.txt
```

If needed:

```bash
pip install segment-anything
```

### 3) Download checkpoints

Place required checkpoints under `ckpts/`:
- `sam_vit_h_4b8939.pth`
- `SETR_MLA/iter_80000.pth`
- `Unified_learned_OCIM_RS200_6x+2x.pth` (for panoptic workflows)

## Daily menu configuration

Edit `configs/daily_menu.json`:

```json
{
  "default": ["soup", "potato", "pork", "tomato", "rice"],
  "by_date": {
    "2026-06-15": ["pasta", "sausage", "tomato"]
  }
}
```

Menu items may be class names or numeric IDs from:
`FoodSAM/FoodSAM_tools/category_id_files/foodseg103_category_id.txt`

## Run the prototype

### Interactive mode (recommended)

```bash
python canteen_auto_capture.py --show-preview --arm-key a --debug
```

Behavior:
- Press `A` to arm one capture.
- Place plate and wait for stable circle detection.
- After capture, the system disarms automatically.
- Press `Q` to quit.

### Headless mode

```bash
python canteen_auto_capture.py --start-armed
```

## Key CLI options

- `--camera-id`: camera index (default `0`)
- `--menu-config`: menu JSON path
- `--stability-frames`: stable frames required before capture
- `--min-plate-radius`: minimum detected circle radius
- `--arm-key`: arming key in preview mode
- `--start-armed`: arm at startup
- `--debug`: print debug logs
- `--debug-every-n-frames`: periodic debug interval

## Outputs

### Artifacts

- `output/canteen_captures/*_full.jpg`: full frame
- `output/canteen_captures/*_crop.jpg`: cropped plate ROI
- `output/canteen_runs/<capture_id>/`: FoodSAM outputs (`pred_mask.png`, `enhance_mask.png`, etc.)

### Database

SQLite database path:
- `output/canteen/canteen_waste.db`

Tables:
- `captures`: one row per capture
- `capture_items`: per-category waste rows for each capture

## Percentage definition

Per-category percentage is calculated relative to ROI pixels (plate area mask), not the full frame:

- `pct = category_pixels / roi_pixels * 100`

Debug logs print:
- `frame_pixels`
- `roi_pixels`
- `roi_pct_of_frame`
- `total_food_pct`
- `unexpected_pct`

## Known limitations

- Area-based proxy only (no true mass/volume without depth or scales).
- Circle detection may fail with unusual plate shapes or occlusion.
- Food class confusion can occur for visually similar dishes.

## Recommended next improvements

1. Add confidence-based reject/review workflow.
2. Add dashboard for daily and weekly waste trends.
3. Add category-level personalization policy engine.
4. Containerize inference service for cross-machine deployment.

## Acknowledgements

This project builds on FoodSAM: Any Food Segmentation by Lan et al.

- Upstream repository: https://github.com/jamesjg/FoodSAM
- Paper: https://arxiv.org/abs/2308.05938

This repository includes vendored and project-integrated FoodSAM components used for semantic food segmentation within the canteen waste workflow. Additional bundled upstream components are documented in `THIRD_PARTY_NOTICES.md`.

## License

This repository includes third-party components from the public FoodSAM repository and related upstream projects.

- Upstream FoodSAM is licensed under Apache 2.0.
- This repository's local modifications and integration code are distributed under the license in `LICENSE`.
- See `THIRD_PARTY_NOTICES.md` for attribution and bundled third-party component details.
