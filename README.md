# IK from Data - Learning-Based Inverse Kinematics for SO-100 Robot Arm

A PyTorch implementation of learning-based inverse kinematics (IK) for the SO-100 robot arm. Trains a simple MLP to map 2D image positions (detected dots on gripper) to 5-joint positions.

## Quick Start

```bash
# 1. Setup
conda env create -f environment.yml
conda activate ik_lekiwi_so100

# 2. Find initial dot positions
python find_initial_dots.py

# 3. Preprocess (use coordinates from step 2)
python preprocess.py \
  --output output/processed_data.npz \
  --dots-cam1 <x1> <y1> <x2> <y2> \
  --dots-cam2 <x1> <y1> <x2> <y2> \
  --visualize 0

# 4. Train
python train.py \
  --data output/processed_data.npz \
  --epochs 100 \
  --output-dir output/checkpoints

# 5. Inference
python inference.py \
  --model output/checkpoints/best_model.pth \
  --input 320.5 240.3 310.2 235.8
```

## Overview

The SO-100 robot arm costs $110 and has no fancy features - you control it by setting its 6 joint positions. This project adds simplified IK by:

1. Physically marking the gripper jaws with two dots
2. Recording videos and joint positions while moving the arm
3. Using CoTracker to trace dots through frames
4. Training an MLP to map dot positions to joint angles

**Dataset**: HuggingFace [`paszea/ik`](https://huggingface.co/datasets/paszea/ik) - 46,314 frames, 2 camera views

## Model Architecture

Simple MLP:
- **Input**: `[x1, y1, x2, y2]` - middle point positions in two camera images
- **Hidden**: 256 → 128 → 64 neurons (ReLU)
- **Output**: 5 joint positions (gripper fixed at 50)

## Detailed Usage

### Preprocessing Options

```bash
# Basic (recommended)
python preprocess.py \
  --output output/processed_data.npz \
  --dots-cam1 <x1> <y1> <x2> <y2> \
  --dots-cam2 <x1> <y1> <x2> <y2> \
  --visualize 0

# With visualization + W&B
wandb login  # First time only
python preprocess.py \
  --output output/processed_data.npz \
  --dots-cam1 <x1> <y1> <x2> <y2> \
  --dots-cam2 <x1> <y1> <x2> <y2> \
  --visualize 1 \
  --vis-frames 300 \
  --wandb \
  --wandb-project ik-preprocessing

# Quick test (1000 frames)
python preprocess.py \
  --output output/test_data.npz \
  --dots-cam1 <x1> <y1> <x2> <y2> \
  --dots-cam2 <x1> <y1> <x2> <y2> \
  --max-frames 1000 \
  --visualize 1
```

**Key parameters**:
- `--max-frames N`: Process only first N frames (for testing)
- `--vis-frames N`: Visualize N frames per episode (default: all)
- `--frame-offset 6`: Lag compensation (default: 6 frames = 0.2s at 30fps)
- `--gpu-chunk-size 50`: GPU processing chunk size (auto-adjusted)

### Training Options

```bash
# Basic
python train.py \
  --data output/processed_data.npz \
  --epochs 100

# With W&B tracking
wandb login
python train.py \
  --data output/processed_data.npz \
  --epochs 100 \
  --wandb \
  --wandb-project ik-from-data \
  --wandb-name "baseline"
```

**Outputs**: `output/checkpoints/best_model.pth`, `normalization_params.json`, training plots

### Inference

**Command line**:
```bash
python inference.py \
  --model output/checkpoints/best_model.pth \
  --input 320.5 240.3 310.2 235.8
```

**Python API**:
```python
from inference import IKPredictor

predictor = IKPredictor(
    model_path='output/checkpoints/best_model.pth',
    normalization_params_path='output/checkpoints/normalization_params.json'
)

joints = predictor.predict_with_gripper([320.5, 240.3, 310.2, 235.8])
# Returns: [j0, j1, j2, j3, j4, 50.0]
```

## Troubleshooting

### Terminal Crashes (Linux)

**Cause**: `systemd-oomd` kills terminal when memory >50% for >20 seconds.

**Diagnosis**: `journalctl --since "2 hours ago" | grep -i "memory pressure"`

**Solutions**:
- Use `--visualize 0` (saves RAM)
- Limit frames: `--vis-frames 300`
- Run in tmux: `tmux new -s preprocess` (survives kills)

### GPU Out of Memory

**Solutions**:
- Reduce GPU chunk: `--gpu-chunk-size 32`
- Force CPU: `CUDA_VISIBLE_DEVICES="" python preprocess.py ...`

### Verify Dot Positions

```bash
# Check if coordinates are correct
python utils/verify_dots.py

# Debug tracking in first 10 frames
python utils/debug_tracking.py
```

## Project Structure

```
ik_from_data/
├── model.py, dataset.py, inference.py  # Core components
├── preprocess.py                        # Step 1: Extract dot positions
├── train.py                             # Step 2: Train model
├── find_initial_dots.py                 # Helper: Find initial dots
├── utils/                               # Utility scripts
│   ├── memory_utils.py                  # Memory monitoring
│   ├── verify_dots.py, debug_tracking.py # Diagnostic tools
│   └── explore_dataset*.py, example.py  # Dataset exploration
├── data/                                # Input videos (not tracked)
└── output/                              # Generated files (not tracked)
    ├── processed_data.npz
    ├── checkpoints/
    └── tracking_visualizations/
```

## Technical Details

- **Framework**: PyTorch
- **Tracking**: CoTracker3 (via torch.hub)
- **Loss**: MSE, Optimizer: Adam, Scheduler: ReduceLROnPlateau
- **Data split**: 80/20 train/val
- **Normalization**: Z-score (inputs + outputs)
- **Memory**: Streaming architecture prevents OOM (~1-2 GB RAM vs ~15-20 GB)

## Citation

- Dataset: https://huggingface.co/datasets/paszea/ik
- CoTracker: https://github.com/facebookresearch/co-tracker

## License

MIT License
