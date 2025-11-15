"""
Debug tracking by showing first frame with initial dots overlaid
and a few sampled frames with tracked positions.

Usage:
    python utils/debug_tracking.py
"""

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Your dot positions
DOTS_CAM1 = [340, 181, 416, 166]  # x1, y1, x2, y2
VIDEO_ROOT_DIR = Path("data/ik_dataset/videos/chunk-000")

# Load first frame
front_video = VIDEO_ROOT_DIR / "observation.images.front" / "episode_000000.mp4"
print(f"Loading first 10 frames from {front_video.name}...")

frames = []
for i, frame in enumerate(iio.imiter(str(front_video), plugin="FFMPEG")):
    frames.append(frame)
    if i >= 9:  # Get first 10 frames
        break

print(f"Loaded {len(frames)} frames")
print(f"Frame shape: {frames[0].shape}")

# Parse dots
dots = np.array(DOTS_CAM1).reshape(2, 2)  # [[x1, y1], [x2, y2]]

# Create comparison
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Front Camera: First 10 Frames with Initial Dot Positions', fontsize=16, weight='bold')

for i in range(10):
    row = i // 5
    col = i % 5

    axes[row, col].imshow(frames[i])
    axes[row, col].set_title(f'Frame {i}', fontsize=10)
    axes[row, col].axis('off')

    # Draw dots
    for j, (x, y) in enumerate(dots):
        color = 'green' if j == 0 else 'red'
        axes[row, col].plot(x, y, 'o', color=color, markersize=10, markeredgewidth=2, fillstyle='none')
        axes[row, col].plot(x, y, '+', color=color, markersize=15, markeredgewidth=2)

plt.tight_layout()
output_file = Path("output/debug_first_10_frames.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n[OK] Saved to: {output_file}")

print("\n" + "=" * 70)
print("CHECK THE IMAGE:")
print("=" * 70)
print("Do the green and red dots stay on the gripper jaws across all 10 frames?")
print("  - If YES: Dots are moving with the gripper, initial positions are good")
print("  - If NO: Dots are fixed while gripper moves - this is what you clicked,")
print("          but CoTracker should be tracking them as they move!")
print("=" * 70)

plt.show()
