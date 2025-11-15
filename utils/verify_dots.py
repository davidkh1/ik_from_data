"""
Verify initial dot positions by overlaying them on the first frame.
This helps diagnose coordinate issues (x/y swap, rotation, etc.).

Usage:
    python utils/verify_dots.py
"""

import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import numpy as np

# Configuration - UPDATE THESE WITH YOUR DOTS
DOTS_CAM1 = [340, 181, 416, 166]  # x1, y1, x2, y2
DOTS_CAM2 = [141, 179, 149, 119]  # x1, y1, x2, y2

VIDEO_ROOT_DIR = Path("data/ik_dataset/videos/chunk-000")


def verify_dot_positions():
    """Load first frames and overlay the specified dot positions."""

    # Load first frames
    front_video = VIDEO_ROOT_DIR / "observation.images.front" / "episode_000000.mp4"
    wrist_video = VIDEO_ROOT_DIR / "observation.images.wrist" / "episode_000000.mp4"

    front_frames = iio.imread(str(front_video), plugin="FFMPEG")
    wrist_frames = iio.imread(str(wrist_video), plugin="FFMPEG")

    first_front = front_frames[0]
    first_wrist = wrist_frames[0]

    print(f"Front frame shape: {first_front.shape}")
    print(f"Wrist frame shape: {first_wrist.shape}")

    # Parse dot positions
    dots_cam1 = np.array(DOTS_CAM1).reshape(2, 2)  # [[x1, y1], [x2, y2]]
    dots_cam2 = np.array(DOTS_CAM2).reshape(2, 2)

    print(f"\nFront camera dots (x, y format):")
    print(f"  Dot 1: ({dots_cam1[0][0]}, {dots_cam1[0][1]})")
    print(f"  Dot 2: ({dots_cam1[1][0]}, {dots_cam1[1][1]})")

    print(f"\nWrist camera dots (x, y format):")
    print(f"  Dot 1: ({dots_cam2[0][0]}, {dots_cam2[0][1]})")
    print(f"  Dot 2: ({dots_cam2[1][0]}, {dots_cam2[1][1]})")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Row 0: Original (x, y) interpretation
    axes[0, 0].imshow(first_front)
    axes[0, 0].set_title('Front Camera - Using (X, Y)', fontsize=12, weight='bold')
    for i, (x, y) in enumerate(dots_cam1):
        color = 'green' if i == 0 else 'red'
        axes[0, 0].plot(x, y, 'o', color=color, markersize=15, markeredgewidth=3, fillstyle='none')
        axes[0, 0].plot(x, y, '+', color=color, markersize=20, markeredgewidth=3)
        axes[0, 0].text(x+10, y-10, f'Dot {i+1}: ({x}, {y})', color=color, fontsize=10, weight='bold')

    axes[0, 1].imshow(first_wrist)
    axes[0, 1].set_title('Wrist Camera - Using (X, Y)', fontsize=12, weight='bold')
    for i, (x, y) in enumerate(dots_cam2):
        color = 'green' if i == 0 else 'red'
        axes[0, 1].plot(x, y, 'o', color=color, markersize=15, markeredgewidth=3, fillstyle='none')
        axes[0, 1].plot(x, y, '+', color=color, markersize=20, markeredgewidth=3)
        axes[0, 1].text(x+10, y-10, f'Dot {i+1}: ({x}, {y})', color=color, fontsize=10, weight='bold')

    # Row 1: Swapped (y, x) interpretation
    axes[1, 0].imshow(first_front)
    axes[1, 0].set_title('Front Camera - Using (Y, X) SWAPPED', fontsize=12, weight='bold', color='orange')
    for i, (x, y) in enumerate(dots_cam1):
        # Swap x and y
        color = 'green' if i == 0 else 'red'
        axes[1, 0].plot(y, x, 'o', color=color, markersize=15, markeredgewidth=3, fillstyle='none')
        axes[1, 0].plot(y, x, '+', color=color, markersize=20, markeredgewidth=3)
        axes[1, 0].text(y+10, x-10, f'Dot {i+1}: ({y}, {x})', color=color, fontsize=10, weight='bold')

    axes[1, 1].imshow(first_wrist)
    axes[1, 1].set_title('Wrist Camera - Using (Y, X) SWAPPED', fontsize=12, weight='bold', color='orange')
    for i, (x, y) in enumerate(dots_cam2):
        # Swap x and y
        color = 'green' if i == 0 else 'red'
        axes[1, 1].plot(y, x, 'o', color=color, markersize=15, markeredgewidth=3, fillstyle='none')
        axes[1, 1].plot(y, x, '+', color=color, markersize=20, markeredgewidth=3)
        axes[1, 1].text(y+10, x-10, f'Dot {i+1}: ({y}, {x})', color=color, fontsize=10, weight='bold')

    plt.tight_layout()

    # Save
    output_file = Path("output/dot_verification.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Verification image saved to: {output_file}")

    print("\n" + "=" * 70)
    print("CHECK THE IMAGE:")
    print("=" * 70)
    print("Top row: Using coordinates as (X, Y)")
    print("Bottom row: Using coordinates as (Y, X) SWAPPED")
    print("\nWhich row shows dots on the GRIPPER JAWS correctly?")
    print("  - If TOP row is correct: Coordinates are fine")
    print("  - If BOTTOM row is correct: We need to swap X and Y in the code!")
    print("=" * 70)

    plt.show()


if __name__ == '__main__':
    verify_dot_positions()
