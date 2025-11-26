"""
View the first frame from both camera videos to identify initial dot positions.
It requires videos from the dataset.

Usage:
  python find_initial_dots.py
"""

import sys
import imageio.v3 as iio
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Circle

# Configuration
VIDEO_ROOT_DIR = Path("data/ik_dataset/videos/chunk-000")


def view_first_frames():
    """Load and display first frames from both cameras."""

    # Paths to video files (episode 0)
    front_video = VIDEO_ROOT_DIR / "observation.images.front" / "episode_000000.mp4"
    wrist_video = VIDEO_ROOT_DIR / "observation.images.wrist" / "episode_000000.mp4"

    if not front_video.exists():
        print(f"Error: Front camera video not found at {front_video}")
        return

    if not wrist_video.exists():
        print(f"Error: Wrist camera video not found at {wrist_video}")
        return

    print("Loading videos...")

    # Load first frame from each video
    front_frames = iio.imread(str(front_video), plugin="FFMPEG")
    wrist_frames = iio.imread(str(wrist_video), plugin="FFMPEG")

    first_front = front_frames[0]
    first_wrist = wrist_frames[0]

    print(f"Front camera: {first_front.shape}, {len(front_frames)} frames")
    print(f"Wrist camera: {first_wrist.shape}, {len(wrist_frames)} frames")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Front camera
    axes[0].imshow(first_front)
    axes[0].set_title('Front Camera - First Frame\nLook for colored dots on gripper jaws', fontsize=12)
    axes[0].set_xlabel(f'Width: {first_front.shape[1]} pixels')
    axes[0].set_ylabel(f'Height: {first_front.shape[0]} pixels')

    # Wrist camera
    axes[1].imshow(first_wrist)
    axes[1].set_title('Wrist Camera - First Frame\nLook for colored dots on gripper jaws', fontsize=12)
    axes[1].set_xlabel(f'Width: {first_wrist.shape[1]} pixels')
    axes[1].set_ylabel(f'Height: {first_wrist.shape[0]} pixels')

    # Add grid lines every 25 pixels
    grid_spacing = 25
    for ax, frame in [(axes[0], first_front), (axes[1], first_wrist)]:
        h, w = frame.shape[:2]
        ax.set_xticks(range(0, w + 1, grid_spacing), minor=True)
        ax.set_yticks(range(0, h + 1, grid_spacing), minor=True)
        ax.grid(which='minor', color='yellow', linestyle='-', linewidth=0.3, alpha=0.5)
        ax.set_xticks(range(0, w + 1, 100))
        ax.set_yticks(range(0, h + 1, 100))
        ax.grid(which='major', color='yellow', linestyle='-', linewidth=0.6, alpha=0.7)

    plt.tight_layout()

    # Save image
    output_file = Path("output/first_frames.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] {output_file}")

    # Show plot
    try:
        print("\nDisplaying frames... (close window to continue)")
        plt.show()
    except:
        print("(Display not available, check the saved image)")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Open output/first_frames.png")
    print("2. Identify the pixel coordinates of the TWO dots on the gripper")
    print("3. Note the coordinates for BOTH cameras")
    print("\nOr run without --view-only to interactively click on dots:")
    print("  python find_initial_dots.py")


def interactive_click_tool():
    """Interactive tool to click and identify dot positions."""

    # Paths to video files
    front_video = VIDEO_ROOT_DIR / "observation.images.front" / "episode_000000.mp4"
    wrist_video = VIDEO_ROOT_DIR / "observation.images.wrist" / "episode_000000.mp4"

    print("=" * 70)
    print("Interactive Dot Position Finder")
    print("=" * 70)
    print("\nLoading videos...")

    # Load first frames
    front_frames = iio.imread(str(front_video), plugin="FFMPEG")
    wrist_frames = iio.imread(str(wrist_video), plugin="FFMPEG")

    first_front = front_frames[0]
    first_wrist = wrist_frames[0]

    # Store clicked points
    front_points = []
    wrist_points = []

    def onclick(event):
        if event.inaxes == axes[0]:  # Front camera
            if event.xdata is not None and event.ydata is not None and len(front_points) < 2:
                x, y = int(event.xdata), int(event.ydata)
                front_points.append([x, y])
                print(f"Front camera - Dot {len(front_points)}: ({x}, {y})")

                # Draw marker
                circle = Circle((x, y), 8, color='red', fill=False, linewidth=3)
                axes[0].add_patch(circle)
                axes[0].plot(x, y, 'r+', markersize=15, markeredgewidth=3)
                fig.canvas.draw()

                if len(front_points) == 2:
                    print("[OK] Front camera dots identified!")

        elif event.inaxes == axes[1]:  # Wrist camera
            if event.xdata is not None and event.ydata is not None and len(wrist_points) < 2:
                x, y = int(event.xdata), int(event.ydata)
                wrist_points.append([x, y])
                print(f"Wrist camera - Dot {len(wrist_points)}: ({x}, {y})")

                # Draw marker
                circle = Circle((x, y), 8, color='blue', fill=False, linewidth=3)
                axes[1].add_patch(circle)
                axes[1].plot(x, y, 'b+', markersize=15, markeredgewidth=3)
                fig.canvas.draw()

                if len(wrist_points) == 2:
                    print("[OK] Wrist camera dots identified!")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(first_front)
    axes[0].set_title('Front Camera - CLICK on the TWO dots', fontsize=14, color='red', weight='bold')

    axes[1].imshow(first_wrist)
    axes[1].set_title('Wrist Camera - CLICK on the TWO dots', fontsize=14, color='blue', weight='bold')

    # Connect click events
    fig.canvas.mpl_connect('button_press_event', onclick)

    print("\n" + "=" * 70)
    print("INSTRUCTIONS:")
    print("=" * 70)
    print("1. Click on the FIRST dot in the FRONT camera (left image)")
    print("2. Click on the SECOND dot in the FRONT camera")
    print("3. Click on the FIRST dot in the WRIST camera (right image)")
    print("4. Click on the SECOND dot in the WRIST camera")
    print("5. Close the window when done")
    print("=" * 70)

    plt.tight_layout()
    plt.show()

    # Print results
    if len(front_points) == 2 and len(wrist_points) == 2:
        print("\n" + "=" * 70)
        print("[OK] ALL DOTS IDENTIFIED!")
        print("=" * 70)
        print(f"\nFront camera dots: {front_points}")
        print(f"Wrist camera dots: {wrist_points}")

        print("\n" + "=" * 70)
        print("Run preprocessing with:")
        print("=" * 70)
        print(f"\npython preprocess.py \\")
        print(f"  --output output/processed_data.npz \\")
        print(f"  --dots-cam1 {front_points[0][0]} {front_points[0][1]} {front_points[1][0]} {front_points[1][1]} \\")
        print(f"  --dots-cam2 {wrist_points[0][0]} {wrist_points[0][1]} {wrist_points[1][0]} {wrist_points[1][1]}")
        print("=" * 70)
    else:
        print("\n[WARNING] Not all dots were identified. Please run the script again.")
        if len(front_points) < 2:
            print(f"  Front camera: {len(front_points)}/2 dots")
        if len(wrist_points) < 2:
            print(f"  Wrist camera: {len(wrist_points)}/2 dots")


if __name__ == '__main__':
    # Interactive mode is default (click to find dots)
    # Use --view-only to just display without clicking
    if len(sys.argv) > 1 and sys.argv[1] == '--view-only':
        view_first_frames()
    else:
        interactive_click_tool()
