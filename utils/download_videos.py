"""
Download video files from the paszea/ik dataset.

The videos are stored separately from the parquet data files.
"""

from huggingface_hub import snapshot_download
from pathlib import Path

def download_dataset_videos():
    """Download the complete dataset including videos."""
    print("=" * 70)
    print("Downloading paszea/ik dataset with videos...")
    print("=" * 70)

    # Download to data/ik_dataset directory
    dataset_path = Path("data/ik_dataset")

    print(f"\nDownloading to: {dataset_path.absolute()}")
    print("This may take a while (1.75 GB)...\n")

    # Download the entire dataset repository
    local_dir = snapshot_download(
        repo_id="paszea/ik",
        repo_type="dataset",
        local_dir=str(dataset_path),
        local_dir_use_symlinks=False,
    )

    print(f"\n[OK] Dataset downloaded to: {local_dir}")

    # List contents
    print("\n" + "=" * 70)
    print("Downloaded files:")
    print("=" * 70)

    for item in sorted(Path(local_dir).rglob("*")):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.relative_to(local_dir)} ({size_mb:.2f} MB)")

    # Check for video files
    video_dir = Path(local_dir) / "videos"
    if video_dir.exists():
        video_files = list(video_dir.glob("*.mp4"))
        print(f"\n[OK] Found {len(video_files)} video files in videos/")
        for vf in sorted(video_files)[:5]:  # Show first 5
            print(f"  - {vf.name}")
        if len(video_files) > 5:
            print(f"  ... and {len(video_files) - 5} more")

    return local_dir


if __name__ == '__main__':
    dataset_path = download_dataset_videos()

    print("\n" + "=" * 70)
    print("Next steps:")
    print("=" * 70)
    print("1. The videos are now available in data/ik_dataset/videos/")
    print("2. You can extract frames and run co-tracker on them")
    print("3. Use the updated preprocessing script")
