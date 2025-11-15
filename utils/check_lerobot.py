"""
Check if this is a LeRobot dataset that needs the lerobot library.
"""

from pathlib import Path
import os

# Try to import lerobot to check if it's installed
try:
    import lerobot
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    lerobot = None

# Check HuggingFace cache
cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
print(f"HuggingFace cache: {cache_dir}")
print(f"Exists: {cache_dir.exists()}")

if cache_dir.exists():
    print("\nSearching for paszea/ik dataset...")

    # Find the dataset
    for root, dirs, files in os.walk(cache_dir):
        if 'paszea' in root.lower() or 'ik' in Path(root).name:
            print(f"\nFound: {root}")
            # List first level contents
            root_path = Path(root)
            for item in sorted(root_path.iterdir())[:10]:
                if item.is_file():
                    size_mb = item.stat().st_size / (1024 * 1024)
                    print(f"  {item.name} ({size_mb:.2f} MB)")
                else:
                    print(f"  {item.name}/ (directory)")

# Check if lerobot is needed
print("\n" + "=" * 70)
print("Checking LeRobot library:")
print("=" * 70)

if LEROBOT_AVAILABLE:
    print("[OK] lerobot is installed")
    print(f"Version: {lerobot.__version__}")
else:
    print("[X] lerobot is NOT installed")
    print("\nThis dataset might require the lerobot library.")
    print("Install with: pip install lerobot")

print("\n" + "=" * 70)
print("Alternative: Download videos manually")
print("=" * 70)
print("The videos might be in a separate Git LFS repository.")
print("You can download them using:")
print("  git clone https://huggingface.co/datasets/paszea/ik data/ik_dataset")
