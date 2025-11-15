"""
Load the dataset properly with video access.
"""

from datasets import load_dataset
from pathlib import Path

print("=" * 70)
print("Loading paszea/ik dataset...")
print("=" * 70)

# Load dataset - this should download everything including videos
print("\nLoading dataset (this will download videos on first run, ~1.75 GB)...")
ds = load_dataset("paszea/ik")

print(f"\n[OK] Dataset loaded!")
print(f"Available splits: {list(ds.keys())}")

# Access train split
train_ds = ds['train']
print(f"\nTraining samples: {len(train_ds)}")

print("\n" + "=" * 70)
print("Dataset Features:")
print("=" * 70)
print(train_ds.features)

print("\n" + "=" * 70)
print("First sample:")
print("=" * 70)
sample = train_ds[0]

for key in sample.keys():
    value = sample[key]
    print(f"\n{key}: {type(value).__name__}", end="")

    if isinstance(value, list):
        print(f" (length: {len(value)})")
        print(f"  Value: {value}")
    elif hasattr(value, 'size'):
        print(f" (size: {value.size})")
    else:
        print(f" = {value}")

# Check if videos are accessible
print("\n" + "=" * 70)
print("Checking video access:")
print("=" * 70)

# Try to find video-related keys
video_keys = [k for k in sample.keys() if 'video' in k.lower() or 'image' in k.lower()]
if video_keys:
    print(f"Found video/image keys: {video_keys}")
    for key in video_keys:
        print(f"\n{key}:")
        print(f"  Type: {type(sample[key])}")
        print(f"  Value: {sample[key]}")
else:
    print("No video/image keys found in the sample.")
    print("\nThis might be a LeRobot dataset where videos are stored separately.")
    print("Checking cache location...")

    # Check HuggingFace cache
    from huggingface_hub import HfFolder
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    print(f"\nHuggingFace cache: {cache_dir}")

    # Look for paszea/ik in cache
    ik_cache = list(cache_dir.glob("**/paszea*ik*"))
    if ik_cache:
        print(f"\nFound dataset cache at:")
        for path in ik_cache:
            print(f"  {path}")
            # List contents
            if path.is_dir():
                for item in sorted(path.rglob("*"))[:20]:  # First 20 items
                    if item.is_file():
                        size_mb = item.stat().st_size / (1024 * 1024)
                        print(f"    {item.relative_to(path)} ({size_mb:.2f} MB)")
