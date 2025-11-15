"""Quick script to check dataset keys."""
from datasets import load_dataset

print("Loading dataset...")
ds = load_dataset('paszea/ik', split='train')
print(f"\nDataset size: {len(ds)}")

print("\n" + "=" * 70)
print("Dataset Features:")
print("=" * 70)
print(ds.features)

print("\n" + "=" * 70)
print("First sample keys:")
print("=" * 70)
for key in ds[0].keys():
    print(f"  - {key}")

print("\n" + "=" * 70)
print("First sample values:")
print("=" * 70)
sample = ds[0]
for key, value in sample.items():
    value_type = type(value).__name__
    if hasattr(value, 'size'):
        value_str = f"size: {value.size}"
    elif hasattr(value, '__len__'):
        value_str = f"length: {len(value)}"
    else:
        value_str = str(value)
    print(f"\n{key}: {value_type} - {value_str}")

# Check if we need to load with specific columns
print("\n" + "=" * 70)
print("Checking column names:")
print("=" * 70)
print(f"Column names: {ds.column_names}")
