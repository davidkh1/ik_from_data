"""Check dataset info and available data."""
from datasets import load_dataset_builder

print("Getting dataset info...")
ds_builder = load_dataset_builder('paszea/ik')

print("\n" + "=" * 70)
print("Dataset Info:")
print("=" * 70)
print(ds_builder.info)

print("\n" + "=" * 70)
print("Dataset Description:")
print("=" * 70)
print(ds_builder.info.description)

print("\n" + "=" * 70)
print("Available Splits:")
print("=" * 70)
print(ds_builder.info.splits)

print("\n" + "=" * 70)
print("Features:")
print("=" * 70)
print(ds_builder.info.features)

print("\n" + "=" * 70)
print("Dataset Config:")
print("=" * 70)
print(f"Config name: {ds_builder.config.name}")
print(f"Available configs: {ds_builder.BUILDER_CONFIGS}")
