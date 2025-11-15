"""
Simple script to download and explore the paszea/ik dataset from HuggingFace.
No visualization - just data exploration.
"""

import numpy as np
from datasets import load_dataset


def load_and_explore():
    """Load and explore the dataset."""
    print("=" * 70)
    print("Loading HuggingFace dataset: paszea/ik")
    print("=" * 70)

    # Load dataset
    print("\nDownloading dataset (this may take a while on first run)...")
    dataset = load_dataset("paszea/ik", split='train')

    print(f"\n[OK] Dataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")

    # Show dataset features
    print("\n" + "=" * 70)
    print("Dataset Features")
    print("=" * 70)
    print(dataset.features)

    # Show sample data
    print("\n" + "=" * 70)
    print("Sample Data (First Row)")
    print("=" * 70)
    sample = dataset[0]

    for key in sample.keys():
        value = sample[key]
        if isinstance(value, (list, np.ndarray)):
            print(f"\n{key}:")
            print(f"  Type: {type(value)}")
            print(f"  Shape/Length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
            print(f"  Value: {value}")
        else:
            print(f"\n{key}: {value} (type: {type(value).__name__})")

    # Statistics on episodes
    print("\n" + "=" * 70)
    print("Episode Statistics")
    print("=" * 70)
    episode_indices = [sample['episode_index'] for sample in dataset]
    unique_episodes = set(episode_indices)
    print(f"Number of episodes: {len(unique_episodes)}")
    print(f"Episode indices: {sorted(unique_episodes)}")

    for ep_idx in sorted(unique_episodes):
        frames_in_episode = sum(1 for idx in episode_indices if idx == ep_idx)
        print(f"  Episode {ep_idx}: {frames_in_episode} frames")

    # Joint position statistics
    print("\n" + "=" * 70)
    print("Joint Position Statistics (observation.state)")
    print("=" * 70)

    all_states = np.array([sample['observation.state'] for sample in dataset])
    print(f"Shape: {all_states.shape}")
    print(f"Number of joints: {all_states.shape[1]}")

    joint_names = ['Shoulder Pan', 'Shoulder Lift', 'Elbow Flex',
                   'Wrist Flex', 'Wrist Roll', 'Gripper']

    for i in range(all_states.shape[1]):
        joint_values = all_states[:, i]
        print(f"\nJoint {i} ({joint_names[i]}):")
        print(f"  Min:  {joint_values.min():.4f}")
        print(f"  Max:  {joint_values.max():.4f}")
        print(f"  Mean: {joint_values.mean():.4f}")
        print(f"  Std:  {joint_values.std():.4f}")

    # Action statistics
    print("\n" + "=" * 70)
    print("Action Statistics")
    print("=" * 70)

    all_actions = np.array([sample['action'] for sample in dataset])
    print(f"Shape: {all_actions.shape}")

    for i in range(all_actions.shape[1]):
        action_values = all_actions[:, i]
        print(f"\nAction {i} ({joint_names[i]}):")
        print(f"  Min:  {action_values.min():.4f}")
        print(f"  Max:  {action_values.max():.4f}")
        print(f"  Mean: {action_values.mean():.4f}")
        print(f"  Std:  {action_values.std():.4f}")

    # Timestamp info
    print("\n" + "=" * 70)
    print("Timestamp Statistics")
    print("=" * 70)
    timestamps = np.array([sample['timestamp'] for sample in dataset])
    print(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
    print(f"Estimated FPS: {len(dataset) / (timestamps[-1] - timestamps[0]):.2f}")

    # Sample some frames
    print("\n" + "=" * 70)
    print("Sample Frames")
    print("=" * 70)

    indices = [0, len(dataset) // 2, len(dataset) - 1]
    for idx in indices:
        sample = dataset[idx]
        print(f"\nFrame {idx}:")
        print(f"  Episode: {sample['episode_index']}, Frame: {sample['frame_index']}")
        print(f"  Timestamp: {sample['timestamp']:.3f}s")
        print(f"  State: {sample['observation.state']}")
        print(f"  Action: {sample['action']}")

    return dataset


if __name__ == '__main__':
    dataset = load_and_explore()

    print("\n" + "=" * 70)
    print("Exploration Complete!")
    print("=" * 70)
    print("\nDataset Summary:")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Episodes: 2")
    print(f"  Cameras: front, wrist")
    print(f"  Joints: 6 (first 5 for IK, last one fixed at ~50)")
    print("\nNext steps:")
    print("1. Use co-tracker to extract dot positions from the video frames")
    print("2. Train the IK model with the preprocessed data")
