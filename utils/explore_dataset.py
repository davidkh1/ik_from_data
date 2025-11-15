"""
Script to download and explore the paszea/ik dataset from HuggingFace.

This script will:
1. Download the dataset
2. Show basic statistics
3. Display sample data
4. Visualize frames from both cameras
"""

import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image


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
        elif isinstance(value, Image.Image):
            print(f"\n{key}:")
            print(f"  Type: PIL Image")
            print(f"  Size: {value.size}")
            print(f"  Mode: {value.mode}")
        else:
            print(f"\n{key}: {value}")

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

    return dataset


def visualize_samples(dataset, num_samples=3):
    """Visualize sample frames from the dataset."""
    print("\n" + "=" * 70)
    print("Visualizing Sample Frames")
    print("=" * 70)

    # Select evenly spaced samples
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]

        # Front camera
        front_img = sample['observation.images.front']
        axes[i, 0].imshow(front_img)
        axes[i, 0].set_title(f'Frame {idx} - Front Camera')
        axes[i, 0].axis('off')

        # Wrist camera
        wrist_img = sample['observation.images.wrist']
        axes[i, 1].imshow(wrist_img)
        axes[i, 1].set_title(f'Frame {idx} - Wrist Camera')
        axes[i, 1].axis('off')

        # Print joint positions
        state = sample['observation.state']
        print(f"\nFrame {idx}:")
        print(f"  State: {state}")
        print(f"  Episode: {sample['episode_index']}, Frame: {sample['frame_index']}")

    plt.tight_layout()
    output_path = Path('output/dataset_samples.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Sample frames saved to: output/dataset_samples.png")
    plt.show()


def plot_joint_trajectories(dataset):
    """Plot joint position trajectories over time."""
    print("\n" + "=" * 70)
    print("Plotting Joint Trajectories")
    print("=" * 70)

    # Get all states
    all_states = np.array([sample['observation.state'] for sample in dataset])
    timestamps = np.array([sample['timestamp'] for sample in dataset])

    # Create plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    joint_names = ['Shoulder Pan', 'Shoulder Lift', 'Elbow Flex',
                   'Wrist Flex', 'Wrist Roll', 'Gripper']

    for i in range(6):
        axes[i].plot(timestamps, all_states[:, i], linewidth=0.5, alpha=0.7)
        axes[i].set_title(f'Joint {i}: {joint_names[i]}')
        axes[i].set_xlabel('Timestamp (s)')
        axes[i].set_ylabel('Position')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path('output/joint_trajectories.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Joint trajectories saved to: output/joint_trajectories.png")
    plt.show()


if __name__ == '__main__':
    # Load and explore dataset
    dataset = load_and_explore()

    # Visualize samples
    print("\n" + "=" * 70)
    print("Do you want to visualize sample frames? (requires display)")
    print("=" * 70)

    try:
        visualize_samples(dataset, num_samples=3)
        plot_joint_trajectories(dataset)
    except Exception as e:
        print(f"Note: Visualization skipped (error: {e})")
        print("This is normal if running without display or if matplotlib backend is not configured")

    print("\n" + "=" * 70)
    print("Exploration Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run preprocessing to extract dot positions (requires co-tracker)")
    print("2. Or provide pre-processed data to train the model")
