"""
Dataset class for loading IK training data from HuggingFace.

The dataset loads from https://huggingface.co/datasets/paszea/ik
and processes it to extract:
- Image positions: middle point of two dots in two camera views [x1, y1, x2, y2]
- Joint positions: 5 joint angles (excluding the fixed gripper joint)

The data includes a 6-frame offset to compensate for lag between arm movement and action.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import json
from pathlib import Path


class IKDataset(Dataset):
    """
    Dataset for inverse kinematics training from HuggingFace dataset.

    Expects either:
    1. Pre-processed dot positions (from co-tracker)
    2. Raw data arrays for image positions and joint positions
    """

    def __init__(
        self,
        image_positions=None,
        joint_positions=None,
        data_path=None,
        normalize_inputs=True,
        normalize_outputs=True,
    ):
        """
        Initialize the IK dataset.

        Args:
            image_positions: numpy array of shape (N, 4) containing [x1, y1, x2, y2]
            joint_positions: numpy array of shape (N, 5) containing joint angles
            data_path: Path to preprocessed data file (.npz)
            normalize_inputs: Whether to normalize image positions
            normalize_outputs: Whether to normalize joint positions
        """
        if data_path is not None:
            image_positions, joint_positions = self._load_from_file(data_path)

        if image_positions is None or joint_positions is None:
            raise ValueError("Must provide either data arrays or data_path")

        self.image_positions = torch.FloatTensor(image_positions)
        self.joint_positions = torch.FloatTensor(joint_positions)

        # Compute normalization statistics
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

        if normalize_inputs:
            self.input_mean = self.image_positions.mean(dim=0)
            self.input_std = self.image_positions.std(dim=0) + 1e-8
            self.image_positions = (self.image_positions - self.input_mean) / self.input_std

        if normalize_outputs:
            self.output_mean = self.joint_positions.mean(dim=0)
            self.output_std = self.joint_positions.std(dim=0) + 1e-8
            self.joint_positions = (self.joint_positions - self.output_mean) / self.output_std

        assert len(self.image_positions) == len(self.joint_positions), \
            "Image positions and joint positions must have same length"

    def _load_from_file(self, data_path):
        """Load preprocessed data from file."""
        data_path = Path(data_path)

        if data_path.suffix == '.npz':
            data = np.load(data_path)
            image_positions = data['image_positions']
            joint_positions = data['joint_positions']
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        return image_positions, joint_positions

    def __len__(self):
        return len(self.image_positions)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            tuple: (image_positions, joint_positions)
        """
        return self.image_positions[idx], self.joint_positions[idx]

    def get_normalization_params(self):
        """
        Get normalization parameters for saving/loading.

        Returns:
            dict: Dictionary containing mean and std for inputs and outputs
        """
        return {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std,
        }

    def denormalize_output(self, normalized_output):
        """
        Convert normalized output back to original scale.

        Args:
            normalized_output: torch.Tensor of normalized joint positions

        Returns:
            torch.Tensor of denormalized joint positions
        """
        if self.output_mean is None or self.output_std is None:
            return normalized_output
        return normalized_output * self.output_std + self.output_mean

    def save_normalization_params(self, save_path):
        """Save normalization parameters to file."""
        params = self.get_normalization_params()
        # Convert tensors to lists for JSON serialization
        params_serializable = {
            k: v.tolist() if v is not None else None
            for k, v in params.items()
        }
        with open(save_path, 'w') as f:
            json.dump(params_serializable, f, indent=2)

    @staticmethod
    def load_normalization_params(load_path):
        """Load normalization parameters from file."""
        with open(load_path, 'r') as f:
            params = json.load(f)
        # Convert lists back to tensors
        return {
            k: torch.FloatTensor(v) if v is not None else None
            for k, v in params.items()
        }


def load_hf_dataset(split='train'):
    """
    Load the raw HuggingFace dataset.

    Args:
        split: Dataset split to load ('train', 'test', etc.)

    Returns:
        HuggingFace dataset object
    """
    dataset = load_dataset("paszea/ik", split=split)
    return dataset


def extract_joint_positions(dataset, frame_offset=6):
    """
    Extract joint positions from HuggingFace dataset.

    Args:
        dataset: HuggingFace dataset
        frame_offset: Number of frames to offset (default 6 for 0.2s lag at 30fps)

    Returns:
        numpy array of shape (N, 5) with first 5 joint positions
    """
    # Get observation states (actual joint positions)
    all_states = []

    for i in range(len(dataset) - frame_offset):
        state = dataset[i]['observation.state']
        # Take first 5 joints (ignore gripper which is fixed at 50)
        all_states.append(state[:5])

    return np.array(all_states)


if __name__ == '__main__':
    print("Loading HuggingFace dataset...")
    hf_dataset = load_hf_dataset()

    print(f"Dataset size: {len(hf_dataset)}")
    print(f"Dataset features: {hf_dataset.features}")

    # Show sample data
    sample = hf_dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Action shape: {np.array(sample['action']).shape}")
    print(f"State shape: {np.array(sample['observation.state']).shape}")
    print(f"Sample action: {sample['action']}")
    print(f"Sample state: {sample['observation.state']}")
