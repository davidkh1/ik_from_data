"""
Simple MLP model for inverse kinematics of SO-100 robot arm.

The model maps from middle-point positions in two camera images to joint positions.
Input: [x1, y1, x2, y2] - middle point of two dots in both camera images
Output: 5 joint positions (joints 0-4, gripper joint 5 is fixed at 50)
"""

import torch
import torch.nn as nn


class IKMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for inverse kinematics.

    Architecture:
    - Input layer: 4 neurons (x1, y1, x2, y2)
    - Hidden layer 1: 256 neurons
    - Hidden layer 2: 128 neurons
    - Hidden layer 3: 64 neurons
    - Output layer: 5 neurons (joint positions 0-4)
    """

    def __init__(self, input_dim=4, output_dim=5, hidden_dims=(256, 128, 64)):
        super(IKMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Build the network layers
        layers = []
        prev_dim = input_dim

        # Hidden layers with ReLU activation
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (no activation - regression task)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: torch.Tensor of shape (batch_size, 4) containing [x1, y1, x2, y2]

        Returns:
            torch.Tensor of shape (batch_size, 5) containing predicted joint positions
        """
        return self.network(x)


def create_model(device='cpu'):
    """
    Create and initialize the IK model.

    Args:
        device: Device to place the model on ('cpu' or 'cuda')

    Returns:
        IKMLP model instance
    """
    model = IKMLP()
    model = model.to(device)
    return model


if __name__ == '__main__':
    # Simple test
    model = create_model()
    print(f"Model architecture:\n{model}")

    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 4)
    output = model(test_input)

    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nSample input: {test_input[0]}")
    print(f"Sample output: {output[0]}")
