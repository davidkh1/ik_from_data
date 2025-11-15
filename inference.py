"""
Inference script for using the trained IK model.

This script loads a trained model and predicts joint positions
from image positions (middle points of dots in two camera views).

Usage:
    # Predict from command line
    python inference.py --model checkpoints/best_model.pth --input 320.5 240.3 310.2 235.8

    # Use in Python code
    from inference import IKPredictor
    predictor = IKPredictor('checkpoints/best_model.pth', 'checkpoints/normalization_params.json')
    joint_positions = predictor.predict([320.5, 240.3, 310.2, 235.8])
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json

from model import IKMLP


class IKPredictor:
    """
    Predictor class for inverse kinematics.

    This class wraps the trained model and handles:
    - Model loading
    - Input normalization
    - Output denormalization
    - Prediction
    """

    def __init__(self, model_path, normalization_params_path, device='auto'):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained model checkpoint (.pth file)
            normalization_params_path: Path to normalization parameters (.json file)
            device: Device to run inference on ('auto', 'cpu', 'cuda', 'mps')
        """
        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load normalization parameters
        self.load_normalization_params(normalization_params_path)

        # Load model
        self.model = IKMLP(input_dim=4, output_dim=5, hidden_dims=(256, 128, 64))
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")

    def load_normalization_params(self, params_path):
        """Load normalization parameters from JSON file."""
        with open(params_path, 'r') as f:
            params = json.load(f)

        self.input_mean = torch.FloatTensor(params['input_mean']) if params['input_mean'] else None
        self.input_std = torch.FloatTensor(params['input_std']) if params['input_std'] else None
        self.output_mean = torch.FloatTensor(params['output_mean']) if params['output_mean'] else None
        self.output_std = torch.FloatTensor(params['output_std']) if params['output_std'] else None

        if self.input_mean is not None:
            self.input_mean = self.input_mean.to(self.device)
            self.input_std = self.input_std.to(self.device)
        if self.output_mean is not None:
            self.output_mean = self.output_mean.to(self.device)
            self.output_std = self.output_std.to(self.device)

    def normalize_input(self, input_tensor):
        """Normalize input using saved statistics."""
        if self.input_mean is not None:
            return (input_tensor - self.input_mean) / self.input_std
        return input_tensor

    def denormalize_output(self, output_tensor):
        """Denormalize output using saved statistics."""
        if self.output_mean is not None:
            return output_tensor * self.output_std + self.output_mean
        return output_tensor

    def predict(self, image_positions):
        """
        Predict joint positions from image positions.

        Args:
            image_positions: List or numpy array of shape (4,) or (N, 4)
                            containing [x1, y1, x2, y2] for N samples

        Returns:
            numpy array of shape (5,) or (N, 5) containing predicted joint positions
        """
        # Convert to tensor
        if isinstance(image_positions, list):
            image_positions = np.array(image_positions)

        # Handle single sample vs batch
        single_sample = False
        if image_positions.ndim == 1:
            single_sample = True
            image_positions = image_positions.reshape(1, -1)

        input_tensor = torch.FloatTensor(image_positions).to(self.device)

        # Normalize input
        input_tensor = self.normalize_input(input_tensor)

        # Predict
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Denormalize output
        output_tensor = self.denormalize_output(output_tensor)

        # Convert to numpy
        joint_positions = output_tensor.cpu().numpy()

        # Return single sample if input was single sample
        if single_sample:
            joint_positions = joint_positions[0]

        return joint_positions

    def predict_with_gripper(self, image_positions, gripper_value=50):
        """
        Predict all 6 joint positions including the fixed gripper.

        Args:
            image_positions: List or numpy array of shape (4,) or (N, 4)
            gripper_value: Fixed gripper value (default: 50)

        Returns:
            numpy array of shape (6,) or (N, 6) containing all joint positions
        """
        # Predict first 5 joints
        joints_5 = self.predict(image_positions)

        # Handle single sample vs batch
        if joints_5.ndim == 1:
            # Single sample
            return np.append(joints_5, gripper_value)
        else:
            # Batch
            gripper_column = np.full((len(joints_5), 1), gripper_value)
            return np.concatenate([joints_5, gripper_column], axis=1)


def main():
    parser = argparse.ArgumentParser(description='Run IK inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--norm-params', type=str, default=None,
                        help='Path to normalization parameters (default: same dir as model)')
    parser.add_argument('--input', type=float, nargs=4, required=True,
                        metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='Input image positions: x1 y1 x2 y2')
    parser.add_argument('--gripper', type=float, default=50,
                        help='Fixed gripper value (default: 50)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto, cpu, cuda, mps)')

    args = parser.parse_args()

    # Determine normalization params path
    if args.norm_params is None:
        model_dir = Path(args.model).parent
        args.norm_params = model_dir / 'normalization_params.json'

    # Create predictor
    predictor = IKPredictor(args.model, args.norm_params, device=args.device)

    # Predict
    image_positions = args.input
    joint_positions = predictor.predict_with_gripper(image_positions, gripper_value=args.gripper)

    # Print results
    print(f"\nInput image positions: {image_positions}")
    print(f"Predicted joint positions: {joint_positions}")
    print(f"\nJoint values:")
    joint_names = ['Shoulder Pan', 'Shoulder Lift', 'Elbow Flex', 'Wrist Flex', 'Wrist Roll', 'Gripper']
    for i, (name, value) in enumerate(zip(joint_names, joint_positions)):
        print(f"  Joint {i} ({name}): {value:.4f}")


if __name__ == '__main__':
    main()
