"""
Example usage of the IK predictor.

This script demonstrates how to use the trained model for inference.
"""

import traceback
import numpy as np
from inference import IKPredictor


def example_single_prediction():
    """Example of predicting a single position."""
    print("=" * 60)
    print("Example 1: Single Prediction")
    print("=" * 60)

    # Initialize the predictor
    predictor = IKPredictor(
        model_path='output/checkpoints/best_model.pth',
        normalization_params_path='output/checkpoints/normalization_params.json',
        device='auto'
    )

    # Example image positions: middle point of dots in two camera views
    # [x1, y1] from camera 1, [x2, y2] from camera 2
    image_positions = [320.5, 240.3, 310.2, 235.8]

    # Predict joint positions (first 5 joints only)
    joint_positions = predictor.predict(image_positions)
    print(f"\nInput: {image_positions}")
    print(f"Predicted joints (5): {joint_positions}")

    # Predict with gripper included (all 6 joints)
    full_joint_positions = predictor.predict_with_gripper(image_positions, gripper_value=50)
    print(f"Predicted joints (6): {full_joint_positions}")

    # Display joint names
    joint_names = [
        'Shoulder Pan',
        'Shoulder Lift',
        'Elbow Flex',
        'Wrist Flex',
        'Wrist Roll',
        'Gripper'
    ]
    print("\nJoint Details:")
    for i, (name, value) in enumerate(zip(joint_names, full_joint_positions)):
        print(f"  Joint {i} ({name:15s}): {value:8.4f}")


def example_batch_prediction():
    """Example of predicting multiple positions at once."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Prediction")
    print("=" * 60)

    # Initialize the predictor
    predictor = IKPredictor(
        model_path='output/checkpoints/best_model.pth',
        normalization_params_path='output/checkpoints/normalization_params.json',
        device='auto'
    )

    # Multiple image positions (batch of 3)
    image_positions = np.array([
        [320.5, 240.3, 310.2, 235.8],
        [350.2, 260.1, 340.5, 255.3],
        [280.8, 220.5, 275.3, 215.9]
    ])

    # Predict for all positions
    joint_positions = predictor.predict(image_positions)

    print(f"\nInput shape: {image_positions.shape}")
    print(f"Output shape: {joint_positions.shape}")

    for i, (img_pos, joint_pos) in enumerate(zip(image_positions, joint_positions)):
        print(f"\nSample {i + 1}:")
        print(f"  Image positions: {img_pos}")
        print(f"  Joint positions: {joint_pos}")


def example_control_loop():
    """Example of using the predictor in a control loop."""
    print("\n" + "=" * 60)
    print("Example 3: Simulated Control Loop")
    print("=" * 60)

    # Initialize the predictor
    predictor = IKPredictor(
        model_path='output/checkpoints/best_model.pth',
        normalization_params_path='output/checkpoints/normalization_params.json',
        device='auto'
    )

    # Simulate a sequence of target positions
    target_positions = [
        [320.0, 240.0, 310.0, 235.0],
        [330.0, 250.0, 320.0, 245.0],
        [340.0, 260.0, 330.0, 255.0],
    ]

    print("\nSimulating control loop with 3 target positions...")
    for i, target in enumerate(target_positions):
        # Predict joint positions
        joints = predictor.predict_with_gripper(target)

        print(f"\nStep {i + 1}:")
        print(f"  Target image pos: {target}")
        print(f"  Commanded joints: {joints}")
        # In a real system, you would send these joint commands to the robot
        # robot.move_to_joint_positions(joints)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("IK Predictor Examples")
    print("=" * 60)
    print("\nNote: Make sure you have trained a model first!")
    print("Expected files:")
    print("  - checkpoints/best_model.pth")
    print("  - checkpoints/normalization_params.json")
    print()

    try:
        # Run examples
        example_single_prediction()
        example_batch_prediction()
        example_control_loop()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train the model first:")
        print("  python train.py --data output/processed_data.npz --epochs 100")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
