"""
Training script for the inverse kinematics MLP model.

This script trains the model to map from image positions [x1, y1, x2, y2]
to joint positions (5 joints, excluding the fixed gripper).

Usage:
    python train.py --data data/processed_data.npz --epochs 100
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run 'pip install wandb' to enable experiment tracking.")

from model import IKMLP
from dataset import IKDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: The IK model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """
    Validate the model.

    Args:
        model: The IK model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def plot_training_history(train_losses, val_losses, save_path):
    """Plot and save training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train IK model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to preprocessed data (.npz file)')
    parser.add_argument('--output-dir', type=str, default='output/checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to train on (auto, cpu, cuda, mps)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='ik-from-data',
                        help='W&B project name (default: ik-from-data)')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='W&B run name (default: auto-generated)')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Initialize Weights & Biases
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: --wandb flag provided but wandb is not installed. Continuing without wandb.")

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'val_split': args.val_split,
                'device': str(device),
                'seed': args.seed,
                'model': 'IKMLP',
                'optimizer': 'Adam',
                'scheduler': 'ReduceLROnPlateau',
            }
        )
        print(f"Weights & Biases initialized: {wandb.run.url}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    dataset = IKDataset(data_path=args.data, normalize_inputs=True, normalize_outputs=True)
    print(f"Dataset size: {len(dataset)}")

    # Save normalization parameters
    norm_params_path = output_dir / 'normalization_params.json'
    dataset.save_normalization_params(norm_params_path)
    print(f"[SAVED] {norm_params_path}")

    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility, increase if needed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create model
    model = IKMLP(input_dim=4, output_dim=5, hidden_dims=(256, 128, 64))
    model = model.to(device)
    print(f"\nModel architecture:\n{model}")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
            })

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

    # Save final model
    final_model_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, final_model_path)
    print(f"\n[SAVED] Models -> {output_dir}/")

    # Plot training history
    plot_path = output_dir / 'training_history.png'
    plot_training_history(train_losses, val_losses, plot_path)

    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
        }, f, indent=2)

    # Log artifacts and summary to wandb
    if use_wandb:
        # Log summary metrics
        wandb.summary['best_val_loss'] = best_val_loss
        wandb.summary['final_train_loss'] = train_losses[-1]
        wandb.summary['final_val_loss'] = val_losses[-1]

        # Log training plot
        wandb.log({'training_history': wandb.Image(str(plot_path))})

        # Save model artifacts
        artifact = wandb.Artifact('ik-model', type='model')
        artifact.add_file(str(best_model_path), name='best_model.pth')
        artifact.add_file(str(final_model_path), name='final_model.pth')
        artifact.add_file(str(norm_params_path), name='normalization_params.json')
        wandb.log_artifact(artifact)

        # Finish wandb run
        wandb.finish()
        print("Weights & Biases logging complete")

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
