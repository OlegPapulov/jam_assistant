#!/usr/bin/env python3
"""
Main training script for LSTM audio generation model.

This script orchestrates the entire training pipeline:
1. Set random seed
2. Load configuration
3. Prepare data
4. Create model
5. Train model
6. Evaluate model
7. Visualize results
"""

import os
import sys
import torch
import torch.optim as optim
import torchaudio.transforms as T

# Set random seed BEFORE any other imports that might use randomness
# This must be done very early
from utils.seed import set_seed

# Now import other modules
from config.config import Config, parse_args
from data.dataloader import create_dataloaders
from models.lstm_model import LSTMModel
from training.trainer import train
from utils.metrics import (
    spectral_centroid_fn,
    spectral_bandwidth_fn,
    spectral_flatness_fn,
    spectral_entropy_fn,
)
from visualization import visualize_training_metrics


def main():
    """Main training function."""
    # Parse command-line arguments
    config = parse_args()
    
    # Set random seed from config (must be done early)
    set_seed(config.random_seed)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize wandb if enabled
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                config=vars(config),
            )
            wandb_run = wandb.run
            print("Wandb initialized")
        except ImportError:
            print("Warning: wandb not installed, disabling wandb logging")
            config.use_wandb = False
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            config.use_wandb = False
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config, device=device)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = LSTMModel(
        mel_bins=config.mel_bins,
        sequence_length_input=config.sequence_length_input,
        sequence_length_output=config.sequence_length_output,
        dropout_rate=config.dropout_rate,
        use_bidirectional=config.use_bidirectional,
        device=device
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    
    # Define loss functions
    loss_functions = {
        'spectral_centroid': spectral_centroid_fn,
        'spectral_bandwidth': spectral_bandwidth_fn,
        'spectral_flatness': spectral_flatness_fn,
        'spectral_entropy': spectral_entropy_fn,
    }
    
    # Define metric functions (same as loss functions for now)
    metric_functions = {
        'spectral_centroid': spectral_centroid_fn,
        'spectral_bandwidth': spectral_bandwidth_fn,
        'spectral_flatness': spectral_flatness_fn,
        'spectral_entropy': spectral_entropy_fn,
    }
    
    # Train model
    print("Starting training...")
    history = train(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_functions=loss_functions,
        metric_functions=metric_functions,
        device=device,
        save_dir='./checkpoints',
        log_dir='./logs',
        use_wandb=config.use_wandb,
        wandb_run=wandb_run
    )
    
    # Visualize training metrics
    print("Generating visualizations...")
    visualize_training_metrics(history, save_path='./logs/training_metrics.png')
    
    # Save final model
    final_model_path = './checkpoints/final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Finish wandb run
    if wandb_run:
        wandb_run.finish()
    
    print("Training completed!")


if __name__ == '__main__':
    main()

