#!/usr/bin/env python3
"""
Evaluation script for trained model.

Loads a trained model and evaluates it on the test set.
"""

import argparse
import torch
import torchaudio.transforms as T

from config.config import Config
from data.dataloader import create_dataloaders
from models.lstm_model import LSTMModel
from training.evaluator import evaluate
from utils.metrics import (
    spectral_centroid_fn,
    spectral_bandwidth_fn,
    spectral_flatness_fn,
    spectral_entropy_fn,
)
from utils.audio import restore_audio_from_melspec, compare_audio
from visualization import visualize_predictions, create_evaluation_report


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--save_audio', action='store_true',
                       help='Save audio samples from predictions')
    parser.add_argument('--audio_samples', type=int, default=3,
                       help='Number of audio samples to save')
    parser.add_argument('--audio_dir', type=str, default='./audio_samples',
                       help='Directory to save audio samples')
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get config from checkpoint or create new one
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = Config()  # Use defaults
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    _, test_loader = create_dataloaders(config, device=device)
    
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
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    
    # Create melspectrogram transform
    melspec = T.MelSpectrogram(
        config.target_sample_rate,
        n_fft=config.n_fft,
        n_mels=config.mel_bins,
        hop_length=config.hop_length
    ).to(device)
    
    # Define loss and metric functions
    loss_functions = {
        'spectral_centroid': spectral_centroid_fn,
        'spectral_bandwidth': spectral_bandwidth_fn,
        'spectral_flatness': spectral_flatness_fn,
        'spectral_entropy': spectral_entropy_fn,
    }
    
    metric_functions = {
        'spectral_centroid': spectral_centroid_fn,
        'spectral_bandwidth': spectral_bandwidth_fn,
        'spectral_flatness': spectral_flatness_fn,
        'spectral_entropy': spectral_entropy_fn,
    }
    
    # Evaluate
    print("Evaluating model...")
    losses, metrics = evaluate(
        model=model,
        val_loader=test_loader,
        melspec_transform=melspec,
        config=config,
        loss_functions=loss_functions,
        metric_functions=metric_functions,
        device=device
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    print("Losses:")
    for key, value in losses.items():
        print(f"  {key}: {value:.4f}")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Create evaluation report
    history = {
        'val_losses': [losses],
        'val_metrics': [metrics],
    }
    create_evaluation_report(history, metrics, save_path='./logs/evaluation_report.png')
    
    # Save audio samples if requested
    if args.save_audio:
        print(f"\nSaving {args.audio_samples} audio samples...")
        import os
        os.makedirs(args.audio_dir, exist_ok=True)
        
        model.eval()
        sample_count = 0
        with torch.no_grad():
            for batch in test_loader:
                if sample_count >= args.audio_samples:
                    break
                
                vocals_X, vocals_y = batch['vocals']
                samples_in_batch = vocals_X.shape[0]
                
                # Take first sample from batch
                stems_y = vocals_y[0:1].to(device)  # [1, time]
                
                # Compute melspectrogram
                melspecs_y = melspec(stems_y)
                
                # Apply log transformation if needed
                if config.target_log:
                    melspecs_X_log = torch.log1p(melspecs_y)
                else:
                    melspecs_X_log = melspecs_y
                
                # Forward pass
                melspecs_pred, _, _ = model(melspecs_X_log, batch_size=1)
                
                # Convert back from log scale if needed
                if config.target_log:
                    melspecs_pred = torch.expm1(melspecs_pred)
                
                # Restore audio from melspectrograms
                target_audio, pred_audio = compare_audio(
                    target_melspec=melspecs_y[0],  # Remove batch dimension
                    pred_melspec=melspecs_pred[0],  # Remove batch dimension
                    config=config,
                    save_dir=args.audio_dir,
                    prefix=f'sample_{sample_count}',
                    device=device
                )
                
                sample_count += 1
        
        print(f"Saved {sample_count} audio samples to {args.audio_dir}")
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()

