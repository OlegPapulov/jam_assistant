"""
Visualization functions for training metrics and predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import torch

from utils.metrics import visualize_melspectrograms


def visualize_training_metrics(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Visualize training metrics over epochs.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Reconstruction loss
    axes[0, 0].plot(
        epochs,
        [loss['reconstruction_loss'] for loss in history['train_losses']],
        label='Train',
        marker='o'
    )
    axes[0, 0].plot(
        epochs,
        [loss['reconstruction_loss'] for loss in history['val_losses']],
        label='Val',
        marker='s'
    )
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Reconstruction Loss')
    axes[0, 0].set_title('Reconstruction Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Bass loss
    axes[0, 1].plot(
        epochs,
        [loss['reconstruction_loss_bass'] for loss in history['train_losses']],
        label='Train',
        marker='o'
    )
    axes[0, 1].plot(
        epochs,
        [loss['reconstruction_loss_bass'] for loss in history['val_losses']],
        label='Val',
        marker='s'
    )
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Bass Loss')
    axes[0, 1].set_title('Bass Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Drums loss
    axes[1, 0].plot(
        epochs,
        [loss['reconstruction_loss_drums'] for loss in history['train_losses']],
        label='Train',
        marker='o'
    )
    axes[1, 0].plot(
        epochs,
        [loss['reconstruction_loss_drums'] for loss in history['val_losses']],
        label='Val',
        marker='s'
    )
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Drums Loss')
    axes[1, 0].set_title('Drums Reconstruction Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Spectral centroid loss
    axes[1, 1].plot(
        epochs,
        [loss['spectral_centroid_loss'] for loss in history['val_losses']],
        label='Val',
        marker='s'
    )
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Spectral Centroid Loss')
    axes[1, 1].set_title('Spectral Centroid Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_predictions(
    target_melspec: torch.Tensor,
    pred_melspec: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Visualize target and predicted melspectrograms.
    
    Args:
        target_melspec: Target melspectrogram
        pred_melspec: Predicted melspectrogram
        save_path: Path to save figure
    """
    visualize_melspectrograms(target_melspec, pred_melspec, save_path=save_path)


def create_evaluation_report(
    history: Dict[str, List],
    metrics: Dict[str, float],
    save_path: Optional[str] = None
):
    """
    Create comprehensive evaluation report.
    
    Args:
        history: Training history
        metrics: Final metrics dictionary
        save_path: Path to save report
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    axes.axis('off')
    
    report_text = "Training Report\n"
    report_text += "=" * 50 + "\n\n"
    
    # Final losses
    if history['val_losses']:
        final_losses = history['val_losses'][-1]
        report_text += "Final Validation Losses:\n"
        for key, value in final_losses.items():
            report_text += f"  {key}: {value:.4f}\n"
        report_text += "\n"
    
    # Final metrics
    if metrics:
        report_text += "Final Metrics:\n"
        for key, value in metrics.items():
            report_text += f"  {key}: {value:.4f}\n"
    
    axes.text(0.1, 0.5, report_text, fontsize=12, verticalalignment='center',
              fontfamily='monospace')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

