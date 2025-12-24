"""
Metrics module for audio/melspectrogram evaluation.

This module contains spectral metrics functions that operate on linear-scale melspectrograms
(before log transformation). All spectral metric functions expect melspectrograms in linear scale.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def spectral_centroid_fn(mel_spec: torch.Tensor) -> torch.Tensor:
    """
    Calculate spectral centroid.
    
    Spectral centroid represents the "brightness" of the sound - which frequencies
    dominate (high or low frequencies).
    
    Args:
        mel_spec: Melspectrogram tensor of shape [batch, mel_bins, time].
                  Expected to be in LINEAR scale (before log transformation).
    
    Returns:
        Spectral centroid tensor of shape [batch, 1, time]
    
    Note:
        This function expects melspectrograms in LINEAR scale (before log transformation).
        If your melspectrograms are log-transformed, convert them back using expm1 before calling this function.
    """
    mel_spec = mel_spec.unsqueeze(1)
    freqs = torch.linspace(0, 1, mel_spec.size(2), device=mel_spec.device)
    freqs = freqs.view(1, 1, -1, 1)
    weighted = mel_spec * freqs
    spectral_centroid = torch.sum(weighted, dim=2) / (torch.sum(mel_spec, dim=2) + 1e-8)
    return spectral_centroid


def spectral_bandwidth_fn(mel_spec: torch.Tensor, mel_bins: Optional[int] = None) -> torch.Tensor:
    """
    Calculate spectral bandwidth - frequency spread around the centroid.
    
    Spectral bandwidth measures the spread of frequencies around the centroid.
    Wider bandwidth indicates more complex timbre.
    
    Args:
        mel_spec: Melspectrogram tensor of shape [batch, mel_bins, time].
                  Expected to be in LINEAR scale (before log transformation).
        mel_bins: Number of mel bins. If None, inferred from mel_spec.shape[1].
    
    Returns:
        Spectral bandwidth tensor of shape [batch, time]
    
    Note:
        This function expects melspectrograms in LINEAR scale (before log transformation).
        If your melspectrograms are log-transformed, convert them back using expm1 before calling this function.
    """
    # Infer mel_bins from input shape if not provided
    # spectral_centroid_fn expects [batch, mel_bins, time] format
    if mel_bins is None:
        if mel_spec.dim() == 3:
            # Assume [batch, mel_bins, time] format (standard format)
            mel_bins = mel_spec.size(1)
        elif mel_spec.dim() == 2:
            # Assume [mel_bins, time] format
            mel_bins = mel_spec.size(0)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {mel_spec.dim()}D tensor")
    
    centroid = spectral_centroid_fn(mel_spec).unsqueeze(1)  # [batch, 1, time]
    
    mel_frequencies = torch.linspace(0, 1, mel_bins, device=mel_spec.device).unsqueeze(0).unsqueeze(-1)
    
    # Normalize
    mel_spec_normalized = mel_spec / (torch.sum(mel_spec, dim=1, keepdim=True) + 1e-8)
    
    # Calculate variance around centroid
    freq_diff = (mel_frequencies - centroid) ** 2
    bandwidth = torch.sum(freq_diff * mel_spec_normalized, dim=1)  # [batch, time]
    
    return torch.sqrt(bandwidth + 1e-8)


def spectral_flatness_fn(mel_spec: torch.Tensor) -> torch.Tensor:
    """
    Calculate spectral flatness - tonality vs noisiness.
    
    Spectral flatness shows how much the spectrum resembles noise (flat) or a tone (peaky).
    Higher values indicate more noise-like characteristics.
    
    Args:
        mel_spec: Melspectrogram tensor of shape [batch, mel_bins, time].
                  Expected to be in LINEAR scale (before log transformation).
    
    Returns:
        Spectral flatness tensor of shape [batch, time]
    
    Note:
        This function expects melspectrograms in LINEAR scale (before log transformation).
        If your melspectrograms are log-transformed, convert them back using expm1 before calling this function.
    """
    # Geometric mean
    geometric_mean = torch.exp(torch.mean(torch.log(mel_spec + 1e-8), dim=1))
    
    # Arithmetic mean
    arithmetic_mean = torch.mean(mel_spec, dim=1)
    
    # Spectral flatness
    flatness = geometric_mean / (arithmetic_mean + 1e-8)
    
    return flatness


def spectral_entropy_fn(mel_spec: torch.Tensor) -> torch.Tensor:
    """
    Calculate spectral entropy - texture complexity.
    
    Spectral entropy measures how evenly distributed the signal is across frequencies
    (noise vs structured signal).
    
    Args:
        mel_spec: Melspectrogram tensor of shape [batch, mel_bins, time].
                  Expected to be in LINEAR scale (before log transformation).
    
    Returns:
        Normalized spectral entropy tensor of shape [batch, time] in range [0, 1]
    
    Note:
        This function expects melspectrograms in LINEAR scale (before log transformation).
        If your melspectrograms are log-transformed, convert them back using expm1 before calling this function.
    """
    # Normalize each time frame
    mel_normalized = mel_spec / (torch.sum(mel_spec, dim=1, keepdim=True) + 1e-8)
    
    # Calculate entropy
    entropy = -torch.sum(mel_normalized * torch.log(mel_normalized + 1e-8), dim=1)
    
    # Normalize entropy to [0, 1]
    max_entropy = torch.log(torch.tensor(mel_spec.size(1), device=mel_spec.device))
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor by its norm.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Normalized tensor
    """
    return tensor / tensor.norm()


def calc_loss(func, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate loss between predictions and targets using a metric function.
    
    Applies the metric function to both predictions and targets, then computes
    mean squared error between the results.
    
    Args:
        func: Metric function to apply (e.g., spectral_centroid_fn)
        preds: Predicted melspectrograms (should be LINEAR scale)
        y: Target melspectrograms (should be LINEAR scale)
    
    Returns:
        Mean squared error loss tensor
    """
    preds_metric = func(preds)
    y_metric = func(y)
    return ((preds_metric - y_metric) ** 2).mean()


def visualize_melspectrograms(
    target_melspec: torch.Tensor,
    pred_melspec: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = 'viridis'
) -> None:
    """
    Visualize target and predicted melspectrograms side-by-side for comparison.
    
    Args:
        target_melspec: Target melspectrogram tensor of shape [batch, mel_bins, time] or [mel_bins, time].
                        If batched, only the first item will be visualized.
                        Expected to be in LINEAR scale (before log transformation).
        pred_melspec: Predicted melspectrogram tensor of shape [batch, mel_bins, time] or [mel_bins, time].
                      If batched, only the first item will be visualized.
                      Expected to be in LINEAR scale (before log transformation).
        save_path: Optional path to save the figure. If None, figure is displayed.
        figsize: Figure size tuple (width, height) in inches.
        cmap: Colormap to use for visualization.
    """
    # Convert to numpy if needed
    if isinstance(target_melspec, torch.Tensor):
        target_melspec = target_melspec.detach().cpu().numpy()
    if isinstance(pred_melspec, torch.Tensor):
        pred_melspec = pred_melspec.detach().cpu().numpy()
    
    # Handle batched inputs - take first item
    if target_melspec.ndim == 3:
        target_melspec = target_melspec[0]
    if pred_melspec.ndim == 3:
        pred_melspec = pred_melspec[0]
    
    # Ensure 2D shape [mel_bins, time]
    if target_melspec.ndim == 1:
        target_melspec = target_melspec.reshape(-1, 1)
    if pred_melspec.ndim == 1:
        pred_melspec = pred_melspec.reshape(-1, 1)
    
    # Transpose if needed to get [mel_bins, time] format
    if target_melspec.shape[0] < target_melspec.shape[1]:
        target_melspec = target_melspec.T
    if pred_melspec.shape[0] < pred_melspec.shape[1]:
        pred_melspec = pred_melspec.T
    
    # Apply log scale for better visualization (but keep original for metrics)
    target_melspec_viz = np.log1p(target_melspec)
    pred_melspec_viz = np.log1p(pred_melspec)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot target
    im1 = axes[0].imshow(
        target_melspec_viz,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        interpolation='nearest'
    )
    axes[0].set_title('Target Melspectrogram')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Mel Frequency Bin')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot prediction
    im2 = axes[1].imshow(
        pred_melspec_viz,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        interpolation='nearest'
    )
    axes[1].set_title('Predicted Melspectrogram')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Mel Frequency Bin')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

