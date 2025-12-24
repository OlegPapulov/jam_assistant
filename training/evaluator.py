"""
Evaluation functions for model evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Callable, Optional
import gc

from config.config import Config
from utils.metrics import normalize, calc_loss


def compute_metrics(
    melspecs_pred: torch.Tensor,
    melspecs_y: torch.Tensor,
    metric_functions: Dict[str, Callable]
) -> Dict[str, float]:
    """
    Compute metrics on predictions and targets.
    
    Args:
        melspecs_pred: Predicted melspectrograms (linear scale)
        melspecs_y: Target melspectrograms (linear scale)
        metric_functions: Dictionary of metric functions
    
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    for metric_name, metric_fn in metric_functions.items():
        if metric_fn is not None:
            try:
                metric_value = calc_loss(metric_fn, melspecs_pred, melspecs_y)
                metrics[metric_name] = metric_value.item()
            except Exception as e:
                print(f"Error computing {metric_name}: {e}")
                metrics[metric_name] = float('nan')
    
    return metrics


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    melspec_transform,
    config: Config,
    loss_functions: Dict[str, Callable],
    metric_functions: Dict[str, Callable],
    device: str = 'cuda',
    use_wandb: bool = False,
    wandb_run=None,
    epoch: Optional[int] = None,
    frequency_weighted_l1: Optional[nn.Module] = None
) -> tuple:
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        melspec_transform: MelSpectrogram transform
        config: Configuration object
        loss_functions: Dictionary of loss functions
        metric_functions: Dictionary of metric functions
        device: Device to use
        use_wandb: Whether to log to wandb
        wandb_run: Wandb run object
        epoch: Current epoch number (for visualization naming)
    
    Returns:
        Tuple of (losses_dict, metrics_dict)
    """
    model.eval()
    
    losses = {
        'reconstruction_loss': [],
        'reconstruction_loss_bass': [],
        'reconstruction_loss_drums': [],
        'spectral_centroid_loss': [],
        'spectral_bandwidth_loss': [],
        'spectral_flatness_loss': [],
        'spectral_entropy_loss': [],
        'frequency_weighted_l1_loss': [],
    }
    
    metrics = {}
    
    # Setup visualization if enabled
    visualization_count = 0
    if config.visualize_during_val:
        import os
        os.makedirs(config.val_visualization_dir, exist_ok=True)
        from utils.audio import visualize_audio_melspectrograms
    
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16):
            for batch_idx, batch in enumerate(val_loader):
                vocals_X, vocals_y = batch['vocals']
                drums_X, drums_y = batch['drums']
                bass_X, bass_y = batch['bass']
                other_X, other_y = batch['other']
                all_X, all_y = batch['all']
                
                samples_in_batch = vocals_X.shape[0]
                
                stems_X = torch.cat([vocals_X, drums_X, bass_X, other_X], dim=0).to(device)
                stems_y = torch.cat([vocals_y, drums_y, bass_y, other_y], dim=0).to(device)
                
                # Compute melspectrograms
                melspecs_X = melspec_transform(stems_X)
                melspecs_y = melspec_transform(stems_y)
                
                # Apply log transformation if needed
                if config.target_log:
                    melspecs_X = torch.log1p(melspecs_X)
                    melspecs_y = torch.log1p(melspecs_y)
                
                # Forward pass
                melspecs_pred, melspecs_pred_bass, melspecs_pred_drums = model(
                    melspecs_X,
                    batch_size=samples_in_batch
                )
                
                # Convert back from log scale if needed
                if config.target_log:
                    melspecs_pred = torch.expm1(melspecs_pred)
                    melspecs_pred_bass = torch.expm1(melspecs_pred_bass)
                    melspecs_pred_drums = torch.expm1(melspecs_pred_drums)
                
                # Normalize for reconstruction loss
                normalized_pred = normalize(melspecs_pred)
                normalized_bass_pred = normalize(melspecs_pred_bass)
                normalized_drums_pred = normalize(melspecs_pred_drums)
                normalized_y = normalize(melspecs_y)
                
                # Reconstruction losses
                reconstruction_loss = ((normalized_pred - normalized_y) ** 2).sum()
                reconstruction_loss_bass = (
                    (normalized_bass_pred - normalized_y[bass_X.shape[0] * 2:bass_X.shape[0] * 3]) ** 2
                ).sum()
                reconstruction_loss_drums = (
                    (normalized_drums_pred - normalized_y[drums_X.shape[0] * 1:bass_X.shape[0] * 2]) ** 2
                ).sum()
                
                # Spectral losses (on linear scale melspectrograms)
                spectral_centroid_loss = calc_loss(
                    loss_functions.get('spectral_centroid', None),
                    melspecs_pred,
                    melspecs_y
                ) if 'spectral_centroid' in loss_functions else torch.tensor(0.0, device=device)
                
                spectral_bandwidth_loss = calc_loss(
                    loss_functions.get('spectral_bandwidth', None),
                    melspecs_pred,
                    melspecs_y
                ) if 'spectral_bandwidth' in loss_functions else torch.tensor(0.0, device=device)
                
                spectral_flatness_loss = calc_loss(
                    loss_functions.get('spectral_flatness', None),
                    melspecs_pred,
                    melspecs_y
                ) if 'spectral_flatness' in loss_functions else torch.tensor(0.0, device=device)
                
                spectral_entropy_loss = calc_loss(
                    loss_functions.get('spectral_entropy', None),
                    melspecs_pred,
                    melspecs_y
                ) if 'spectral_entropy' in loss_functions else torch.tensor(0.0, device=device)
                
                # Frequency-weighted L1 loss (on linear scale melspectrograms)
                frequency_weighted_l1_loss_val = torch.tensor(0.0, device=device)
                if frequency_weighted_l1 is not None:
                    frequency_weighted_l1_loss_val = frequency_weighted_l1(melspecs_pred, melspecs_y)
                
                # Store losses
                losses['reconstruction_loss'].append(reconstruction_loss.item())
                losses['reconstruction_loss_bass'].append(reconstruction_loss_bass.item())
                losses['reconstruction_loss_drums'].append(reconstruction_loss_drums.item())
                losses['spectral_centroid_loss'].append(spectral_centroid_loss.item())
                losses['spectral_bandwidth_loss'].append(spectral_bandwidth_loss.item())
                losses['spectral_flatness_loss'].append(spectral_flatness_loss.item())
                losses['spectral_entropy_loss'].append(spectral_entropy_loss.item())
                losses['frequency_weighted_l1_loss'].append(frequency_weighted_l1_loss_val.item())
                
                # Compute metrics
                batch_metrics = compute_metrics(melspecs_pred, melspecs_y, metric_functions)
                for key, value in batch_metrics.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
                
                # Visualization during validation if enabled
                if config.visualize_during_val and visualization_count < config.val_visualization_samples:
                    # Visualize first sample from batch (vocals)
                    try:
                        # Get first sample from vocals predictions
                        target_melspec_sample = melspecs_y[0].cpu()  # [mel_bins, time]
                        pred_melspec_sample = melspecs_pred[0].cpu()  # [mel_bins, time]
                        
                        # Create save path
                        epoch_str = f"epoch_{epoch}" if epoch is not None else f"batch_{batch_idx}"
                        save_path = os.path.join(
                            config.val_visualization_dir,
                            f"val_{epoch_str}_sample_{visualization_count}.png"
                        )
                        
                        # Visualize
                        visualize_audio_melspectrograms(
                            target_melspec=target_melspec_sample,
                            pred_melspec=pred_melspec_sample,
                            config=config,
                            save_path=save_path,
                            show_difference=True
                        )
                        
                        visualization_count += 1
                        if epoch is not None:
                            print(f"Saved validation visualization: {save_path}")
                    except Exception as e:
                        print(f"Warning: Failed to save validation visualization: {e}")
                
                # Clean up
                del batch, stems_X, stems_y, melspecs_X, melspecs_y
                del vocals_X, vocals_y, drums_X, drums_y, bass_X, bass_y
                del other_X, other_y, all_X, all_y, melspecs_pred
                torch.cuda.empty_cache() if device == 'cuda' else None
                gc.collect()
                
                # Break early if we've visualized enough samples
                if config.visualize_during_val and visualization_count >= config.val_visualization_samples:
                    break
    
    # Average losses and metrics
    avg_losses = {
        key: sum(values) / len(values) if values else 0.0
        for key, values in losses.items()
    }
    
    avg_metrics = {
        key: sum(values) / len(values) if values else 0.0
        for key, values in metrics.items()
    }
    
    # Log to wandb if enabled
    if use_wandb and wandb_run:
        log_dict = {f'val/{k}': v for k, v in avg_losses.items()}
        log_dict.update({f'val/metric_{k}': v for k, v in avg_metrics.items()})
        wandb_run.log(log_dict)
    
    return avg_losses, avg_metrics

