"""
Training functions for model training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Callable, List, Optional
import gc

from config.config import Config
from utils.metrics import normalize, calc_loss
from utils.audio import restore_audio_from_melspec


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    melspec_transform,
    config: Config,
    loss_functions: Dict[str, Callable],
    metric_functions: Dict[str, Callable],
    device: str = 'cuda',
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_wandb: bool = False,
    wandb_run=None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        melspec_transform: MelSpectrogram transform
        config: Configuration object
        loss_functions: Dictionary of loss functions
        metric_functions: Dictionary of metric functions
        device: Device to use
        scaler: Gradient scaler for mixed precision
        use_wandb: Whether to log to wandb
        wandb_run: Wandb run object
    
    Returns:
        Dictionary of average losses for the epoch
    """
    model.train()
    optimizer.zero_grad()
    
    epoch_losses = {
        'reconstruction_loss': [],
        'reconstruction_loss_bass': [],
        'reconstruction_loss_drums': [],
        'spectral_centroid_loss': [],
    }
    
    for batch in train_loader:
        with torch.autocast(device_type=device, dtype=torch.float16):
            vocals_X, vocals_y = batch['vocals']
            drums_X, drums_y = batch['drums']
            bass_X, bass_y = batch['bass']
            other_X, other_y = batch['other']
            all_X, all_y = batch['all']
            
            samples_in_batch = vocals_X.shape[0]
            
            stems_X = torch.cat([vocals_X, drums_X, bass_X, other_X], dim=0).to(device)
            stems_y = torch.cat([vocals_y, drums_y, bass_y, other_y], dim=0).to(device)
            
            # Clean up
            del vocals_X, vocals_y, drums_X, drums_y, bass_X, bass_y
            del other_X, other_y, all_X, all_y
            
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
                (normalized_bass_pred - normalized_y[samples_in_batch * 2:samples_in_batch * 3]) ** 2
            ).sum()
            reconstruction_loss_drums = (
                (normalized_drums_pred - normalized_y[samples_in_batch * 1:samples_in_batch * 2]) ** 2
            ).sum()
            
            # Spectral losses (on linear scale melspectrograms)
            spectral_centroid_loss = calc_loss(
                loss_functions.get('spectral_centroid', None),
                melspecs_pred,
                melspecs_y
            ) if 'spectral_centroid' in loss_functions else torch.tensor(0.0, device=device)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(reconstruction_loss).backward(retain_graph=True)
            scaler.scale(reconstruction_loss_bass).backward(retain_graph=True)
            scaler.scale(reconstruction_loss_drums).backward(retain_graph=True)
            
            if config.spectral_backward and spectral_centroid_loss.item() > 0:
                scaler.scale(spectral_centroid_loss).backward(retain_graph=True)
        else:
            reconstruction_loss.backward(retain_graph=True)
            reconstruction_loss_bass.backward(retain_graph=True)
            reconstruction_loss_drums.backward(retain_graph=True)
            
            if config.spectral_backward and spectral_centroid_loss.item() > 0:
                spectral_centroid_loss.backward(retain_graph=True)
        
        # Store losses
        epoch_losses['reconstruction_loss'].append(reconstruction_loss.item())
        epoch_losses['reconstruction_loss_bass'].append(reconstruction_loss_bass.item())
        epoch_losses['reconstruction_loss_drums'].append(reconstruction_loss_drums.item())
        epoch_losses['spectral_centroid_loss'].append(spectral_centroid_loss.item())
        
        # Log to wandb if enabled
        if use_wandb and wandb_run:
            wandb_run.log({
                'train/reconstruction_loss': reconstruction_loss.item(),
                'train/reconstruction_loss_bass': reconstruction_loss_bass.item(),
                'train/reconstruction_loss_drums': reconstruction_loss_drums.item(),
                'train/spectral_centroid_loss': spectral_centroid_loss.item(),
            })
        
        # Clean up
        del batch, stems_X, stems_y, melspecs_X, melspecs_y, melspecs_pred
        torch.cuda.empty_cache() if device == 'cuda' else None
        gc.collect()
    
    # Step optimizer
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    
    # Return average losses
    return {
        key: sum(values) / len(values) if values else 0.0
        for key, values in epoch_losses.items()
    }


def train(
    config: Config,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_functions: Dict[str, Callable],
    metric_functions: Dict[str, Callable],
    device: str = 'cuda',
    save_dir: str = './checkpoints',
    log_dir: str = './logs',
    use_wandb: bool = False,
    wandb_run=None
) -> Dict[str, List[float]]:
    """
    Main training function.
    
    Args:
        config: Configuration object
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        loss_functions: Dictionary of loss functions
        metric_functions: Dictionary of metric functions
        device: Device to use
        save_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        use_wandb: Whether to use wandb logging
        wandb_run: Wandb run object
    
    Returns:
        Dictionary of training history
    """
    import os
    import torchaudio.transforms as T
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create melspectrogram transform
    melspec = T.MelSpectrogram(
        config.target_sample_rate,
        n_fft=config.n_fft,
        n_mels=config.mel_bins,
        hop_length=config.hop_length
    ).to(device)
    
    # Create gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_metrics': [],
        'val_metrics': [],
    }
    
    # Training loop
    for epoch in range(config.num_epochs):
        # Train epoch
        train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            melspec_transform=melspec,
            config=config,
            loss_functions=loss_functions,
            metric_functions=metric_functions,
            device=device,
            scaler=scaler,
            use_wandb=use_wandb,
            wandb_run=wandb_run
        )
        
        # Evaluate
        from training import evaluate
        val_losses, val_metrics = evaluate(
            model=model,
            val_loader=val_loader,
            melspec_transform=melspec,
            config=config,
            loss_functions=loss_functions,
            metric_functions=metric_functions,
            device=device,
            use_wandb=use_wandb,
            wandb_run=wandb_run
        )
        
        # Store history
        history['train_losses'].append(train_losses)
        history['val_losses'].append(val_losses)
        history['val_metrics'].append(val_metrics)
        
        # Print progress
        print(f'Epoch {epoch + 1}/{config.num_epochs}')
        print(f'  Train - Reconstruction Loss: {train_losses["reconstruction_loss"]:.4f}')
        print(f'  Val - Reconstruction Loss: {val_losses["reconstruction_loss"]:.4f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'history': history,
        }, checkpoint_path)
        
        # Save sample audio every few epochs or at the end
        if (epoch + 1) % max(1, config.num_epochs // 3) == 0 or (epoch + 1) == config.num_epochs:
            try:
                from utils.audio import compare_audio
                # Get a sample from validation set
                model.eval()
                with torch.no_grad():
                    sample_batch = next(iter(val_loader))
                    vocals_X, vocals_y = sample_batch['vocals']
                    stems_y = vocals_y[0:1].to(device)
                    
                    # Compute melspectrograms
                    melspecs_y = melspec(stems_y)
                    if config.target_log:
                        melspecs_X_log = torch.log1p(melspecs_y)
                    else:
                        melspecs_X_log = melspecs_y
                    
                    # Forward pass
                    melspecs_pred, _, _ = model(melspecs_X_log, batch_size=1)
                    
                    # Convert back from log scale if needed
                    if config.target_log:
                        melspecs_pred = torch.expm1(melspecs_pred)
                    
                    # Save audio comparison
                    audio_dir = os.path.join(log_dir, 'audio_samples')
                    compare_audio(
                        target_melspec=melspecs_y[0],
                        pred_melspec=melspecs_pred[0],
                        config=config,
                        save_dir=audio_dir,
                        prefix=f'epoch_{epoch + 1}',
                        device=device
                    )
                model.train()
            except Exception as e:
                print(f"Warning: Could not save audio sample: {e}")
        
        if use_wandb and wandb_run:
            wandb_run.log({
                'epoch': epoch + 1,
                'train/reconstruction_loss': train_losses['reconstruction_loss'],
                'val/reconstruction_loss': val_losses['reconstruction_loss'],
            })
    
    return history

