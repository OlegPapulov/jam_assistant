"""
Audio utilities for converting melspectrograms to audio and comparison.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Union
from torch.utils.data import DataLoader, Dataset
from config.config import Config


def restore_audio_from_melspec(
    mel_spec: torch.Tensor,
    config: Config,
    n_iter: int = 32,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Restore audio waveform from melspectrogram using torchaudio's Griffin-Lim algorithm.
    
    This function uses all necessary parameters from config:
    - target_sample_rate: Sample rate for audio output
    - n_fft: FFT window size
    - hop_length: Hop length for STFT
    - mel_bins: Number of mel frequency bins
    
    Note: Perfect reconstruction from melspectrogram is not possible, but Griffin-Lim
    provides a reasonable approximation.
    
    Args:
        mel_spec: Melspectrogram tensor of shape [batch, mel_bins, time] or [mel_bins, time].
                  Expected to be in LINEAR scale (before log transformation).
        config: Configuration object containing target_sample_rate, n_fft, hop_length, mel_bins
        n_iter: Number of iterations for Griffin-Lim algorithm (default: 32)
        device: Device to use
    
    Returns:
        Audio waveform tensor of shape [batch, time] or [time]
    """
    # Handle batched input
    batched = mel_spec.dim() == 3
    if batched:
        mel_spec = mel_spec.to(device)
    else:
        mel_spec = mel_spec.unsqueeze(0).to(device)
    
    # Create inverse mel scale transform
    # InverseMelScale converts mel-scale spectrogram back to linear-scale spectrogram
    inverse_mel = T.InverseMelScale(
        n_stft=config.n_fft // 2 + 1,
        n_mels=config.mel_bins,
        sample_rate=config.target_sample_rate,
        f_min=0.0,
        f_max=config.target_sample_rate // 2
    ).to(device)
    
    # Convert melspectrogram to linear spectrogram
    # Input shape: [batch, mel_bins, time] -> Output shape: [batch, n_stft, time]
    linear_spec = inverse_mel(mel_spec)
    
    # Use Griffin-Lim to convert spectrogram to waveform
    # Uses n_fft and hop_length from config
    griffin_lim = T.GriffinLim(
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_iter=n_iter
    ).to(device)
    
    # Convert to waveform
    waveform = griffin_lim(linear_spec)
    
    # Remove batch dimension if input wasn't batched
    if not batched:
        waveform = waveform.squeeze(0)
    
    return waveform


def melspectrogram_to_audio(
    mel_spec: torch.Tensor,
    config: Config,
    n_iter: int = 32,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Convert melspectrogram back to audio waveform using Griffin-Lim algorithm.
    
    This is an alias for restore_audio_from_melspec() for backward compatibility.
    
    Args:
        mel_spec: Melspectrogram tensor of shape [batch, mel_bins, time] or [mel_bins, time].
                  Expected to be in LINEAR scale (before log transformation).
        config: Configuration object containing sample_rate, n_fft, hop_length, mel_bins
        n_iter: Number of iterations for Griffin-Lim algorithm
        device: Device to use
    
    Returns:
        Audio waveform tensor of shape [batch, time] or [time]
    """
    return restore_audio_from_melspec(mel_spec, config, n_iter=n_iter, device=device)


def compare_audio(
    target_melspec: torch.Tensor,
    pred_melspec: torch.Tensor,
    config: Config,
    save_dir: Optional[str] = None,
    prefix: str = "comparison",
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert target and predicted melspectrograms to audio and save for comparison.
    
    Args:
        target_melspec: Target melspectrogram (linear scale)
        pred_melspec: Predicted melspectrogram (linear scale)
        config: Configuration object
        save_dir: Directory to save audio files (if None, files are not saved)
        prefix: Prefix for saved audio files
        device: Device to use
    
    Returns:
        Tuple of (target_audio, pred_audio) waveforms
    """
    # Convert melspectrograms to audio using restoration function
    print("Converting target melspectrogram to audio...")
    target_audio = restore_audio_from_melspec(target_melspec, config, device=device)
    
    print("Converting predicted melspectrogram to audio...")
    pred_audio = restore_audio_from_melspec(pred_melspec, config, device=device)
    
    # Ensure same length (take minimum)
    min_length = min(target_audio.shape[-1], pred_audio.shape[-1])
    target_audio = target_audio[..., :min_length]
    pred_audio = pred_audio[..., :min_length]
    
    # Normalize audio to [-1, 1] range
    target_audio = target_audio / (target_audio.abs().max() + 1e-8)
    pred_audio = pred_audio / (pred_audio.abs().max() + 1e-8)
    
    # Save audio files if save_dir is provided
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save as WAV files
        target_path = os.path.join(save_dir, f"{prefix}_target.wav")
        pred_path = os.path.join(save_dir, f"{prefix}_predicted.wav")
        
        # Ensure 2D shape [channels, samples] for torchaudio.save
        if target_audio.dim() == 1:
            target_audio_save = target_audio.unsqueeze(0)
        else:
            target_audio_save = target_audio
        
        if pred_audio.dim() == 1:
            pred_audio_save = pred_audio.unsqueeze(0)
        else:
            pred_audio_save = pred_audio
        
        torchaudio.save(target_path, target_audio_save.cpu(), config.target_sample_rate)
        torchaudio.save(pred_path, pred_audio_save.cpu(), config.target_sample_rate)
        
        print(f"Saved target audio to: {target_path}")
        print(f"Saved predicted audio to: {pred_path}")
        
        # Create concatenated comparison (target then predicted)
        comparison_audio = torch.cat([target_audio_save, pred_audio_save], dim=-1)
        comparison_path = os.path.join(save_dir, f"{prefix}_comparison.wav")
        torchaudio.save(comparison_path, comparison_audio.cpu(), config.target_sample_rate)
        print(f"Saved comparison audio (target + predicted) to: {comparison_path}")
    
    return target_audio, pred_audio


def play_audio_comparison(
    target_melspec: torch.Tensor,
    pred_melspec: torch.Tensor,
    config: Config,
    save_dir: Optional[str] = None,
    prefix: str = "comparison",
    device: str = 'cpu',
    play: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert melspectrograms to audio, save files, and optionally play them.
    
    Args:
        target_melspec: Target melspectrogram (linear scale)
        pred_melspec: Predicted melspectrogram (linear scale)
        config: Configuration object
        save_dir: Directory to save audio files
        prefix: Prefix for saved audio files
        device: Device to use
        play: Whether to play audio (requires IPython.display or sounddevice)
    
    Returns:
        Tuple of (target_audio, pred_audio) waveforms
    """
    target_audio, pred_audio = compare_audio(
        target_melspec=target_melspec,
        pred_melspec=pred_melspec,
        config=config,
        save_dir=save_dir,
        prefix=prefix,
        device=device
    )
    
    if play:
        try:
            # Try IPython.display for Jupyter notebooks
            from IPython.display import Audio, display
            
            print("\nPlaying target audio...")
            display(Audio(target_audio.cpu().numpy(), rate=config.target_sample_rate))
            
            print("Playing predicted audio...")
            display(Audio(pred_audio.cpu().numpy(), rate=config.target_sample_rate))
        except ImportError:
            try:
                # Try sounddevice for command-line playback
                import sounddevice as sd
                
                print("\nPlaying target audio...")
                sd.play(target_audio.cpu().numpy(), samplerate=config.target_sample_rate)
                sd.wait()
                
                print("Playing predicted audio...")
                sd.play(pred_audio.cpu().numpy(), samplerate=config.target_sample_rate)
                sd.wait()
            except ImportError:
                print("Warning: No audio playback library available. Install 'sounddevice' or use in Jupyter notebook.")
    
    return target_audio, pred_audio


def get_audio_sample_from_dataloader(
    dataloader: DataLoader,
    sample_idx: Optional[int] = None,
    stem_type: str = 'vocals',
    return_input: bool = True,
    return_output: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Extract audio sample(s) from a dataloader (convenience wrapper).
    
    Args:
        dataloader: DataLoader instance
        sample_idx: Index of sample to extract. If None, returns first sample.
        stem_type: Type of stem to extract ('vocals', 'drums', 'bass', 'other', 'all')
        return_input: Whether to return input audio (X)
        return_output: Whether to return output audio (y)
    
    Returns:
        Dictionary containing requested audio tensors
    """
    return get_audio_sample_from_dataset(
        dataset=dataloader,
        sample_idx=sample_idx,
        stem_type=stem_type,
        return_input=return_input,
        return_output=return_output
    )


def get_audio_sample_from_dataset(
    dataset: Union[Dataset, DataLoader],
    sample_idx: Optional[int] = None,
    stem_type: str = 'vocals',
    return_input: bool = True,
    return_output: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Extract audio sample(s) from a dataset or dataloader.
    
    Args:
        dataset: Dataset or DataLoader instance
        sample_idx: Index of sample to extract. If None, returns first sample.
                   If dataset is a DataLoader, extracts from first batch.
        stem_type: Type of stem to extract ('vocals', 'drums', 'bass', 'other', 'all')
        return_input: Whether to return input audio (X)
        return_output: Whether to return output audio (y)
    
    Returns:
        Dictionary containing requested audio tensors:
        - 'input': Input audio waveform (if return_input=True)
        - 'output': Output audio waveform (if return_output=True)
        - 'stem_type': The stem type used
    """
    # Handle DataLoader
    if isinstance(dataset, DataLoader):
        batch = next(iter(dataset))
        if sample_idx is None:
            sample_idx = 0
        
        stem_data = batch[stem_type]
        if return_input and return_output:
            X, y = stem_data
            X_sample = X[sample_idx] if X.dim() > 1 else X
            y_sample = y[sample_idx] if y.dim() > 1 else y
            return {
                'input': X_sample,
                'output': y_sample,
                'stem_type': stem_type
            }
        elif return_input:
            X, _ = stem_data
            X_sample = X[sample_idx] if X.dim() > 1 else X
            return {
                'input': X_sample,
                'stem_type': stem_type
            }
        elif return_output:
            _, y = stem_data
            y_sample = y[sample_idx] if y.dim() > 1 else y
            return {
                'output': y_sample,
                'stem_type': stem_type
            }
    else:
        # Handle Dataset
        if sample_idx is None:
            sample_idx = 0
        
        sample = dataset[sample_idx]
        stem_data = sample[stem_type]
        
        if return_input and return_output:
            X, y = stem_data
            return {
                'input': X,
                'output': y,
                'stem_type': stem_type
            }
        elif return_input:
            X, _ = stem_data
            return {
                'input': X,
                'stem_type': stem_type
            }
        elif return_output:
            _, y = stem_data
            return {
                'output': y,
                'stem_type': stem_type
            }
    
    return {}


def create_melspectrogram_transform(
    config: Config,
    device: str = 'cpu'
) -> T.MelSpectrogram:
    """
    Create a MelSpectrogram transform from config.
    
    Args:
        config: Configuration object
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        MelSpectrogram transform ready to use
    """
    return T.MelSpectrogram(
        sample_rate=config.target_sample_rate,
        n_fft=config.n_fft,
        n_mels=config.mel_bins,
        hop_length=config.hop_length
    ).to(device)


def visualize_audio_melspectrograms(
    target_melspec: torch.Tensor,
    pred_melspec: torch.Tensor,
    config: Config,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = 'viridis',
    show_difference: bool = True
) -> None:
    """
    Visualize target and predicted melspectrograms side-by-side with optional difference plot.
    
    This function is designed to work with audio samples from datasets and can be used
    with other audio utilities.
    
    Args:
        target_melspec: Target melspectrogram tensor of shape [batch, mel_bins, time] or [mel_bins, time].
                       Expected to be in LINEAR scale (before log transformation).
        pred_melspec: Predicted melspectrogram tensor of shape [batch, mel_bins, time] or [mel_bins, time].
                     Expected to be in LINEAR scale (before log transformation).
        config: Configuration object
        save_path: Optional path to save the figure. If None, figure is displayed.
        figsize: Figure size tuple (width, height) in inches.
        cmap: Colormap to use for visualization.
        show_difference: Whether to show a difference plot between target and prediction.
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
    
    # Ensure same dimensions for comparison
    min_mel_bins = min(target_melspec.shape[0], pred_melspec.shape[0])
    min_time = min(target_melspec.shape[1], pred_melspec.shape[1])
    target_melspec = target_melspec[:min_mel_bins, :min_time]
    pred_melspec = pred_melspec[:min_mel_bins, :min_time]
    
    # Apply log scale for better visualization
    target_melspec_viz = np.log1p(target_melspec)
    pred_melspec_viz = np.log1p(pred_melspec)
    
    # Create figure
    num_plots = 3 if show_difference else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    if num_plots == 2:
        axes = [axes[0], axes[1]]
    else:
        axes = [axes[0], axes[1], axes[2]]
    
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
    
    # Plot difference if requested
    if show_difference:
        difference = np.abs(target_melspec_viz - pred_melspec_viz)
        im3 = axes[2].imshow(
            difference,
            aspect='auto',
            origin='lower',
            cmap='hot',
            interpolation='nearest'
        )
        axes[2].set_title('Absolute Difference')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Mel Frequency Bin')
        plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[Config] = None,
    device: str = 'cpu'
) -> Tuple[torch.nn.Module, Config]:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        config: Optional config object. If None, will try to load from checkpoint or use defaults.
        device: Device to load model on
    
    Returns:
        Tuple of (model, config)
    """
    from models.lstm_model import LSTMModel
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or use provided/default
    if config is None:
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = Config()
    
    # Get model state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    
    # Create model
    model = LSTMModel(
        mel_bins=config.mel_bins,
        sequence_length_input=config.sequence_length_input,
        sequence_length_output=config.sequence_length_output,
        dropout_rate=config.dropout_rate,
        use_bidirectional=config.use_bidirectional,
        device=device
    ).to(device)
    
    # Load weights
    model.load_state_dict(model_state_dict)
    model.eval()
    
    return model, config


def predict_melspectrogram(
    model: torch.nn.Module,
    audio_input: torch.Tensor,
    config: Config,
    melspec_transform: Optional[T.MelSpectrogram] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Make prediction on audio input using trained model.
    
    Args:
        model: Trained model
        audio_input: Input audio waveform tensor [time] or [batch, time]
        config: Configuration object
        melspec_transform: Optional melspectrogram transform. If None, will create one.
        device: Device to use
    
    Returns:
        Predicted melspectrogram tensor [mel_bins, time] or [batch, mel_bins, time]
    """
    model.eval()
    
    # Create melspectrogram transform if not provided
    if melspec_transform is None:
        melspec_transform = create_melspectrogram_transform(config, device)
    
    # Ensure audio has channel dimension for melspectrogram transform
    # MelSpectrogram expects [batch, channels, time] or [channels, time]
    if audio_input.dim() == 1:
        audio_input = audio_input.unsqueeze(0)  # [1, time] -> add channel dim
    elif audio_input.dim() == 2:
        # Could be [batch, time] or [channels, time]
        # If first dim is small, assume it's channels, else add channel dim
        if audio_input.shape[0] > 10:  # Likely [batch, time]
            audio_input = audio_input.unsqueeze(1)  # [batch, 1, time]
        # Otherwise assume [channels, time] which is fine
    
    # Move to device
    audio_input = audio_input.to(device)
    
    # Compute melspectrogram - outputs [batch, mel_bins, time]
    melspec_input = melspec_transform(audio_input)
    
    # Apply log transformation if needed
    if config.target_log:
        melspec_input = torch.log1p(melspec_input)
    
    # Forward pass - model expects [batch, mel_bins, time] format
    with torch.no_grad():
        batch_size = melspec_input.shape[0]
        melspec_pred, _, _ = model(melspec_input, batch_size=batch_size)
    
    # Convert back from log scale if needed
    if config.target_log:
        melspec_pred = torch.expm1(melspec_pred)
    
    return melspec_pred


def visualize_sample_prediction(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    config: Config,
    stem_type: str = 'vocals',
    save_path: Optional[str] = None,
    device: str = 'cpu'
) -> None:
    """
    Visualize model prediction on a dataset sample.
    
    This is a complete pipeline that:
    1. Takes a sample from dataset
    2. Makes prediction with model
    3. Visualizes target vs predicted melspectrograms
    
    Args:
        model: Trained model
        sample: Sample dictionary from get_audio_sample_from_dataset()
        config: Configuration object
        stem_type: Stem type to visualize ('vocals', 'drums', 'bass', 'other')
        save_path: Optional path to save visualization
        device: Device to use
    """
    # Get target audio
    target_audio = sample['output']  # Output is the target
    
    # Create melspectrogram transform
    melspec_transform = create_melspectrogram_transform(config, device)
    
    # Get target melspectrogram
    if target_audio.dim() == 1:
        target_audio = target_audio.unsqueeze(0)  # Add channel dimension
    
    target_melspec = melspec_transform(target_audio.to(device))
    
    # Make prediction
    pred_melspec = predict_melspectrogram(
        model=model,
        audio_input=sample['input'],  # Use input for prediction
        config=config,
        melspec_transform=melspec_transform,
        device=device
    )
    
    # Visualize
    visualize_audio_melspectrograms(
        target_melspec=target_melspec[0] if target_melspec.dim() == 3 else target_melspec,
        pred_melspec=pred_melspec[0] if pred_melspec.dim() == 3 else pred_melspec,
        config=config,
        save_path=save_path,
        show_difference=True
    )

