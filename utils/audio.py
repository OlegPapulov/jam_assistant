"""
Audio utilities for converting melspectrograms to audio and comparison.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Optional, Tuple
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

