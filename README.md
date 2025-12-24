# Jam Assistant - Audio Generation Model

LSTM-based model for audio sequence generation using melspectrograms.

## Project Structure

```
jam_assistant_cursor/
├── config/              # Configuration management
├── data/                # Dataset and preprocessing
├── models/              # Model architectures
├── training/            # Training and evaluation
├── utils/               # Utility functions (metrics, seed)
├── visualization/       # Visualization tools
├── train.py            # Main training script
├── evaluate.py         # Evaluation script
├── prepare_data.py     # Data preparation script
└── requirements.txt    # Dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Prepare the MUSDB dataset:
```bash
python prepare_data.py --use_7s --target_sample_rate 22050
```

### Training

Train the model with default settings:
```bash
python train.py
```

Train with custom parameters:
```bash
python train.py \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --num_epochs 10 \
    --mel_bins 128 \
    --use_7s \
    --target_log \
    --random_seed 42 \
    --use_wandb \
    --wandb_api_key YOUR_API_KEY
```

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --checkpoint ./checkpoints/final_model.pt
```

## Configuration

All configuration parameters can be set via command-line arguments. See `python train.py --help` for full list.

Key parameters:
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate
- `--num_epochs`: Number of training epochs
- `--mel_bins`: Number of mel frequency bins
- `--use_7s`: Use MUSDB 7s dataset (default: True)
- `--target_log`: Apply log transformation to melspectrograms
- `--spectral_backward`: Use spectral losses for backpropagation
- `--random_seed`: Random seed for reproducibility
- `--use_wandb`: Enable wandb logging
- `--wandb_api_key`: Wandb API key (or set WANDB_API_KEY env var)

## Features

- **Modular Architecture**: Separated into config, data, models, training, and visualization modules
- **Reproducibility**: Random seed configuration for consistent results
- **Flexible Training**: Configurable loss functions and metrics
- **Visualization**: Built-in visualization tools for training metrics and predictions
- **Checkpointing**: Automatic checkpoint saving during training
- **Wandb Integration**: Optional wandb logging for experiment tracking

## Metrics

The model uses several metrics for evaluation:
- Reconstruction Loss: MSE on normalized melspectrograms
- Spectral Centroid: Frequency brightness
- Spectral Bandwidth: Frequency spread
- Spectral Flatness: Tonality vs noisiness
- Spectral Entropy: Texture complexity

All spectral metrics operate on linear-scale melspectrograms (before log transformation).

## Audio Restoration and Comparison

The `utils.audio` module provides functions to restore audio waveforms from melspectrograms using torchaudio's Griffin-Lim algorithm:

### Audio Restoration

```python
from utils.audio import restore_audio_from_melspec
from config.config import Config

# Restore audio from melspectrogram
# Uses all parameters from config: target_sample_rate, n_fft, hop_length, mel_bins
audio_waveform = restore_audio_from_melspec(
    mel_spec=predicted_melspec,  # Linear scale melspectrogram
    config=config,               # Config object with all audio parameters
    n_iter=32,                   # Griffin-Lim iterations (default: 32)
    device='cuda'
)
```

### Audio Comparison

```python
from utils.audio import compare_audio, play_audio_comparison

# Convert melspectrograms to audio and save
target_audio, pred_audio = compare_audio(
    target_melspec=target_melspec,  # Linear scale melspectrogram
    pred_melspec=pred_melspec,      # Linear scale melspectrogram
    config=config,                  # Config object with sample_rate
    save_dir='./audio_comparisons',
    prefix='epoch_1'
)

# Or play audio directly (in Jupyter notebook or with sounddevice)
target_audio, pred_audio = play_audio_comparison(
    target_melspec=target_melspec,
    pred_melspec=pred_melspec,
    config=config,
    save_dir='./audio_comparisons',
    play=True  # Requires IPython.display or sounddevice
)
```

### During Evaluation

Save audio samples during evaluation:
```bash
python evaluate.py --checkpoint ./checkpoints/final_model.pt --save_audio --audio_samples 5
```

The restoration function uses torchaudio's Griffin-Lim algorithm with parameters from config:
- `target_sample_rate`: Sample rate for audio output
- `n_fft`: FFT window size
- `hop_length`: Hop length for STFT
- `mel_bins`: Number of mel frequency bins

Note: Perfect reconstruction from melspectrogram is not possible, but Griffin-Lim provides a reasonable approximation for comparison.

## Notes

- The model expects melspectrograms in linear scale for spectral metrics
- If `target_log=True`, melspectrograms are log-transformed for training but converted back to linear scale for metrics
- Checkpoints are saved after each epoch in `./checkpoints/`
- Training logs and visualizations are saved in `./logs/`
- Audio comparison uses sample rate from config (`target_sample_rate`)
- For audio playback, install `sounddevice` (optional): `pip install sounddevice`

