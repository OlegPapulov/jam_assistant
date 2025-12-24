"""
Configuration module for training.

Supports loading configuration from command-line arguments, environment variables,
and YAML/JSON files.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Training configuration class."""
    
    # Data parameters
    current_sample_rate: int = 22050
    target_sample_rate: int = 22050
    num_input_seconds: float = 5.0
    num_output_seconds: float = 0.5
    one_bit_seconds: float = 0.2
    test_size: float = 0.2
    use_7s: bool = True
    
    # Model parameters
    n_fft: int = 1024
    mel_bins: int = 128
    dropout_rate: float = 0.2
    use_bidirectional: bool = False
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 10
    target_log: bool = True
    spectral_backward: bool = False
    
    # Reproducibility
    random_seed: int = 42
    
    # Wandb configuration
    use_wandb: bool = False
    wandb_api_key: Optional[str] = None
    wandb_project: str = "jam-assistant"
    
    # Computed parameters (set automatically)
    hop_length: Optional[int] = None
    sequence_length_input: Optional[int] = None
    sequence_length_output: Optional[int] = None
    
    def __post_init__(self):
        """Compute derived parameters after initialization."""
        # Compute hop_length
        if self.hop_length is None:
            self.hop_length = self.n_fft // 2
        
        # Compute sequence lengths
        if self.sequence_length_input is None:
            self.sequence_length_input = int(
                1 + (self.num_input_seconds * self.target_sample_rate - self.n_fft) 
                // self.hop_length + 2
            )
        
        if self.sequence_length_output is None:
            self.sequence_length_output = int(
                1 + (self.num_output_seconds * self.target_sample_rate - self.n_fft) 
                // self.hop_length + 2
            )
        
        # Set wandb API key from environment if not provided
        if self.wandb_api_key is None:
            self.wandb_api_key = os.environ.get('WANDB_API_KEY')
        
        # Set environment variable if wandb_api_key is provided
        if self.wandb_api_key and self.use_wandb:
            os.environ['WANDB_API_KEY'] = self.wandb_api_key


def parse_args() -> Config:
    """
    Parse command-line arguments and return Config object.
    
    Returns:
        Config object with parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train LSTM model for audio generation')
    
    # Data parameters
    parser.add_argument('--current_sample_rate', type=int, default=22050,
                       help='Current sample rate of audio')
    parser.add_argument('--target_sample_rate', type=int, default=22050,
                       help='Target sample rate for resampling')
    parser.add_argument('--num_input_seconds', type=float, default=5.0,
                       help='Number of seconds for input sequence')
    parser.add_argument('--num_output_seconds', type=float, default=0.5,
                       help='Number of seconds for output sequence')
    parser.add_argument('--one_bit_seconds', type=float, default=0.2,
                       help='Time step for sampling')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--use_7s', action='store_true', default=True,
                       help='Use MUSDB 7s dataset')
    parser.add_argument('--no_use_7s', dest='use_7s', action='store_false',
                       help='Use full MUSDB dataset instead of 7s')
    
    # Model parameters
    parser.add_argument('--n_fft', type=int, default=1024,
                       help='FFT window size')
    parser.add_argument('--mel_bins', type=int, default=128,
                       help='Number of mel frequency bins')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--use_bidirectional', action='store_true', default=False,
                       help='Use bidirectional LSTM')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--target_log', action='store_true', default=True,
                       help='Apply log transformation to melspectrograms')
    parser.add_argument('--no_target_log', dest='target_log', action='store_false',
                       help='Do not apply log transformation')
    parser.add_argument('--spectral_backward', action='store_true', default=False,
                       help='Use spectral losses for backpropagation')
    
    # Reproducibility
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Wandb configuration
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Enable wandb logging')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                       help='Wandb API key (or use WANDB_API_KEY env var)')
    parser.add_argument('--wandb_project', type=str, default='jam-assistant',
                       help='Wandb project name')
    
    args = parser.parse_args()
    
    # Convert argparse Namespace to Config
    config = Config(
        current_sample_rate=args.current_sample_rate,
        target_sample_rate=args.target_sample_rate,
        num_input_seconds=args.num_input_seconds,
        num_output_seconds=args.num_output_seconds,
        one_bit_seconds=args.one_bit_seconds,
        test_size=args.test_size,
        use_7s=args.use_7s,
        n_fft=args.n_fft,
        mel_bins=args.mel_bins,
        dropout_rate=args.dropout_rate,
        use_bidirectional=args.use_bidirectional,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        target_log=args.target_log,
        spectral_backward=args.spectral_backward,
        random_seed=args.random_seed,
        use_wandb=args.use_wandb,
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
    )
    
    return config

