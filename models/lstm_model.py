"""
LSTM model architectures for audio generation.
"""

import torch
import torch.nn as nn
from typing import Optional


class BassFromHiddenModel(nn.Module):
    """Model for generating bass from hidden representations."""
    
    def __init__(
        self,
        sequence_length_input: int,
        sequence_length_output: int,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.sequence_length_input = sequence_length_input
        self.sequence_length_output = sequence_length_output
        
        self.bass_parameters = nn.Parameter(
            torch.rand(
                self.sequence_length_input,
                self.sequence_length_output,
                device=device,
                dtype=dtype
            ),
            requires_grad=True
        )
        
        self.lr1 = nn.Linear(
            self.sequence_length_output,
            self.sequence_length_output // 4,
            device=device,
            dtype=dtype
        )
        self.lr2 = nn.Linear(
            self.sequence_length_output // 4,
            self.sequence_length_output,
            device=device,
            dtype=dtype
        )
        self.activ = nn.Sigmoid()
    
    def forward(self, X):
        return self.lr2(self.activ(self.lr1(X + self.bass_parameters)))


class DrumsFromHiddenModel(nn.Module):
    """Model for generating drums from hidden representations."""
    
    def __init__(
        self,
        sequence_length_input: int,
        sequence_length_output: int,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.sequence_length_input = sequence_length_input
        self.sequence_length_output = sequence_length_output
        
        self.drums_parameters = nn.Parameter(
            torch.rand(
                self.sequence_length_input,
                self.sequence_length_output,
                device=device,
                dtype=dtype
            ),
            requires_grad=True
        )
        
        self.lr1 = nn.Linear(
            self.sequence_length_output,
            self.sequence_length_output // 4,
            device=device,
            dtype=dtype
        )
        self.lr2 = nn.Linear(
            self.sequence_length_output // 4,
            self.sequence_length_output,
            device=device,
            dtype=dtype
        )
        self.activ = nn.Sigmoid()
    
    def forward(self, X):
        return self.lr2(self.activ(self.lr1(X + self.drums_parameters)))


class LSTMModel(nn.Module):
    """
    LSTM model for audio sequence generation.
    
    Uses three LSTM layers to generate next sequence, bass, and drums predictions.
    """
    
    def __init__(
        self,
        mel_bins: int,
        sequence_length_input: int,
        sequence_length_output: int,
        dropout_rate: float,
        use_bidirectional: bool,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.mel_bins = mel_bins
        self.sequence_length_input = sequence_length_input
        self.sequence_length_output = sequence_length_output
        self.dropout_rate = dropout_rate
        self.use_bidirectional = use_bidirectional
        self.correction_coef = 1 if not use_bidirectional else 1 / 2
        
        self.lstm1 = nn.LSTM(
            input_size=self.sequence_length_input,
            hidden_size=int(self.sequence_length_input * self.correction_coef),
            num_layers=self.mel_bins,
            batch_first=True,
            bidirectional=self.use_bidirectional,
            dropout=self.dropout_rate if self.mel_bins > 1 else 0,
            device=device,
            dtype=dtype
        )
        
        self.lstm2 = nn.LSTM(
            input_size=self.sequence_length_input,
            hidden_size=int(self.sequence_length_input * self.correction_coef),
            num_layers=self.mel_bins,
            batch_first=True,
            bidirectional=self.use_bidirectional,
            dropout=self.dropout_rate if self.mel_bins > 1 else 0,
            device=device,
            dtype=dtype
        )
        
        self.lstm3 = nn.LSTM(
            input_size=self.sequence_length_input,
            hidden_size=int(self.sequence_length_input * self.correction_coef),
            num_layers=self.mel_bins,
            batch_first=True,
            bidirectional=self.use_bidirectional,
            dropout=self.dropout_rate if self.mel_bins > 1 else 0,
            device=device,
            dtype=dtype
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.final_layer = nn.Linear(
            self.sequence_length_input,
            self.sequence_length_output,
            device=device,
            dtype=dtype
        )
    
    def _continue(self, X):
        """Continue sequence generation."""
        next_seq, (hidden, cell) = self.lstm1(X)
        bass, _ = self.lstm2(next_seq, (hidden, cell))
        drums, _ = self.lstm3(next_seq, (hidden, cell))
        
        return (
            self.final_layer(next_seq),
            self.final_layer(bass),
            self.final_layer(drums)
        )
    
    def forward(self, X, batch_size: int):
        """
        Forward pass.
        
        Args:
            X: Input tensor
            batch_size: Batch size for output slicing
        
        Returns:
            Tuple of (next_seq, bass_pred, drums_pred)
        """
        next_seq, bass_pred, drums_pred = self._continue(X)
        return (
            next_seq,
            bass_pred[:batch_size],
            drums_pred[:batch_size]
        )


class MockNet(nn.Module):
    """Mock network for testing."""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(self.input_size, self.output_size)
        self.activ = nn.Sigmoid()
    
    def forward(self, X):
        return self.activ(self.linear(X))

