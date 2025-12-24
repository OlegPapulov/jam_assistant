"""
Dataset classes for MUSDB audio data.
"""

import os
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
from typing import Optional
import joblib


class MusDBDataset(Dataset):
    """
    Dataset for MUSDB 7s dataset.
    
    Loads audio tracks from MUSDB dataset and creates input/output pairs
    for training.
    """
    
    def __init__(
        self,
        mus_dataset,
        current_sample_rate: int,
        target_sample_rate: int,
        num_input_seconds: float,
        num_output_seconds: float,
        one_bit_seconds: float,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.mus_dataset = mus_dataset
        self.current_sample_rate = current_sample_rate
        self.target_sample_rate = target_sample_rate
        self.num_input_seconds = num_input_seconds
        self.num_output_seconds = num_output_seconds
        self.one_bit_seconds = one_bit_seconds
        self.num_bits_in_second = 1 / self.one_bit_seconds  # Avoid rounding error
        self.device = device
        
        self.resampler = T.Resample(self.current_sample_rate, self.target_sample_rate)
        
        self.durations = list(map(lambda track: track.duration, self.mus_dataset))
        self.num_possible_samples_from_track = list(
            map(
                lambda duration: int(
                    (duration - self.num_input_seconds - self.num_output_seconds)
                    * self.num_bits_in_second
                ) + 1,
                self.durations
            )
        )
        
        self.start_seconds = []
        for idx, num_samples in enumerate(self.num_possible_samples_from_track):
            self.start_seconds.extend([
                (idx, i / self.num_bits_in_second)
                for i in range(int(num_samples))
            ])
        
        self.cached_tracks = {}
        self.cached_parts = {}
        
        self.expected_shape_X = int(self.num_input_seconds * self.target_sample_rate)
        self.expected_shape_y = int(self.num_output_seconds * self.target_sample_rate)
    
    def _change_sample_rate(self, array, current_sample_rate, target_sample_rate):
        """Change sample rate of audio array."""
        array = torch.tensor(array, dtype=torch.float32)
        
        if current_sample_rate != target_sample_rate:
            array = self.resampler(array)
        
        return array.to(self.device)
    
    def __len__(self):
        return sum(self.num_possible_samples_from_track)
    
    def _pad(self, item, expectation):
        """Pad item to expected shape."""
        return F.pad(item, (0, expectation - item.shape[0]))
    
    def _prepare_track(self, track_num, start):
        """Prepare track data for a given start time."""
        if self.cached_tracks.get(track_num, 'no') == 'no':
            track = self.mus_dataset[track_num]
            
            vocals = self._change_sample_rate(
                track.targets['vocals'].audio.mean(axis=-1),
                self.current_sample_rate,
                self.target_sample_rate
            )
            drums = self._change_sample_rate(
                track.targets['drums'].audio.mean(axis=-1),
                self.current_sample_rate,
                self.target_sample_rate
            )
            bass = self._change_sample_rate(
                track.targets['bass'].audio.mean(axis=-1),
                self.current_sample_rate,
                self.target_sample_rate
            )
            other = self._change_sample_rate(
                track.targets['other'].audio.mean(axis=-1),
                self.current_sample_rate,
                self.target_sample_rate
            )
            all_audio = self._change_sample_rate(
                track.audio.mean(axis=-1),
                self.current_sample_rate,
                self.target_sample_rate
            )
            
            self.cached_tracks[track_num] = {
                'vocals': vocals,
                'drums': drums,
                'bass': bass,
                'other': other,
                'all': all_audio
            }
        
        track = self.cached_tracks[track_num]
        vocals = track['vocals']
        drums = track['drums']
        bass = track['bass']
        other = track['other']
        all_audio = track['all']
        
        end_X = start + self.num_input_seconds
        end_y = end_X + self.num_output_seconds
        
        start_X = int(start * self.target_sample_rate)
        end_X = int(end_X * self.target_sample_rate)
        end_y = int(end_y * self.target_sample_rate)
        
        vocals_X = self._pad(vocals[start_X:end_X], self.expected_shape_X)
        vocals_y = self._pad(vocals[end_X:end_y], self.expected_shape_y)
        
        drums_X = self._pad(drums[start_X:end_X], self.expected_shape_X)
        drums_y = self._pad(drums[end_X:end_y], self.expected_shape_y)
        
        bass_X = self._pad(bass[start_X:end_X], self.expected_shape_X)
        bass_y = self._pad(bass[end_X:end_y], self.expected_shape_y)
        
        other_X = self._pad(other[start_X:end_X], self.expected_shape_X)
        other_y = self._pad(other[end_X:end_y], self.expected_shape_y)
        
        all_X = self._pad(all_audio[start_X:end_X], self.expected_shape_X)
        all_y = self._pad(all_audio[end_X:end_y], self.expected_shape_y)
        
        return {
            'vocals': (vocals_X[:self.expected_shape_X], vocals_y[:self.expected_shape_y]),
            'drums': (drums_X[:self.expected_shape_X], drums_y[:self.expected_shape_y]),
            'bass': (bass_X[:self.expected_shape_X], bass_y[:self.expected_shape_y]),
            'other': (other_X[:self.expected_shape_X], other_y[:self.expected_shape_y]),
            'all': (all_X[:self.expected_shape_X], all_y[:self.expected_shape_y])
        }
    
    def __getitem__(self, idx):
        track_num, start_timestep = self.start_seconds[idx]
        preprocessed_data = self._prepare_track(track_num, start_timestep)
        return preprocessed_data


class MusDBDatasetCached(Dataset):
    """
    Dataset for cached MUSDB data (pickle files).
    
    Loads preprocessed audio tracks from pickle files.
    """
    
    def __init__(
        self,
        mus_dataset_path: str,
        current_sample_rate: int,
        target_sample_rate: int,
        num_input_seconds: float,
        num_output_seconds: float,
        one_bit_seconds: float,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.mus_dataset_path = mus_dataset_path
        self.mus_dataset_filenames = os.listdir(mus_dataset_path)
        self.current_sample_rate = current_sample_rate
        self.target_sample_rate = target_sample_rate
        self.num_input_seconds = num_input_seconds
        self.num_output_seconds = num_output_seconds
        self.one_bit_seconds = one_bit_seconds
        self.num_bits_in_second = 1 / self.one_bit_seconds
        self.device = device
        
        self.durations = list(
            map(
                lambda filename: float(filename.split('.')[-3] + '.' + filename.split('.')[-2]),
                self.mus_dataset_filenames
            )
        )
        self.num_possible_samples_from_track = list(
            map(
                lambda duration: int(
                    (duration - self.num_input_seconds - self.num_output_seconds)
                    * self.num_bits_in_second
                ) + 1,
                self.durations
            )
        )
        
        self.start_seconds = []
        for idx, num_samples in enumerate(self.num_possible_samples_from_track):
            self.start_seconds.extend([
                (idx, i / self.num_bits_in_second)
                for i in range(int(num_samples))
            ])
        
        self.cached_parts = {}
        
        self.expected_shape_X = int(self.num_input_seconds * self.target_sample_rate)
        self.expected_shape_y = int(self.num_output_seconds * self.target_sample_rate)
    
    def __len__(self):
        return sum(self.num_possible_samples_from_track)
    
    def _pad(self, item, expectation):
        """Pad item to expected shape."""
        return F.pad(item, (0, expectation - item.shape[0]))
    
    def _prepare_track(self, track_num, start):
        """Prepare track data from cached pickle file."""
        with open(
            os.path.join(self.mus_dataset_path, self.mus_dataset_filenames[track_num]),
            'rb'
        ) as f:
            track = joblib.load(f)
        
        vocals = torch.tensor(track['vocals'], device=self.device)
        drums = torch.tensor(track['drums'], device=self.device)
        bass = torch.tensor(track['bass'], device=self.device)
        other = torch.tensor(track['other'], device=self.device)
        all_audio = torch.tensor(track['all'], device=self.device)
        
        end_X = start + self.num_input_seconds
        end_y = end_X + self.num_output_seconds
        
        start_X = int(start * self.target_sample_rate)
        end_X = int(end_X * self.target_sample_rate)
        end_y = int(end_y * self.target_sample_rate)
        
        vocals_X = self._pad(vocals[start_X:end_X], self.expected_shape_X)
        vocals_y = self._pad(vocals[end_X:end_y], self.expected_shape_y)
        
        drums_X = self._pad(drums[start_X:end_X], self.expected_shape_X)
        drums_y = self._pad(drums[end_X:end_y], self.expected_shape_y)
        
        bass_X = self._pad(bass[start_X:end_X], self.expected_shape_X)
        bass_y = self._pad(bass[end_X:end_y], self.expected_shape_y)
        
        other_X = self._pad(other[start_X:end_X], self.expected_shape_X)
        other_y = self._pad(other[end_X:end_y], self.expected_shape_y)
        
        all_X = self._pad(all_audio[start_X:end_X], self.expected_shape_X)
        all_y = self._pad(all_audio[end_X:end_y], self.expected_shape_y)
        
        return {
            'vocals': (vocals_X[:self.expected_shape_X], vocals_y[:self.expected_shape_y]),
            'drums': (drums_X[:self.expected_shape_X], drums_y[:self.expected_shape_y]),
            'bass': (bass_X[:self.expected_shape_X], bass_y[:self.expected_shape_y]),
            'other': (other_X[:self.expected_shape_X], other_y[:self.expected_shape_y]),
            'all': (all_X[:self.expected_shape_X], all_y[:self.expected_shape_y])
        }
    
    def __getitem__(self, idx):
        track_num, start_timestep = self.start_seconds[idx]
        preprocessed_data = self._prepare_track(track_num, start_timestep)
        return preprocessed_data

