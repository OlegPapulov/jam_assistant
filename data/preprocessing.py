"""
Data preprocessing functions for MUSDB dataset.
"""

import os
import numpy as np
import joblib
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import musdb
import stempeg


@dataclass
class Stem:
    """Stem audio data."""
    audio: Optional[np.ndarray] = None


@dataclass
class Track:
    """Track audio data with stems."""
    audio: Optional[np.ndarray] = None
    targets: dict = field(default_factory=lambda: {
        'vocals': Stem(),
        'drums': Stem(),
        'bass': Stem(),
        'other': Stem()
    })


def download_musdb(sample_rate: int = 22050, use_7s: bool = True):
    """
    Download MUSDB dataset.
    
    Args:
        sample_rate: Target sample rate
        use_7s: If True, download 7s sample dataset; otherwise use full dataset
    
    Returns:
        MUSDB dataset object
    """
    if use_7s:
        mus = musdb.DB(download=True, sample_rate=sample_rate)
        return mus
    else:
        # For full dataset, user needs to download manually
        # This is a placeholder - actual implementation would need gdown
        raise NotImplementedError(
            "Full MUSDB dataset download not implemented. "
            "Please download manually or use use_7s=True"
        )


def extract_stems(
    input_folder: str,
    output_folder: str,
    sample_rate: int = 22050,
    stems_map: Optional[dict] = None
):
    """
    Extract stems from MUSDB files and save as pickle files.
    
    Args:
        input_folder: Folder containing MUSDB stem files
        output_folder: Folder to save extracted pickle files
        sample_rate: Target sample rate
        stems_map: Mapping of stem IDs to names (default: {0: 'mixture', 1: 'drums', 2: 'bass', 3: 'other', 4: 'vocals'})
    """
    if stems_map is None:
        stems_map = {0: 'mixture', 1: 'drums', 2: 'bass', 3: 'other', 4: 'vocals'}
    
    os.makedirs(output_folder, exist_ok=True)
    
    filenames = [f for f in os.listdir(input_folder) if not f.startswith('_') and not f.startswith('.')]
    
    for filename in tqdm(filenames, desc="Extracting stems"):
        try:
            track = Track()
            
            for i in range(5):
                audio, _ = stempeg.read_stems(
                    os.path.join(input_folder, filename),
                    stem_id=i,
                    sample_rate=sample_rate,
                    ffmpeg_format="s16le"
                )
                
                audio = audio.astype('float16')
                if audio.shape[-1] == 2:
                    audio = audio.mean(-1).reshape(-1, 1)
                
                if i == 0:
                    track.audio = audio
                else:
                    track.targets[stems_map[i]].audio = audio
            
            track.duration = track.audio.shape[0] / sample_rate
            
            # Extract stems
            vocals = track.targets['vocals'].audio.mean(axis=-1) if track.targets['vocals'].audio is not None else None
            drums = track.targets['drums'].audio.mean(axis=-1) if track.targets['drums'].audio is not None else None
            bass = track.targets['bass'].audio.mean(axis=-1) if track.targets['bass'].audio is not None else None
            other = track.targets['other'].audio.mean(axis=-1) if track.targets['other'].audio is not None else None
            all_audio = track.audio.mean(axis=-1) if track.audio is not None else None
            
            # Save as pickle
            output_filename = f"{filename}.{round(track.duration, 2)}.pkl"
            with open(os.path.join(output_folder, output_filename), 'wb') as f:
                joblib.dump({
                    'vocals': vocals,
                    'drums': drums,
                    'bass': bass,
                    'other': other,
                    'all': all_audio
                }, f)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue


def prepare_data(config):
    """
    Prepare data based on configuration.
    
    Args:
        config: Config object with data preparation parameters
    
    Returns:
        Tuple of (train_dataset, test_dataset) or (train_path, test_path) for cached data
    """
    if config.use_7s:
        # Download and use MUSDB 7s dataset
        mus = download_musdb(config.target_sample_rate, use_7s=True)
        return mus
    else:
        # Use cached pickle files
        train_path = './musdb18/train_extracted'
        test_path = './musdb18/test_extracted'
        
        # Check if paths exist, if not, extract
        if not os.path.exists(train_path) or not os.listdir(train_path):
            print("Extracting training stems...")
            extract_stems('./musdb18/train', train_path, config.target_sample_rate)
        
        if not os.path.exists(test_path) or not os.listdir(test_path):
            print("Extracting test stems...")
            extract_stems('./musdb18/test', test_path, config.target_sample_rate)
        
        return train_path, test_path

