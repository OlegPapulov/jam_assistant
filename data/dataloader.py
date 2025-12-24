"""
DataLoader creation utilities.
"""

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple
import musdb

from data.dataset import MusDBDataset, MusDBDatasetCached
from config.config import Config


def create_dataloaders(
    config: Config,
    device: str = 'cpu'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders based on configuration.
    
    Args:
        config: Configuration object
        device: Device to use for data loading
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if config.use_7s:
        # Use MUSDB 7s dataset
        mus = musdb.DB(download=True, sample_rate=config.target_sample_rate)
        mus_train, mus_test = train_test_split(
            mus,
            test_size=config.test_size,
            random_state=config.random_seed
        )
        
        train_dataset = MusDBDataset(
            mus_train,
            current_sample_rate=config.target_sample_rate,
            target_sample_rate=config.target_sample_rate,
            num_input_seconds=config.num_input_seconds,
            num_output_seconds=config.num_output_seconds,
            one_bit_seconds=config.one_bit_seconds,
            device=device
        )
        
        test_dataset = MusDBDataset(
            mus_test,
            current_sample_rate=config.target_sample_rate,
            target_sample_rate=config.target_sample_rate,
            num_input_seconds=config.num_input_seconds,
            num_output_seconds=config.num_output_seconds,
            one_bit_seconds=config.one_bit_seconds,
            device=device
        )
    else:
        # Use cached pickle files
        train_dataset = MusDBDatasetCached(
            './musdb18/train_extracted',
            current_sample_rate=config.target_sample_rate,
            target_sample_rate=config.target_sample_rate,
            num_input_seconds=config.num_input_seconds,
            num_output_seconds=config.num_output_seconds,
            one_bit_seconds=config.one_bit_seconds,
            device=device
        )
        
        test_dataset = MusDBDatasetCached(
            './musdb18/test_extracted',
            current_sample_rate=config.target_sample_rate,
            target_sample_rate=config.target_sample_rate,
            num_input_seconds=config.num_input_seconds,
            num_output_seconds=config.num_output_seconds,
            one_bit_seconds=config.one_bit_seconds,
            device=device
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    return train_loader, test_loader

