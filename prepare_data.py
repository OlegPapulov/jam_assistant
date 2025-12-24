#!/usr/bin/env python3
"""
Data preparation script.

Downloads and preprocesses MUSDB dataset.
"""

import argparse
from config.config import Config, parse_args
from data.preprocessing import prepare_data, download_musdb, extract_stems


def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description='Prepare MUSDB dataset')
    parser.add_argument('--use_7s', action='store_true', default=True,
                       help='Use MUSDB 7s dataset')
    parser.add_argument('--no_use_7s', dest='use_7s', action='store_false',
                       help='Use full MUSDB dataset')
    parser.add_argument('--target_sample_rate', type=int, default=22050,
                       help='Target sample rate')
    parser.add_argument('--extract_only', type=str, default=None,
                       help='Only extract stems from folder (provide input folder)')
    parser.add_argument('--output_folder', type=str, default=None,
                       help='Output folder for extracted stems')
    args = parser.parse_args()
    
    if args.extract_only:
        # Extract stems only
        if args.output_folder is None:
            print("Error: --output_folder required when using --extract_only")
            return
        
        print(f"Extracting stems from {args.extract_only} to {args.output_folder}")
        extract_stems(
            input_folder=args.extract_only,
            output_folder=args.output_folder,
            sample_rate=args.target_sample_rate
        )
        print("Extraction completed!")
    else:
        # Full data preparation
        config = Config()
        config.use_7s = args.use_7s
        config.target_sample_rate = args.target_sample_rate
        
        print("Preparing data...")
        result = prepare_data(config)
        
        if config.use_7s:
            print(f"MUSDB 7s dataset ready: {len(result)} tracks")
        else:
            train_path, test_path = result
            print(f"Data prepared:")
            print(f"  Train: {train_path}")
            print(f"  Test: {test_path}")
        
        print("Data preparation completed!")


if __name__ == '__main__':
    main()

