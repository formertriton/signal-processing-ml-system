"""
Dataset Generation Script
Generates a complete dataset of RF signals for training
"""

import numpy as np
import os
import yaml
from signal_generator import SignalGenerator
import pickle
from datetime import datetime


def load_config():
    """Load configuration from yaml file"""
    config_path = os.path.join('config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_complete_dataset(config):
    """
    Generate complete dataset with all signal types.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing all signals and metadata
    """
    print("="*60)
    print("GENERATING SIGNAL DATASET")
    print("="*60)
    
    # Get parameters from config
    sample_rate = config['signal_generation']['sample_rate']
    duration = config['signal_generation']['duration']
    num_samples = config['signal_generation']['num_samples']
    signal_types = config['signal_generation']['signal_types']
    
    # Initialize generator
    generator = SignalGenerator(sample_rate=sample_rate, duration=duration)
    
    all_signals = []
    all_labels = []
    
    # Generate signals for each type
    for signal_type in signal_types:
        print(f"\nGenerating {num_samples} {signal_type.upper()} signals...")
        
        batch = generator.generate_signal_batch(
            signal_type=signal_type,
            num_samples=num_samples,
            add_noise_flag=True,
            snr_range=(10, 30)
        )
        
        all_signals.append(batch['signals'])
        all_labels.extend(batch['labels'])
        
        print(f"  ✓ Generated {len(batch['signals'])} signals")
        print(f"  ✓ Shape: {batch['signals'].shape}")
    
    # Combine all signals
    all_signals = np.vstack(all_signals)
    all_labels = np.array(all_labels)
    
    # Create dataset dictionary
    dataset = {
        'signals': all_signals,
        'labels': all_labels,
        'sample_rate': sample_rate,
        'duration': duration,
        'num_samples_per_type': num_samples,
        'signal_types': signal_types,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total signals: {len(all_signals)}")
    print(f"Signal types: {len(signal_types)}")
    print(f"Samples per type: {num_samples}")
    print(f"Signal shape: {all_signals.shape}")
    print(f"Sample rate: {sample_rate:,} Hz")
    print(f"Duration: {duration*1000} ms")
    
    # Print label distribution
    print("\nLabel Distribution:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")
    
    return dataset


def save_dataset(dataset, output_dir='data/raw'):
    """
    Save dataset to disk.
    
    Args:
        dataset: Dataset dictionary
        output_dir: Directory to save to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"signal_dataset_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # Save using pickle
    print(f"\nSaving dataset to: {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"✓ Dataset saved successfully!")
    print(f"  File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
    
    # Also save a latest version for easy access
    latest_path = os.path.join(output_dir, 'signal_dataset_latest.pkl')
    with open(latest_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"✓ Latest version saved to: {latest_path}")
    
    return filepath


def main():
    """Main execution function"""
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config()
        print("✓ Configuration loaded\n")
        
        # Generate dataset
        dataset = generate_complete_dataset(config)
        
        # Save dataset
        filepath = save_dataset(dataset)
        
        print("\n" + "="*60)
        print("DATASET GENERATION COMPLETE!")
        print("="*60)
        print(f"\nDataset saved to: {filepath}")
        print("\nYou can now use this dataset for:")
        print("  1. Feature extraction")
        print("  2. ML model training")
        print("  3. Visualization")
        print("  4. Analysis")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()