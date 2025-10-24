"""
Feature Extraction Script
Loads dataset and extracts features for ML training
"""

import numpy as np
import pickle
import os
from feature_extractor import FeatureExtractor
from datetime import datetime


def load_dataset(dataset_path='data/raw/signal_dataset_latest.pkl'):
    """Load the signal dataset"""
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"✓ Dataset loaded: {dataset['signals'].shape[0]} signals")
    return dataset


def extract_and_save_features(dataset_path='data/raw/signal_dataset_latest.pkl',
                              output_dir='data/processed'):
    """
    Extract features from dataset and save.
    
    Args:
        dataset_path: Path to signal dataset
        output_dir: Directory to save processed features
    """
    print("="*60)
    print("FEATURE EXTRACTION")
    print("="*60)
    
    # Load dataset
    dataset = load_dataset(dataset_path)
    
    signals = dataset['signals']
    labels = dataset['labels']
    sample_rate = dataset['sample_rate']
    
    # Initialize feature extractor
    extractor = FeatureExtractor(sample_rate=sample_rate)
    
    # Extract features
    print("\nExtracting features...")
    feature_matrix, feature_names = extractor.extract_features_batch(signals)
    
    # Create processed dataset
    processed_dataset = {
        'features': feature_matrix,
        'labels': labels,
        'feature_names': feature_names,
        'sample_rate': sample_rate,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'original_signals_shape': signals.shape
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"processed_features_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    print(f"\nSaving processed features to: {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(processed_dataset, f)
    
    # Save latest version
    latest_path = os.path.join(output_dir, 'processed_features_latest.pkl')
    with open(latest_path, 'wb') as f:
        pickle.dump(processed_dataset, f)
    
    print(f"✓ Features saved!")
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total signals processed: {len(signals)}")
    print(f"Total features extracted: {len(feature_names)}")
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
    
    print("\nFeature categories:")
    time_features = [f for f in feature_names if any(x in f for x in ['mean', 'std', 'variance', 'rms', 'energy', 'zero_crossing'])]
    freq_features = [f for f in feature_names if 'spectral' in f or 'frequency' in f or 'bandwidth' in f]
    wavelet_features = [f for f in feature_names if 'wavelet' in f]
    spectrogram_features = [f for f in feature_names if 'spectrogram' in f or 'temporal' in f]
    
    print(f"  Time-domain: {len(time_features)}")
    print(f"  Frequency-domain: {len(freq_features)}")
    print(f"  Wavelet: {len(wavelet_features)}")
    print(f"  Spectrogram: {len(spectrogram_features)}")
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*60)
    
    return filepath


if __name__ == "__main__":
    try:
        extract_and_save_features()
    except FileNotFoundError:
        print("❌ Dataset not found!")
        print("Please run generate_dataset.py first.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()