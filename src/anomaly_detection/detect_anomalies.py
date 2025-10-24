"""
Anomaly Detection Script
Trains anomaly detectors and identifies unusual signals
"""

import numpy as np
import pickle
from anomaly_detector import AnomalyDetector
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def load_processed_features(filepath='data/processed/processed_features_latest.pkl'):
    """Load processed feature data"""
    print(f"Loading features from: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    labels = data['labels']
    
    print(f"✓ Loaded {len(features)} signals")
    return features, labels


def main():
    """Main anomaly detection pipeline"""
    
    print("="*60)
    print("ANOMALY DETECTION SYSTEM")
    print("="*60)
    
    # Load data
    features, labels = load_processed_features()
    
    # Train on all data (assuming most are normal)
    # In a real scenario, you'd train only on known normal signals
    print("\nTraining anomaly detectors on signal features...")
    
    detector = AnomalyDetector(contamination=0.1)
    detector.fit(features)
    
    # Analyze the dataset
    results = detector.analyze_dataset(features, labels=labels)
    
    # Visualize results for each method
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    for method in ['isolation_forest', 'one_class_svm', 'ensemble']:
        predictions = results[method]['predictions']
        detector.visualize_anomalies(
            features, 
            predictions, 
            labels=labels,
            method=method
        )
    
    # Save detector
    print("\n" + "="*60)
    print("SAVING ANOMALY DETECTOR")
    print("="*60)
    
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'models/trained/anomaly_detector_{timestamp}.pkl'
    detector.save(save_path)
    
    # Also save as latest
    detector.save('models/trained/anomaly_detector_latest.pkl')
    
    print("\n" + "="*60)
    print("ANOMALY DETECTION COMPLETE!")
    print("="*60)
    print("\nResults:")
    print("  - Visualizations saved to: results/")
    print("  - Detector saved to: models/trained/")
    
    # Summary
    print("\nSummary:")
    for method, result in results.items():
        print(f"  {method.replace('_', ' ').title()}: {result['n_anomalies']} anomalies ({result['anomaly_rate']:.1f}%)")
    
    print("\nThe anomaly detector can now be used to:")
    print("  - Identify suspicious or unusual signals")
    print("  - Detect jamming or interference")
    print("  - Flag unknown signal types for investigation")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run extract_features.py first!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()