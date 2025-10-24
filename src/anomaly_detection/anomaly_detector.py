"""
Anomaly Detection Module
Detects unusual or suspicious signal patterns
"""

import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


class AnomalyDetector:
    """
    Detects anomalous signals using multiple methods.
    
    Used to identify signals that don't fit normal patterns - 
    could indicate jamming, interference, or unknown signal types.
    """
    
    def __init__(self, contamination=0.1):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.1 = 10%)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.models = {}
        self.is_fitted = False
        self.col_means = None
        
    def _handle_nan(self, X, fit=False):
        """
        Handle NaN values in feature matrix.
        
        Args:
            X: Feature matrix
            fit: If True, calculate and store column means
            
        Returns:
            Feature matrix with NaN values replaced
        """
        X = X.copy()
        nan_mask = np.isnan(X)
        
        if np.any(nan_mask):
            if fit:
                # Calculate and store column means
                self.col_means = np.nanmean(X, axis=0)
                # If entire column is NaN, use 0
                self.col_means[np.isnan(self.col_means)] = 0
            
            # Replace NaN with column means
            nan_indices = np.where(nan_mask)
            for col in np.unique(nan_indices[1]):
                col_mask = nan_indices[1] == col
                X[nan_indices[0][col_mask], col] = self.col_means[col]
        
        return X
    
    def fit(self, X_normal):
        """
        Fit anomaly detectors on normal data.
        
        Args:
            X_normal: Feature matrix of normal signals
        """
        print("Training anomaly detectors...")
        
        # Handle NaN values
        print("  Checking for NaN values...")
        nan_count = np.sum(np.isnan(X_normal))
        if nan_count > 0:
            print(f"  Found {nan_count} NaN values, replacing with column means...")
        
        X_clean = self._handle_nan(X_normal, fit=True)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Isolation Forest
        print("  Training Isolation Forest...")
        self.models['isolation_forest'] = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.models['isolation_forest'].fit(X_scaled)
        
        # One-Class SVM
        print("  Training One-Class SVM...")
        self.models['one_class_svm'] = OneClassSVM(
            gamma='auto',
            nu=self.contamination
        )
        self.models['one_class_svm'].fit(X_scaled)
        
        self.is_fitted = True
        print("✓ Anomaly detectors trained!")
        
    def predict(self, X, method='isolation_forest'):
        """
        Predict if signals are anomalous.
        
        Args:
            X: Feature matrix to check
            method: Detection method ('isolation_forest' or 'one_class_svm')
            
        Returns:
            Array of predictions (1 = normal, -1 = anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first!")
        
        # Handle NaN values
        X_clean = self._handle_nan(X, fit=False)
        
        X_scaled = self.scaler.transform(X_clean)
        predictions = self.models[method].predict(X_scaled)
        
        return predictions
    
    def predict_all_methods(self, X):
        """
        Predict using all methods and combine results.
        
        Args:
            X: Feature matrix to check
            
        Returns:
            Dictionary with predictions from each method
        """
        results = {}
        
        for method in self.models.keys():
            predictions = self.predict(X, method=method)
            results[method] = predictions
        
        # Ensemble: signal is anomaly if ANY method flags it
        results['ensemble'] = np.where(
            (results['isolation_forest'] == -1) | (results['one_class_svm'] == -1),
            -1, 1
        )
        
        return results
    
    def score_samples(self, X, method='isolation_forest'):
        """
        Get anomaly scores (more negative = more anomalous).
        
        Args:
            X: Feature matrix
            method: Detection method
            
        Returns:
            Array of anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first!")
        
        # Handle NaN values
        X_clean = self._handle_nan(X, fit=False)
        
        X_scaled = self.scaler.transform(X_clean)
        
        if method == 'isolation_forest':
            scores = self.models[method].score_samples(X_scaled)
        elif method == 'one_class_svm':
            scores = self.models[method].decision_function(X_scaled)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return scores
    
    def analyze_dataset(self, X, labels=None):
        """
        Comprehensive anomaly analysis of a dataset.
        
        Args:
            X: Feature matrix
            labels: Signal labels (optional)
            
        Returns:
            Dictionary with analysis results
        """
        print("\n" + "="*60)
        print("ANOMALY DETECTION ANALYSIS")
        print("="*60)
        
        # Get predictions from all methods
        predictions = self.predict_all_methods(X)
        
        # Calculate statistics
        results = {}
        
        for method, preds in predictions.items():
            n_anomalies = np.sum(preds == -1)
            anomaly_rate = n_anomalies / len(preds) * 100
            
            results[method] = {
                'n_anomalies': n_anomalies,
                'anomaly_rate': anomaly_rate,
                'predictions': preds
            }
            
            print(f"\n{method.replace('_', ' ').title()}:")
            print(f"  Anomalies detected: {n_anomalies}/{len(preds)} ({anomaly_rate:.1f}%)")
            
            # If labels provided, show which signal types are flagged
            if labels is not None:
                anomaly_idx = preds == -1
                if np.sum(anomaly_idx) > 0:
                    anomaly_labels = labels[anomaly_idx]
                    unique, counts = np.unique(anomaly_labels, return_counts=True)
                    print("  Anomalies by signal type:")
                    for label, count in zip(unique, counts):
                        print(f"    {label}: {count}")
        
        return results
    
    def visualize_anomalies(self, X, predictions, labels=None, 
                           method='isolation_forest', output_dir='results'):
        """
        Visualize anomaly detection results using PCA.
        
        Args:
            X: Feature matrix
            predictions: Anomaly predictions
            labels: Signal labels (optional)
            method: Detection method name
            output_dir: Directory to save plot
        """
        print(f"\nCreating visualization for {method}...")
        
        # Handle NaN and scale
        X_clean = self._handle_nan(X, fit=False)
        X_scaled = self.scaler.transform(X_clean)
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot normal vs anomalous points
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        
        ax.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], 
                  c='blue', alpha=0.6, s=30, label='Normal', edgecolors='k', linewidth=0.5)
        ax.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1], 
                  c='red', alpha=0.8, s=100, label='Anomaly', marker='X', edgecolors='k', linewidth=1)
        
        ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title(f'Anomaly Detection - {method.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'anomaly_detection_{method}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to: {save_path}")
    
    def save(self, filepath='models/trained/anomaly_detector.pkl'):
        """
        Save the trained anomaly detector.
        
        Args:
            filepath: Path to save the detector
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"✓ Anomaly detector saved to: {filepath}")
    
    @staticmethod
    def load(filepath='models/trained/anomaly_detector.pkl'):
        """
        Load a trained anomaly detector.
        
        Args:
            filepath: Path to the saved detector
            
        Returns:
            Loaded AnomalyDetector object
        """
        with open(filepath, 'rb') as f:
            detector = pickle.load(f)
        
        print(f"✓ Anomaly detector loaded from: {filepath}")
        return detector


if __name__ == "__main__":
    print("Anomaly Detector Module")
    print("This module should be run from detect_anomalies.py")