"""
Feature Extraction Module
Extracts time-domain, frequency-domain, and time-frequency features from signals
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats
import pywt


class FeatureExtractor:
    """
    Extracts features from RF signals for machine learning.
    
    Features help ML models understand and classify signals by capturing
    important characteristics in different domains.
    """
    
    def __init__(self, sample_rate: int = 1000000):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate
    
    # ==================== TIME DOMAIN FEATURES ====================
    
    def extract_time_features(self, signal_data: np.ndarray) -> dict:
        """
        Extract statistical features from time domain.
        
        Args:
            signal_data: Input signal array
            
        Returns:
            Dictionary of time-domain features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['variance'] = np.var(signal_data)
        features['max'] = np.max(signal_data)
        features['min'] = np.min(signal_data)
        features['peak_to_peak'] = np.ptp(signal_data)
        
        # Signal energy and power
        features['energy'] = np.sum(signal_data ** 2)
        features['rms'] = np.sqrt(np.mean(signal_data ** 2))
        features['peak_amplitude'] = np.max(np.abs(signal_data))
        
        # Peak-to-Average Ratio (PAR)
        avg_power = np.mean(signal_data ** 2)
        peak_power = np.max(signal_data ** 2)
        features['par'] = peak_power / (avg_power + 1e-10)
        
        # Zero crossing rate (how often signal crosses zero)
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
        
        # Higher order statistics
        features['skewness'] = stats.skew(signal_data)
        features['kurtosis'] = stats.kurtosis(signal_data)
        
        # Crest factor (peak vs RMS)
        features['crest_factor'] = features['peak_amplitude'] / (features['rms'] + 1e-10)
        
        return features
    
    # ==================== FREQUENCY DOMAIN FEATURES ====================
    
    def extract_frequency_features(self, signal_data: np.ndarray) -> dict:
        """
        Extract features from frequency domain (using FFT).
        
        Args:
            signal_data: Input signal array
            
        Returns:
            Dictionary of frequency-domain features
        """
        features = {}
        
        # Compute FFT
        fft = np.fft.fft(signal_data)
        frequencies = np.fft.fftfreq(len(signal_data), 1/self.sample_rate)
        
        # Only positive frequencies
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        magnitude = np.abs(fft[positive_freq_idx])
        power = magnitude ** 2
        
        # Normalize
        power_norm = power / (np.sum(power) + 1e-10)
        
        # Spectral centroid (center of mass of spectrum)
        features['spectral_centroid'] = np.sum(frequencies * power_norm)
        
        # Spectral bandwidth (spread of spectrum)
        features['spectral_bandwidth'] = np.sqrt(
            np.sum(((frequencies - features['spectral_centroid']) ** 2) * power_norm)
        )
        
        # Dominant frequency
        features['dominant_frequency'] = frequencies[np.argmax(magnitude)]
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum = np.cumsum(power_norm)
        rolloff_idx = np.where(cumsum >= 0.85)[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = frequencies[rolloff_idx[0]]
        else:
            features['spectral_rolloff'] = frequencies[-1]
        
        # Spectral flatness (how noise-like vs tone-like)
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
        arithmetic_mean = np.mean(magnitude)
        features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Spectral entropy (randomness of spectrum)
        features['spectral_entropy'] = -np.sum(power_norm * np.log2(power_norm + 1e-10))
        
        # Peak to mean ratio in frequency domain
        features['freq_peak_to_mean'] = np.max(magnitude) / (np.mean(magnitude) + 1e-10)
        
        # Bandwidth (frequency range with significant energy)
        threshold = 0.1 * np.max(magnitude)
        significant_freqs = frequencies[magnitude > threshold]
        if len(significant_freqs) > 0:
            features['bandwidth'] = np.max(significant_freqs) - np.min(significant_freqs)
        else:
            features['bandwidth'] = 0
        
        return features
    
    # ==================== TIME-FREQUENCY FEATURES ====================
    
    def extract_wavelet_features(self, signal_data: np.ndarray, 
                                 wavelet: str = 'db4', 
                                 level: int = 5) -> dict:
        """
        Extract features using wavelet transform.
        Wavelets capture both time and frequency information.
        
        Args:
            signal_data: Input signal array
            wavelet: Wavelet type (db4 = Daubechies 4)
            level: Decomposition level
            
        Returns:
            Dictionary of wavelet features
        """
        features = {}
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # Extract features from each level
        for i, coeff in enumerate(coeffs):
            prefix = f'wavelet_level_{i}'
            features[f'{prefix}_energy'] = np.sum(coeff ** 2)
            features[f'{prefix}_mean'] = np.mean(np.abs(coeff))
            features[f'{prefix}_std'] = np.std(coeff)
            features[f'{prefix}_max'] = np.max(np.abs(coeff))
        
        # Total wavelet energy
        features['total_wavelet_energy'] = sum([np.sum(c ** 2) for c in coeffs])
        
        # Energy distribution across levels
        energies = [np.sum(c ** 2) for c in coeffs]
        total_energy = sum(energies) + 1e-10
        for i, energy in enumerate(energies):
            features[f'wavelet_energy_ratio_level_{i}'] = energy / total_energy
        
        return features
    
    def extract_spectrogram_features(self, signal_data: np.ndarray,
                                    nperseg: int = 256) -> dict:
        """
        Extract features from spectrogram (STFT).
        
        Args:
            signal_data: Input signal array
            nperseg: Length of each segment for STFT
            
        Returns:
            Dictionary of spectrogram features
        """
        features = {}
        
        # Compute spectrogram
        frequencies, times, Sxx = scipy_signal.spectrogram(
            signal_data,
            self.sample_rate,
            nperseg=nperseg
        )
        
        # Flatten spectrogram for statistical features
        Sxx_flat = Sxx.flatten()
        
        features['spectrogram_mean'] = np.mean(Sxx_flat)
        features['spectrogram_std'] = np.std(Sxx_flat)
        features['spectrogram_max'] = np.max(Sxx_flat)
        features['spectrogram_energy'] = np.sum(Sxx_flat)
        
        # Temporal variation (how much spectrum changes over time)
        features['temporal_variation'] = np.mean(np.std(Sxx, axis=1))
        
        # Frequency variation (how much time signal varies across frequencies)
        features['frequency_variation'] = np.mean(np.std(Sxx, axis=0))
        
        return features
    
    # ==================== COMBINED FEATURE EXTRACTION ====================
    
    def extract_all_features(self, signal_data: np.ndarray) -> dict:
        """
        Extract all features from a signal.
        
        Args:
            signal_data: Input signal array
            
        Returns:
            Dictionary containing all features
        """
        all_features = {}
        
        # Time domain features
        time_features = self.extract_time_features(signal_data)
        all_features.update(time_features)
        
        # Frequency domain features
        freq_features = self.extract_frequency_features(signal_data)
        all_features.update(freq_features)
        
        # Wavelet features
        wavelet_features = self.extract_wavelet_features(signal_data)
        all_features.update(wavelet_features)
        
        # Spectrogram features
        spectrogram_features = self.extract_spectrogram_features(signal_data)
        all_features.update(spectrogram_features)
        
        return all_features
    
    def extract_features_batch(self, signals: np.ndarray) -> np.ndarray:
        """
        Extract features from multiple signals.
        
        Args:
            signals: Array of signals (shape: [n_signals, n_samples])
            
        Returns:
            Feature matrix (shape: [n_signals, n_features])
        """
        print(f"Extracting features from {len(signals)} signals...")
        
        feature_list = []
        
        for i, signal_data in enumerate(signals):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(signals)} signals...")
            
            features = self.extract_all_features(signal_data)
            feature_list.append(features)
        
        # Convert list of dicts to numpy array
        feature_names = list(feature_list[0].keys())
        feature_matrix = np.array([[f[name] for name in feature_names] for f in feature_list])
        
        print(f"✓ Feature extraction complete!")
        print(f"  Feature matrix shape: {feature_matrix.shape}")
        print(f"  Number of features: {len(feature_names)}")
        
        return feature_matrix, feature_names


if __name__ == "__main__":
    # Quick test
    print("Testing Feature Extractor...")
    
    # Create a simple test signal
    sample_rate = 1000000
    duration = 0.001
    time = np.linspace(0, duration, int(sample_rate * duration))
    test_signal = np.sin(2 * np.pi * 10000 * time)
    
    # Extract features
    extractor = FeatureExtractor(sample_rate=sample_rate)
    features = extractor.extract_all_features(test_signal)
    
    print(f"\n✓ Extracted {len(features)} features")
    print("\nSample features:")
    for i, (name, value) in enumerate(list(features.items())[:5]):
        print(f"  {name}: {value:.4f}")
    print("  ...")
    
    print("\nFeature Extractor working correctly! ✓")