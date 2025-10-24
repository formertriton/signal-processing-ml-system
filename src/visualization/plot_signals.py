"""
Signal Visualization Module
Creates plots to visualize different signal types
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import pickle
import os


def plot_signal_time_domain(signal_data, time, title="Signal", ax=None):
    """
    Plot signal in time domain.
    
    Args:
        signal_data: Signal array
        time: Time array
        title: Plot title
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(time * 1000, signal_data, linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_signal_frequency_domain(signal_data, sample_rate, title="Frequency Spectrum", ax=None):
    """
    Plot signal in frequency domain (FFT).
    
    Args:
        signal_data: Signal array
        sample_rate: Sampling rate in Hz
        title: Plot title
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Compute FFT
    fft = np.fft.fft(signal_data)
    frequencies = np.fft.fftfreq(len(signal_data), 1/sample_rate)
    
    # Only plot positive frequencies
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    magnitude = np.abs(fft[positive_freq_idx])
    
    ax.plot(frequencies / 1000, magnitude)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Magnitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_spectrogram(signal_data, sample_rate, title="Spectrogram", ax=None):
    """
    Plot time-frequency spectrogram.
    
    Args:
        signal_data: Signal array
        sample_rate: Sampling rate in Hz
        title: Plot title
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Compute spectrogram
    frequencies, times, Sxx = scipy_signal.spectrogram(
        signal_data, 
        sample_rate,
        nperseg=256
    )
    
    # Plot
    pcm = ax.pcolormesh(times * 1000, frequencies / 1000, 10 * np.log10(Sxx + 1e-10), 
                        shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_xlabel('Time (ms)')
    ax.set_title(title)
    plt.colorbar(pcm, ax=ax, label='Power (dB)')
    
    return ax


def plot_all_signal_types(dataset_path='data/raw/signal_dataset_latest.pkl', 
                          output_dir='results'):
    """
    Create comprehensive visualization of all signal types.
    
    Args:
        dataset_path: Path to dataset file
        output_dir: Directory to save plots
    """
    print("Loading dataset...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    signals = dataset['signals']
    labels = dataset['labels']
    sample_rate = dataset['sample_rate']
    duration = dataset['duration']
    signal_types = dataset['signal_types']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating visualizations for {len(signal_types)} signal types...")
    
    time = np.linspace(0, duration, signals.shape[1])
    
    # Plot each signal type
    for signal_type in signal_types:
        print(f"  Plotting {signal_type}...")
        
        # Get first signal of this type
        idx = np.where(labels == signal_type)[0][0]
        sig = signals[idx]
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f'{signal_type.upper()} Signal Analysis', fontsize=16, fontweight='bold')
        
        # Time domain
        plot_signal_time_domain(sig, time, f'{signal_type.upper()} - Time Domain', axes[0])
        
        # Frequency domain
        plot_signal_frequency_domain(sig, sample_rate, f'{signal_type.upper()} - Frequency Domain', axes[1])
        
        # Spectrogram
        plot_spectrogram(sig, sample_rate, f'{signal_type.upper()} - Spectrogram', axes[2])
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(output_dir, f'{signal_type}_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved to {save_path}")
    
    # Create comparison plot with all signal types
    print("\n  Creating comparison plot...")
    n_types = len(signal_types)
    fig, axes = plt.subplots(n_types, 1, figsize=(14, 2.5*n_types))
    fig.suptitle('All Signal Types - Time Domain Comparison', fontsize=16, fontweight='bold')
    
    for i, signal_type in enumerate(signal_types):
        idx = np.where(labels == signal_type)[0][0]
        sig = signals[idx]
        
        if n_types == 1:
            ax = axes
        else:
            ax = axes[i]
            
        plot_signal_time_domain(sig, time, f'{signal_type.upper()}', ax)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'all_signals_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved to {save_path}")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nPlots saved to: {output_dir}/")
    print(f"Total plots created: {len(signal_types) + 1}")


if __name__ == "__main__":
    try:
        plot_all_signal_types()
    except FileNotFoundError:
        print("❌ Dataset not found!")
        print("Please run generate_dataset.py first to create the dataset.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()