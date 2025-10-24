"""
Signal Generator Module
Creates synthetic RF signals for training ML models
"""

import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class SignalGenerator:
    """
    Base class for generating various types of RF signals.
    
    This creates synthetic signals that mimic real-world radar and 
    communication signals for training our ML models.
    """
    
    def __init__(self, sample_rate: int = 1000000, duration: float = 0.001):
        """
        Initialize the signal generator.
        
        Args:
            sample_rate: Samples per second (Hz) - how fast we sample the signal
            duration: Length of signal in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_points = int(sample_rate * duration)
        self.time = np.linspace(0, duration, self.num_points)
        
    def generate_pulse(self, frequency: float = 10000, 
                       amplitude: float = 1.0) -> np.ndarray:
        """
        Generate a simple pulse signal (like a radar pulse).
        
        Args:
            frequency: Signal frequency in Hz
            amplitude: Signal strength
            
        Returns:
            Array of signal values
        """
        signal = amplitude * np.sin(2 * np.pi * frequency * self.time)
        
        # Create pulse envelope (signal only exists for part of the time)
        pulse_width = 0.0003  # 0.3 milliseconds
        envelope = np.zeros_like(self.time)
        pulse_samples = int(pulse_width * self.sample_rate)
        envelope[:pulse_samples] = 1.0
        
        return signal * envelope
    
    def generate_chirp(self, start_freq: float = 5000, 
                       end_freq: float = 50000,
                       amplitude: float = 1.0) -> np.ndarray:
        """
        Generate a chirp signal (frequency increases over time).
        Used in radar systems to improve range resolution.
        
        Args:
            start_freq: Starting frequency in Hz
            end_freq: Ending frequency in Hz
            amplitude: Signal strength
            
        Returns:
            Array of signal values
        """
        # Calculate instantaneous frequency that increases linearly
        chirp_rate = (end_freq - start_freq) / self.duration
        instantaneous_freq = start_freq + chirp_rate * self.time
        
        # Create the chirp signal
        phase = 2 * np.pi * (start_freq * self.time + 
                            0.5 * chirp_rate * self.time**2)
        signal = amplitude * np.sin(phase)
        
        return signal
    
    def generate_fsk(self, freq1: float = 10000, freq2: float = 20000,
                     bit_rate: float = 1000, amplitude: float = 1.0) -> np.ndarray:
        """
        Generate FSK (Frequency Shift Keying) signal.
        Used in digital communications - two frequencies represent 0 and 1.
        
        Args:
            freq1: Frequency for bit 0
            freq2: Frequency for bit 1
            bit_rate: How many bits per second
            amplitude: Signal strength
            
        Returns:
            Array of signal values
        """
        samples_per_bit = int(self.sample_rate / bit_rate)
        num_bits = int(self.duration * bit_rate)
        
        # Generate random bit sequence (digital data to transmit)
        bits = np.random.randint(0, 2, num_bits)
        
        signal = np.zeros(self.num_points)
        
        for i, bit in enumerate(bits):
            start_idx = i * samples_per_bit
            end_idx = min((i + 1) * samples_per_bit, self.num_points)
            time_segment = self.time[start_idx:end_idx]
            
            # Use freq1 for bit 0, freq2 for bit 1
            freq = freq1 if bit == 0 else freq2
            signal[start_idx:end_idx] = amplitude * np.sin(2 * np.pi * freq * time_segment)
        
        return signal
    
    def generate_psk(self, frequency: float = 15000, 
                     bit_rate: float = 1000,
                     amplitude: float = 1.0) -> np.ndarray:
        """
        Generate PSK (Phase Shift Keying) signal.
        Used in digital communications - phase changes represent data.
        
        Args:
            frequency: Carrier frequency in Hz
            bit_rate: How many bits per second
            amplitude: Signal strength
            
        Returns:
            Array of signal values
        """
        samples_per_bit = int(self.sample_rate / bit_rate)
        num_bits = int(self.duration * bit_rate)
        
        # Generate random bit sequence
        bits = np.random.randint(0, 2, num_bits)
        
        signal = np.zeros(self.num_points)
        
        for i, bit in enumerate(bits):
            start_idx = i * samples_per_bit
            end_idx = min((i + 1) * samples_per_bit, self.num_points)
            time_segment = self.time[start_idx:end_idx]
            
            # Phase shift: 0 degrees for bit 0, 180 degrees for bit 1
            phase = 0 if bit == 0 else np.pi
            signal[start_idx:end_idx] = amplitude * np.sin(2 * np.pi * frequency * time_segment + phase)
        
        return signal
    
    def generate_qam(self, frequency: float = 15000,
                     symbol_rate: float = 500,
                     amplitude: float = 1.0) -> np.ndarray:
        """
        Generate QAM (Quadrature Amplitude Modulation) signal.
        Advanced modulation used in high-speed communications.
        
        Args:
            frequency: Carrier frequency in Hz
            symbol_rate: Symbols per second
            amplitude: Signal strength
            
        Returns:
            Array of signal values
        """
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        num_symbols = int(self.duration * symbol_rate)
        
        # Generate random I and Q components (in-phase and quadrature)
        I_data = np.random.choice([-1, 1], num_symbols)
        Q_data = np.random.choice([-1, 1], num_symbols)
        
        signal = np.zeros(self.num_points)
        
        for i in range(num_symbols):
            start_idx = i * samples_per_symbol
            end_idx = min((i + 1) * samples_per_symbol, self.num_points)
            time_segment = self.time[start_idx:end_idx]
            
            # QAM combines amplitude and phase modulation
            I_signal = I_data[i] * np.cos(2 * np.pi * frequency * time_segment)
            Q_signal = Q_data[i] * np.sin(2 * np.pi * frequency * time_segment)
            signal[start_idx:end_idx] = amplitude * (I_signal + Q_signal)
        
        return signal
    
    def generate_noise(self, amplitude: float = 0.5) -> np.ndarray:
        """
        Generate white noise (random signal).
        Represents interference or jamming.
        
        Args:
            amplitude: Noise level
            
        Returns:
            Array of noise values
        """
        return amplitude * np.random.randn(self.num_points)
    
    def add_noise(self, signal: np.ndarray, snr_db: float = 20) -> np.ndarray:
        """
        Add realistic noise to a clean signal.
        
        Args:
            signal: Clean signal
            snr_db: Signal-to-Noise Ratio in decibels (higher = cleaner)
            
        Returns:
            Noisy signal
        """
        # Calculate signal power
        signal_power = np.mean(signal ** 2)
        
        # Calculate noise power needed for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate and add noise
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        return signal + noise
    
    def generate_signal_batch(self, signal_type: str, 
                              num_samples: int = 100,
                              add_noise_flag: bool = True,
                              snr_range: Tuple[float, float] = (10, 30)) -> Dict:
        """
        Generate multiple signals of the same type with variations.
        
        Args:
            signal_type: Type of signal ('pulse', 'chirp', 'fsk', 'psk', 'qam', 'noise')
            num_samples: How many signals to generate
            add_noise_flag: Whether to add noise
            snr_range: Range of SNR values if adding noise
            
        Returns:
            Dictionary with signals and labels
        """
        signals = []
        labels = []
        
        for _ in range(num_samples):
            # Generate base signal based on type
            if signal_type == 'pulse':
                freq = np.random.uniform(5000, 50000)
                signal = self.generate_pulse(frequency=freq)
            elif signal_type == 'chirp':
                start_f = np.random.uniform(5000, 20000)
                end_f = np.random.uniform(30000, 80000)
                signal = self.generate_chirp(start_freq=start_f, end_freq=end_f)
            elif signal_type == 'fsk':
                f1 = np.random.uniform(5000, 15000)
                f2 = np.random.uniform(20000, 40000)
                signal = self.generate_fsk(freq1=f1, freq2=f2)
            elif signal_type == 'psk':
                freq = np.random.uniform(10000, 30000)
                signal = self.generate_psk(frequency=freq)
            elif signal_type == 'qam':
                freq = np.random.uniform(10000, 30000)
                signal = self.generate_qam(frequency=freq)
            elif signal_type == 'noise':
                signal = self.generate_noise()
            else:
                raise ValueError(f"Unknown signal type: {signal_type}")
            
            # Add noise if requested
            if add_noise_flag and signal_type != 'noise':
                snr = np.random.uniform(snr_range[0], snr_range[1])
                signal = self.add_noise(signal, snr_db=snr)
            
            signals.append(signal)
            labels.append(signal_type)
        
        return {
            'signals': np.array(signals),
            'labels': np.array(labels),
            'signal_type': signal_type,
            'sample_rate': self.sample_rate,
            'duration': self.duration
        }


if __name__ == "__main__":
    # Quick test of the signal generator
    print("Testing Signal Generator...")
    
    gen = SignalGenerator(sample_rate=1000000, duration=0.001)
    
    # Test each signal type
    pulse = gen.generate_pulse()
    print(f"✓ Pulse signal generated: {len(pulse)} samples")
    
    chirp = gen.generate_chirp()
    print(f"✓ Chirp signal generated: {len(chirp)} samples")
    
    fsk = gen.generate_fsk()
    print(f"✓ FSK signal generated: {len(fsk)} samples")
    
    print("\nSignal Generator working correctly! ✓")