import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple

class SignalProcessor:
    def __init__(self, fs: int = 30, lowcut: float = 0.7, highcut: float = 4.0):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
    
    def bandpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter (0.7-4 Hz) to signal"""
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data, axis=0)
        return filtered
    
    def compute_bpm_from_fft(self, signal_data: np.ndarray) -> float:
        """Compute BPM using FFT"""
        n = len(signal_data)
        yf = fft(signal_data)
        xf = fftfreq(n, 1 / self.fs)
        
        # Get positive frequencies within bandpass range
        mask = (xf >= self.lowcut) & (xf <= self.highcut) & (xf > 0)
        freqs = xf[mask]
        magnitudes = np.abs(yf[mask])
        
        if len(magnitudes) > 0:
            peak_freq = freqs[np.argmax(magnitudes)]
            bpm = peak_freq * 60
            return bpm
        return 0.0
    
    def normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance"""
        mean = np.mean(signal_data, axis=0)
        std = np.std(signal_data, axis=0)
        std[std == 0] = 1.0
        return (signal_data - mean) / std
    
    def detrend_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Remove linear trend from signal"""
        return signal.detrend(signal_data, axis=0)
