import torch
import torch.nn as nn
from scipy.fft import fft
from typing import Tuple

class RPPGLoss(nn.Module):
    def __init__(
        self,
        signal_weight: float = 1.0,
        bpm_weight: float = 0.5,
        freq_weight: float = 0.3
    ):
        super().__init__()
        
        self.signal_weight = signal_weight
        self.bpm_weight = bpm_weight
        self.freq_weight = freq_weight
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def frequency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute frequency domain loss using FFT"""
        # Compute FFT
        pred_fft = torch.fft.fft(pred, dim=-1)
        target_fft = torch.fft.fft(target, dim=-1)
        
        # Compute magnitude
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # MSE in frequency domain (focus on relevant frequencies)
        loss = self.mse_loss(pred_mag, target_mag)
        return loss
    
    def compute_bpm_from_signal(self, signal: torch.Tensor, fs: int = 30) -> torch.Tensor:
        """Compute BPM from signal using FFT"""
        # FFT
        signal_fft = torch.fft.fft(signal, dim=-1)
        signal_mag = torch.abs(signal_fft)
        
        # Frequency bins
        freqs = torch.fft.fftfreq(signal.size(-1), d=1/fs).to(signal.device)
        
        # Find peak frequency in 0.7-4 Hz range
        mask = (freqs >= 0.7) & (freqs <= 4.0) & (freqs > 0)
        
        if mask.sum() > 0:
            masked_mag = signal_mag * mask.to(signal_mag.device)
            peak_idx = torch.argmax(masked_mag, dim=-1)
            peak_freq = freqs[peak_idx]
            bpm = peak_freq * 60
            return bpm
        return torch.zeros(signal.size(0), device=signal.device)
    
    def forward(
        self,
        pred_signal: torch.Tensor,
        pred_bpm: torch.Tensor,
        target_signal: torch.Tensor,
        target_bpm: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss
        
        Args:
            pred_signal: (batch_size, seq_len)
            pred_bpm: (batch_size, 1)
            target_signal: (batch_size, seq_len)
            target_bpm: (batch_size, 1)
            
        Returns:
            total_loss: scalar
            loss_dict: dictionary of individual losses
        """
        # Signal loss (MSE)
        signal_loss = self.mse_loss(pred_signal, target_signal)
        
        # BPM loss (MAE)
        bpm_loss = self.mae_loss(pred_bpm, target_bpm)
        
        # Frequency domain loss
        freq_loss = self.frequency_loss(pred_signal, target_signal)
        
        # Combined loss
        total_loss = (
            self.signal_weight * signal_loss +
            self.bpm_weight * bpm_loss +
            self.freq_weight * freq_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'signal_loss': signal_loss.item(),
            'bpm_loss': bpm_loss.item(),
            'freq_loss': freq_loss.item()
        }
        
        return total_loss, loss_dict
