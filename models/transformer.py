import torch
import torch.nn as nn
import math
from typing import Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input
        
        Args:
            x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 300
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Output heads
        self.signal_head = nn.Linear(d_model, 1)  # rPPG signal regression
        self.bpm_head = nn.Linear(d_model, 1)  # BPM regression
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            x: (batch_size, seq_len, 3) - RGB temporal signal
            
        Returns:
            rppg_signal: (batch_size, seq_len) - predicted rPPG signal
            bpm: (batch_size, 1) - predicted BPM
        """
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # rPPG signal prediction (sequence)
        rppg_signal = self.signal_head(encoded).squeeze(-1)  # (batch_size, seq_len)
        
        # BPM prediction (use mean of encoded sequence)
        pooled = encoded.mean(dim=1)  # (batch_size, d_model)
        bpm = self.bpm_head(pooled)  # (batch_size, 1)
        
        return rppg_signal, bpm
