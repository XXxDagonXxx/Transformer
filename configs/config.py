import torch
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    # Dataset
    data_root: str = "/path/to/UBFC-rPPG"
    video_dir: str = "videos"
    ground_truth_path: str = "ground_truth"
    sequence_length: int = 300  # T in (T, 3)
    fps: int = 30
    batch_size: int = 16
    num_workers: int = 4
    
    # ROI Extraction
    roi_regions: List[str] = None  # forehead, left_cheek, right_cheek
    face_detection_confidence: float = 0.5
    
    # Signal Processing
    bandpass_low: float = 0.7  # Hz
    bandpass_high: float = 4.0  # Hz
    sampling_rate: int = 30
    
    # Model
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_length: int = 300
    
    # Training
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    signal_loss_weight: float = 1.0
    bpm_loss_weight: float = 0.5
    freq_loss_weight: float = 0.3
    
    # Optimization
    use_gpu: bool = torch.cuda.is_available()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "checkpoints"
    model_name: str = "transformer_rppg.pth"
    
    # Evaluation
    metrics: List[str] = None  # MAE, RMSE, Pearson
    
    def __post_init__(self):
        if self.roi_regions is None:
            self.roi_regions = ["forehead", "left_cheek", "right_cheek"]
        if self.metrics is None:
            self.metrics = ["MAE", "RMSE", "Pearson"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
