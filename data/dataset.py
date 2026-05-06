import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from typing import List, Tuple
from utils.face_detection import FaceDetector
from utils.roi_extraction import ROIExtractor
from utils.signal_processing import SignalProcessor

class UBFCDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        sequence_length: int = 300,
        fps: int = 30,
        is_train: bool = True,
        transform=None
    ):
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.fps = fps
        self.is_train = is_train
        self.transform = transform
        
        self.face_detector = FaceDetector()
        self.roi_extractor = ROIExtractor()
        self.signal_processor = SignalProcessor(fs=fps)
        
        # Load video paths and ground truth
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[dict]:
        """Load video paths and corresponding ground truth BPM"""
        samples = []
        
        # UBFC-rPPG structure: subject folders with videos and ground truth
        for subject_dir in os.listdir(self.data_root):
            subject_path = os.path.join(self.data_root, subject_dir)
            if not os.path.isdir(subject_path):
                continue
            
            video_path = os.path.join(subject_path, "vid.avi")
            gt_path = os.path.join(subject_path, "ground_truth.txt")
            
            if os.path.exists(video_path) and os.path.exists(gt_path):
                with open(gt_path, 'r') as f:
                    gt_bpm = float(f.read().strip())
                
                samples.append({
                    'video_path': video_path,
                    'gt_bpm': gt_bpm
                })
        
        return samples
    
    def _extract_rgb_signal(self, video_path: str) -> np.ndarray:
        """Extract RGB temporal signal from video"""
        cap = cv2.VideoCapture(video_path)
        rgb_signals = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            bbox = self.face_detector.detect(frame)
            if bbox:
                roi_regions = self.roi_extractor.extract_roi(frame, bbox)
                mean_rgb = self.roi_extractor.compute_mean_rgb(roi_regions)
                rgb_signals.append(mean_rgb)
        
        cap.release()
        
        if len(rgb_signals) == 0:
            return np.zeros((self.sequence_length, 3))
        
        rgb_signals = np.array(rgb_signals)
        
        # Apply bandpass filter
        rgb_signals = self.signal_processor.bandpass_filter(rgb_signals)
        
        # Normalize
        rgb_signals = self.signal_processor.normalize_signal(rgb_signals)
        
        return rgb_signals
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        video_path = sample['video_path']
        gt_bpm = sample['gt_bpm']
        
        # Extract RGB signal
        rgb_signal = self._extract_rgb_signal(video_path)
        
        # Create windowed sequences
        if len(rgb_signal) >= self.sequence_length:
            start = np.random.randint(0, len(rgb_signal) - self.sequence_length) if self.is_train else 0
            rgb_signal = rgb_signal[start:start + self.sequence_length]
        else:
            # Pad if too short
            pad_len = self.sequence_length - len(rgb_signal)
            rgb_signal = np.pad(rgb_signal, ((0, pad_len), (0, 0)), mode='constant')
        
        # Convert to tensor
        rgb_tensor = torch.FloatTensor(rgb_signal)  # (T, 3)
        
        # Compute ground truth rPPG signal (using green channel as proxy)
        gt_signal = torch.FloatTensor(rgb_signal[:, 1])  # (T,)
        
        # Ground truth BPM
        gt_bpm_tensor = torch.FloatTensor([gt_bpm])
        
        return rgb_tensor, gt_signal, gt_bpm_tensor
    
    def __del__(self):
        if hasattr(self, 'face_detector'):
            self.face_detector.close()
