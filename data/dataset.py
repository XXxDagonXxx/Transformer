import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from typing import List, Tuple
import scipy.io
from scipy import signal as sp_signal

class UBFCDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        sequence_length: int = 300,
        fps: int = 30,
        ppg_fs: int = 60,
        is_train: bool = True,
        split_ratio: float = 0.8
    ):
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.fps = fps
        self.ppg_fs = ppg_fs
        self.is_train = is_train
        
        # Load samples
        all_samples = self._load_samples()
        
        # Split train/val
        split_idx = int(len(all_samples) * split_ratio)
        if is_train:
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        print(f"{'Train' if is_train else 'Val'} samples: {len(self.samples)}")
    
    def _load_samples(self) -> List[dict]:
        """Load video paths and ground truth PPG signals"""
        samples = []
        
        if not os.path.exists(self.data_root):
            print(f"WARNING: Data root {self.data_root} does not exist!")
            return samples
        
        # UBFC-rPPG structure
        for subject_dir in sorted(os.listdir(self.data_root)):
            subject_path = os.path.join(self.data_root, subject_dir)
            if not os.path.isdir(subject_path):
                continue
            
            # Find video file
            video_path = None
            for vid_name in ["vid.avi", "video.avi", "subject.avi"]:
                path = os.path.join(subject_path, vid_name)
                if os.path.exists(path):
                    video_path = path
                    break
            
            # Find ground truth PPG file
            gt_path = None
            for gt_name in ["ground_truth.mat", "bvp.mat", "ppg.mat", "ground_truth.txt"]:
                path = os.path.join(subject_path, gt_name)
                if os.path.exists(path):
                    gt_path = path
                    break
            
            if video_path and gt_path:
                samples.append({
                    'subject_id': subject_dir,
                    'video_path': video_path,
                    'gt_path': gt_path
                })
        
        print(f"Total samples found: {len(samples)}")
        return samples
    
    def _load_ground_truth(self, gt_path: str, target_length: int) -> np.ndarray:
        """Load and resample ground truth PPG/BVP signal"""
        # Try preprocessed file first
        preprocessed_dir = os.path.join(os.path.dirname(self.data_root), "preprocessed")
        subject_id = os.path.basename(os.path.dirname(gt_path))
        preprocessed_path = os.path.join(preprocessed_dir, f"{subject_id}_gt.npy")
        
        if os.path.exists(preprocessed_path):
            signal = np.load(preprocessed_path)
        else:
            if gt_path.endswith('.mat'):
                mat_data = scipy.io.loadmat(gt_path)
                signal = None
                for key in ['bvp', 'BVP', 'ppg', 'PPG', 'signal', 'ground_truth']:
                    if key in mat_data:
                        signal = mat_data[key].flatten()
                        break
                if signal is None:
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            signal = mat_data[key].flatten()
                            break
            elif gt_path.endswith('.txt'):
                signal = np.loadtxt(gt_path).flatten()
            else:
                return np.zeros(target_length)
            
            if signal is None:
                return np.zeros(target_length)
        
        # Resample to target length
        if len(signal) != target_length:
            signal = sp_signal.resample(signal, target_length)
        
        return signal
    
    def _extract_rgb_signal(self, video_path: str) -> np.ndarray:
        """Load preprocessed RGB signal or extract from video"""
        # Try to load preprocessed file
        preprocessed_dir = os.path.join(os.path.dirname(self.data_root), "preprocessed")
        subject_id = os.path.basename(os.path.dirname(video_path))
        preprocessed_path = os.path.join(preprocessed_dir, f"{subject_id}.npy")
        
        if os.path.exists(preprocessed_path):
            return np.load(preprocessed_path)
        
        # Fallback: process on-the-fly
        from utils.face_detection import FaceDetector
        from utils.roi_extraction import ROIExtractor
        from utils.signal_processing import SignalProcessor
        
        face_detector = FaceDetector()
        roi_extractor = ROIExtractor()
        signal_processor = SignalProcessor(fs=self.fps)
        
        cap = cv2.VideoCapture(video_path)
        rgb_signals = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            bbox = face_detector.detect(frame)
            if bbox:
                roi_regions = roi_extractor.extract_roi(frame, bbox)
                mean_rgb = roi_extractor.compute_mean_rgb(roi_regions)
                rgb_signals.append(mean_rgb)
        
        cap.release()
        face_detector.close()
        
        if len(rgb_signals) == 0:
            return np.zeros((self.sequence_length, 3))
        
        rgb_signals = np.array(rgb_signals)
        rgb_signals = signal_processor.bandpass_filter(rgb_signals)
        rgb_signals = signal_processor.normalize_signal(rgb_signals)
        
        return rgb_signals
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        video_path = sample['video_path']
        gt_path = sample['gt_path']
        
        # Extract RGB signal from video
        rgb_signal = self._extract_rgb_signal(video_path)
        
        # Load ground truth PPG signal
        min_len = min(len(rgb_signal), self.sequence_length * 2)  # Load enough for random crop
        gt_ppg = self._load_ground_truth(gt_path, min_len)
        
        # Create windowed sequences
        if len(rgb_signal) >= self.sequence_length:
            if self.is_train:
                start = np.random.randint(0, len(rgb_signal) - self.sequence_length)
            else:
                start = 0
            rgb_signal = rgb_signal[start:start + self.sequence_length]
            
            # Align ground truth - resample to match rgb_signal length
            if len(gt_ppg) > 0:
                gt_signal = sp_signal.resample(gt_ppg, self.sequence_length)
            else:
                gt_signal = np.zeros(self.sequence_length)
        else:
            # Pad if too short
            pad_len = self.sequence_length - len(rgb_signal)
            rgb_signal = np.pad(rgb_signal, ((0, pad_len), (0, 0)), mode='constant')
            if len(gt_ppg) > 0:
                gt_signal = sp_signal.resample(gt_ppg, self.sequence_length)
            else:
                gt_signal = np.zeros(self.sequence_length)
        
        # Convert to tensor
        rgb_tensor = torch.FloatTensor(rgb_signal)  # (T, 3)
        gt_signal_tensor = torch.FloatTensor(gt_signal)  # (T,)
        
        # Compute BPM from ground truth signal
        from utils.signal_processing import SignalProcessor
        sp = SignalProcessor(fs=self.fps)
        gt_bpm = sp.compute_bpm_from_fft(gt_signal)
        gt_bpm_tensor = torch.FloatTensor([gt_bpm])
        
        return rgb_tensor, gt_signal_tensor, gt_bpm_tensor
