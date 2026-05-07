"""
Preprocess UBFC-rPPG dataset: Extract and cache RGB signals from videos
Run this once before training to speed up data loading
"""
import numpy as np
import cv2
import os
import sys
from tqdm import tqdm
import traceback

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.face_detection import FaceDetector
from utils.roi_extraction import ROIExtractor
from utils.signal_processing import SignalProcessor
from configs.config import config

def preprocess_all_videos(data_root, output_dir):
    """Extract RGB signals from all videos and save as .npy files"""
    os.makedirs(output_dir, exist_ok=True)
    
    face_detector = FaceDetector()
    roi_extractor = ROIExtractor()
    signal_processor = SignalProcessor(fs=config.fps)
    
    # Get all subject directories
    subjects = sorted([d for d in os.listdir(data_root) 
                      if os.path.isdir(os.path.join(data_root, d))])
    
    print(f"Found {len(subjects)} subjects")
    
    for subject in tqdm(subjects, desc="Processing videos"):
        subject_path = os.path.join(data_root, subject)
        output_path = os.path.join(output_dir, f"{subject}.npy")
        
        # Skip if already processed
        if os.path.exists(output_path):
            continue
        
        # Find video file
        video_path = None
        for vid_name in ["vid.avi", "video.avi", "subject.avi"]:
            path = os.path.join(subject_path, vid_name)
            if os.path.exists(path):
                video_path = path
                break
        
        if video_path is None:
            print(f"\nNo video found for {subject}")
            continue
        
        # Extract RGB signal
        cap = cv2.VideoCapture(video_path)
        rgb_signals = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            bbox = face_detector.detect(frame)
            if bbox:
                roi_regions = roi_extractor.extract_roi(frame, bbox)
                mean_rgb = roi_extractor.compute_mean_rgb(roi_regions)
                rgb_signals.append(mean_rgb)
        
        cap.release()
        
        if len(rgb_signals) == 0:
            print(f"\nNo face detected in {subject}")
            continue
        
        rgb_signals = np.array(rgb_signals)
        
        # Apply signal processing
        rgb_signals = signal_processor.bandpass_filter(rgb_signals)
        rgb_signals = signal_processor.normalize_signal(rgb_signals)
        
        # Save
        np.save(output_path, rgb_signals)
    
    face_detector.close()
    print(f"\nPreprocessing complete. Files saved to {output_dir}")

def preprocess_ground_truth(data_root, output_dir):
    """Extract and resample ground truth PPG signals"""
    import scipy.io
    from scipy import signal as sp_signal
    
    os.makedirs(output_dir, exist_ok=True)
    
    subjects = sorted([d for d in os.listdir(data_root) 
                      if os.path.isdir(os.path.join(data_root, d))])
    
    for subject in tqdm(subjects, desc="Processing ground truth"):
        subject_path = os.path.join(data_root, subject)
        output_path = os.path.join(output_dir, f"{subject}_gt.npy")
        
        if os.path.exists(output_path):
            continue
        
        # Find ground truth file
        gt_path = None
        for gt_name in ["ground_truth.mat", "bvp.mat", "ppg.mat"]:
            path = os.path.join(subject_path, gt_name)
            if os.path.exists(path):
                gt_path = path
                break
        
        if gt_path is None:
            continue
        
        # Load ground truth
        if gt_path.endswith('.mat'):
            mat_data = scipy.io.loadmat(gt_path)
            signal = None
            for key in ['bvp', 'BVP', 'ppg', 'PPG', 'signal']:
                if key in mat_data:
                    signal = mat_data[key].flatten()
                    break
            if signal is None:
                continue
        else:
            continue
        
        # Save
        np.save(output_path, signal)
    
    print(f"\nGround truth preprocessing complete. Files saved to {output_dir}")

if __name__ == "__main__":
    data_root = config.data_root
    preprocessed_dir = os.path.join(data_root, "../preprocessed")
    preprocessed_dir = os.path.abspath(preprocessed_dir)
    
    print("="*60)
    print("Preprocessing UBFC-rPPG Dataset")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Output dir: {preprocessed_dir}")
    
    # Preprocess videos
    print("\n1. Extracting RGB signals from videos...")
    preprocess_all_videos(data_root, preprocessed_dir)
    
    # Preprocess ground truth
    print("\n2. Processing ground truth signals...")
    preprocess_ground_truth(data_root, preprocessed_dir)
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
