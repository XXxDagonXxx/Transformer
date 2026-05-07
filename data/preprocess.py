import numpy as np
import cv2
import os
import sys
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.face_detection import FaceDetector
from utils.roi_extraction import ROIExtractor
from utils.signal_processing import SignalProcessor
from configs.config import config

def preprocess_dataset(data_root, output_dir, sequence_length=300, fps=30):
    """Preprocess all videos and save RGB signals as .npy files"""
    os.makedirs(output_dir, exist_ok=True)
    
    face_detector = FaceDetector()
    roi_extractor = ROIExtractor()
    signal_processor = SignalProcessor(fs=fps)
    
    subjects = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    for subject in tqdm(subjects, desc="Preprocessing subjects"):
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
            print(f"No video found for {subject}")
            continue
        
        # Extract RGB signal
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
        
        if len(rgb_signals) == 0:
            print(f"No face detected in {subject}")
            continue
        
        rgb_signals = np.array(rgb_signals)
        
        # Apply signal processing
        rgb_signals = signal_processor.bandpass_filter(rgb_signals)
        rgb_signals = signal_processor.normalize_signal(rgb_signals)
        
        # Save
        np.save(output_path, rgb_signals)
    
    face_detector.close()
    print(f"Preprocessing complete. Files saved to {output_dir}")

if __name__ == "__main__":
    data_root = config.data_root
    output_dir = os.path.join(data_root, "../preprocessed")
    preprocess_dataset(data_root, output_dir, config.sequence_length, config.fps)
