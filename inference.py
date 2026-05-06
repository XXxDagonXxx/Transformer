import torch
import cv2
import numpy as np
import os
import sys
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import config
from models.transformer import TransformerEncoder
from utils.face_detection import FaceDetector
from utils.roi_extraction import ROIExtractor
from utils.signal_processing import SignalProcessor

class RPPGPredictor:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or config.device
        
        # Initialize model
        self.model = TransformerEncoder(
            input_dim=3,
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_seq_length=config.max_seq_length
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize utilities
        self.face_detector = FaceDetector()
        self.roi_extractor = ROIExtractor()
        self.signal_processor = SignalProcessor(
            fs=config.fps,
            lowcut=config.bandpass_low,
            highcut=config.bandpass_high
        )
        
        print(f"Model loaded from {model_path}")
    
    def process_video(self, video_path: str) -> Tuple[np.ndarray, float]:
        """Process video and predict rPPG signal and BPM"""
        cap = cv2.VideoCapture(video_path)
        rgb_signals = []
        
        print("Extracting RGB signals from video...")
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
            print("No face detected in video!")
            return None, None
        
        rgb_signals = np.array(rgb_signals)
        
        # Apply signal processing
        rgb_signals = self.signal_processor.bandpass_filter(rgb_signals)
        rgb_signals = self.signal_processor.normalize_signal(rgb_signals)
        
        # Create sequences
        sequence_length = config.sequence_length
        if len(rgb_signals) < sequence_length:
            pad_len = sequence_length - len(rgb_signals)
            rgb_signals = np.pad(rgb_signals, ((0, pad_len), (0, 0)), mode='constant')
        else:
            rgb_signals = rgb_signals[:sequence_length]
        
        # Convert to tensor
        rgb_tensor = torch.FloatTensor(rgb_signals).unsqueeze(0).to(self.device)  # (1, T, 3)
        
        # Predict
        with torch.no_grad():
            pred_signal, pred_bpm = self.model(rgb_tensor)
        
        pred_signal = pred_signal.cpu().numpy().flatten()
        pred_bpm = pred_bpm.cpu().numpy().flatten()[0]
        
        return pred_signal, pred_bpm
    
    def predict_from_webcam(self):
        """Real-time rPPG prediction from webcam"""
        cap = cv2.VideoCapture(0)
        rgb_buffer = []
        
        print("Press 'q' to quit...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            bbox = self.face_detector.detect(frame)
            if bbox:
                roi_regions = self.roi_extractor.extract_roi(frame, bbox)
                mean_rgb = self.roi_extractor.compute_mean_rgb(roi_regions)
                rgb_buffer.append(mean_rgb)
                
                # Draw bounding box
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow('rPPG Prediction', frame)
            
            # Predict when buffer is full
            if len(rgb_buffer) >= config.sequence_length:
                rgb_signals = np.array(rgb_buffer[-config.sequence_length:])
                rgb_signals = self.signal_processor.bandpass_filter(rgb_signals)
                rgb_signals = self.signal_processor.normalize_signal(rgb_signals)
                
                rgb_tensor = torch.FloatTensor(rgb_signals).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    pred_signal, pred_bpm = self.model(rgb_tensor)
                
                pred_bpm = pred_bpm.cpu().numpy().flatten()[0]
                print(f"Predicted BPM: {pred_bpm:.1f}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def close(self):
        self.face_detector.close()

def main():
    model_path = os.path.join(config.save_dir, config.model_name)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    predictor = RPPGPredictor(model_path)
    
    # Example: predict from video
    video_path = input("Enter video path (or press Enter for webcam): ").strip()
    
    if video_path:
        pred_signal, pred_bpm = predictor.process_video(video_path)
        if pred_signal is not None:
            print(f"Predicted rPPG signal shape: {pred_signal.shape}")
            print(f"Predicted BPM: {pred_bpm:.1f}")
    else:
        predictor.predict_from_webcam()
    
    predictor.close()

if __name__ == "__main__":
    main()
