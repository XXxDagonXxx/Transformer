import mediapipe as mp
import numpy as np
import cv2
from typing import Tuple, Optional

class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face and return bounding box (x, y, w, h)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            return (x, y, width, height)
        return None
    
    def get_landmarks(self, frame: np.ndarray) -> Optional[dict]:
        """Get face landmarks for ROI extraction"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]
            landmarks = {}
            h, w = frame.shape[:2]
            
            for i, landmark in enumerate(detection.location_data.relative_keypoints):
                landmarks[i] = (int(landmark.x * w), int(landmark.y * h))
            
            return landmarks
        return None
    
    def close(self):
        self.face_detection.close()
