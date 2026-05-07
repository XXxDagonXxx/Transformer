import cv2
import numpy as np

class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        # Use OpenCV's Haar Cascade as fallback (more reliable)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Try to load MediaPipe if available
        self.use_mediapipe = False
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=min_detection_confidence
            )
            self.use_mediapipe = True
            print("Using MediaPipe for face detection")
        except (ImportError, AttributeError):
            print("MediaPipe not available, using OpenCV Haar Cascade")
    
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect face and return bounding box (x, y, w, h)"""
        if self.use_mediapipe:
            return self._detect_mediapipe(frame)
        else:
            return self._detect_opencv(frame)
    
    def _detect_mediapipe(self, frame: np.ndarray) -> np.ndarray:
        """Detect face using MediaPipe"""
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
    
    def _detect_opencv(self, frame: np.ndarray) -> np.ndarray:
        """Detect face using OpenCV Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Return the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return (x, y, w, h)
        return None
    
    def extract_roi_from_landmarks(self, frame: np.ndarray, bbox) -> dict:
        """Extract ROI regions from bounding box (simplified without landmarks)"""
        x, y, w, h = bbox
        
        roi_regions = {}
        
        # Forehead (top 20% of face)
        forehead_y = y
        forehead_h = int(h * 0.2)
        forehead_x = x + int(w * 0.25)
        forehead_w = int(w * 0.5)
        roi_regions["forehead"] = (forehead_x, forehead_y, forehead_w, forehead_h)
        
        # Left cheek
        left_cheek_x = x
        left_cheek_y = y + int(h * 0.4)
        left_cheek_w = int(w * 0.3)
        left_cheek_h = int(h * 0.3)
        roi_regions["left_cheek"] = (left_cheek_x, left_cheek_y, left_cheek_w, left_cheek_h)
        
        # Right cheek
        right_cheek_x = x + int(w * 0.7)
        right_cheek_y = y + int(h * 0.4)
        right_cheek_w = int(w * 0.3)
        right_cheek_h = int(h * 0.3)
        roi_regions["right_cheek"] = (right_cheek_x, right_cheek_y, right_cheek_w, right_cheek_h)
        
        return roi_regions
    
    def close(self):
        if self.use_mediapipe:
            self.face_detection.close()
