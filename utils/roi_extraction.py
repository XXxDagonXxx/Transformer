import cv2
import numpy as np
from typing import List, Dict, Tuple

class ROIExtractor:
    def __init__(self, regions: List[str] = None):
        self.regions = regions or ["forehead", "left_cheek", "right_cheek"]
    
    def extract_roi(self, frame: np.ndarray, bbox) -> dict:
        """Extract ROI regions from face bounding box"""
        x, y, w, h = bbox
        roi_regions = {}
        
        if "forehead" in self.regions:
            forehead_y = y
            forehead_h = int(h * 0.15)
            forehead_x = x + int(w * 0.25)
            forehead_w = int(w * 0.5)
            roi_regions["forehead"] = frame[
                forehead_y:forehead_y + forehead_h,
                forehead_x:forehead_x + forehead_w
            ]
        
        if "left_cheek" in self.regions:
            left_cheek_x = x
            left_cheek_y = y + int(h * 0.4)
            left_cheek_w = int(w * 0.3)
            left_cheek_h = int(h * 0.3)
            roi_regions["left_cheek"] = frame[
                left_cheek_y:left_cheek_y + left_cheek_h,
                left_cheek_x:left_cheek_x + left_cheek_w
            ]
        
        if "right_cheek" in self.regions:
            right_cheek_x = x + int(w * 0.7)
            right_cheek_y = y + int(h * 0.4)
            right_cheek_w = int(w * 0.3)
            right_cheek_h = int(h * 0.3)
            roi_regions["right_cheek"] = frame[
                right_cheek_y:right_cheek_y + right_cheek_h,
                right_cheek_x:right_cheek_x + right_cheek_w
            ]
        
        return roi_regions
    
    def compute_mean_rgb(self, roi_regions: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute mean RGB values across all ROI regions"""
        rgb_values = []
        
        for region_name, roi in roi_regions.items():
            if roi.size > 0:
                mean_rgb = np.mean(roi, axis=(0, 1))
                rgb_values.append(mean_rgb)
        
        if rgb_values:
            return np.mean(rgb_values, axis=0)
        else:
            return np.zeros(3)
