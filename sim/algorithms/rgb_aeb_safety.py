"""
Vision AEB Safety System

Core production algorithm for F1Tenth autonomous vehicle.
Uses free space detection with optional depth enhancement for obstacle avoidance.
"""

import numpy as np
import cv2


class VisionAEBSafety:
    """AEB safety system using RGB vision data with optional depth enhancement"""
    
    def __init__(self, min_free_space_percentage=0.5, safety_buffer=0.2, 
                 min_safe_distance=1.5, critical_distance=0.3, aeb_fov_fraction=0.4,
                 depth_min_valid=0.1, depth_max_valid=10.0, depth_percentile=5.0):
        self.min_free_space_percentage = min_free_space_percentage  # minimum required free space fraction in ROI
        self.safety_buffer = safety_buffer  # additional safety margin (0.0 to 1.0)
        self.min_safe_distance = min_safe_distance  # minimum safe distance in meters (for depth)
        self.critical_distance = critical_distance  # critical distance triggering emergency stop (for depth)
        self.aeb_fov_fraction = aeb_fov_fraction  # fraction of image width to check for AEB (0.4 = center 40%)
        self.depth_min_valid = depth_min_valid  # minimum valid depth reading in meters (filters sensor noise)
        self.depth_max_valid = depth_max_valid  # maximum valid depth reading in meters (filters unreliable readings)
        self.depth_percentile = depth_percentile  # percentile to use for depth safety check (10.0 = 10th percentile)
        
        self.roi_top_fraction = 0.7
        self.roi_bottom_fraction = 1.0
        self.roi_left_fraction = 0.0
        self.roi_right_fraction = 1.0
    
    def get_roi_coordinates(self, height, width):  # get ROI coordinates
        top = int(height * self.roi_top_fraction)
        bottom = int(height * self.roi_bottom_fraction)
        left = int(width * self.roi_left_fraction)
        right = int(width * self.roi_right_fraction)
        return top, bottom, left, right
    
    # so instead of checking entire ROI, we check the area directly in front of the car for safety decisions
    def get_aeb_safety_zone(self, height, width): 
        top = int(height * self.roi_top_fraction)
        bottom = int(height * self.roi_bottom_fraction)
        center_x = width // 2
        half_aeb_width = int(width * self.aeb_fov_fraction / 2)
        left = center_x - half_aeb_width
        right = center_x + half_aeb_width
        left = max(0, left)
        right = min(width - 1, right)
        
        return top, bottom, left, right
    
    # check if it's safe to continue driving based on RGB and optional depth data
    def check_safety(self, rgb_image, depth_image=None):  
        if rgb_image is None:
            return False, "Missing image data", {}
        
        # handle both color (3D) and grayscale (2D)
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image
        
        height, width = gray.shape
        top, bottom, left, right = self.get_aeb_safety_zone(height, width)
        roi_gray = gray[top:bottom, left:right]
        roi_height, roi_width = roi_gray.shape
        total_roi_pixels = roi_height * roi_width
        
        # anything above 80 is marked as free space
        _, free_space = cv2.threshold(roi_gray, 80, 255, cv2.THRESH_BINARY)
        
        # we will have noise, use morhological open and close to clean up noisy regions
        kernel = np.ones((5,5), np.uint8)
        free_space = cv2.morphologyEx(free_space, cv2.MORPH_CLOSE, kernel)
        free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel)
        
        # 0 is an obstacle, anything >0 is free space, so we sum up each pixel >0 to find out free space %
        free_space_pixels = np.sum(free_space > 0)
        free_space_percentage = free_space_pixels / total_roi_pixels
        
        # we can modify this, for now just add safety buffer
        required_free_space = self.min_free_space_percentage + self.safety_buffer
        is_safe = free_space_percentage >= required_free_space
        
        if is_safe:
            reason = f"Sufficient free space: {free_space_percentage:.1%} >= {required_free_space:.1%}"
        else:
            reason = f"Insufficient free space: {free_space_percentage:.1%} < {required_free_space:.1%}"
        # Simple depth enhancement if available
        min_depth = None
        method = 'rgb_only'
        
        if depth_image is not None:
            roi_depth = depth_image[top:bottom, left:right]
            valid_depths = roi_depth[(roi_depth > self.depth_min_valid) & (roi_depth < self.depth_max_valid)]
            
            if len(valid_depths) > 0:
                # check if our lower percentile is under threshold to avoid noisy readings
                min_depth = float(np.percentile(valid_depths, self.depth_percentile))
                method = 'rgb_plus_depth'
                
                # override pure RGB decision if we have critical depth readings
                if min_depth < self.critical_distance:
                    is_safe = False
                    reason = f"Critical obstacle at {min_depth:.2f}m ({self.depth_percentile}th percentile) < {self.critical_distance}m"
        
        safety_data = {
            'free_space_percentage': float(free_space_percentage),
            'method': method,
            'min_depth': min_depth
        }
        
        return is_safe, reason, safety_data
