"""
Vision-based Automatic Emergency Braking (AEB) Safety System

Simple safety checker that:
1. Uses depth data to detect nearby obstacles
2. Checks if there's enough clearance in free space areas
3. Triggers emergency stop if no safe path exists
"""

import numpy as np
import cv2


class VisionAEBSafety:
    """Simple AEB safety system using vision and depth data"""
    
    def __init__(self, safety_distance=1.0, min_gap_width=0.5):
        """
        Initialize AEB safety parameters
        
        Args:
            safety_distance (float): Minimum safe distance in meters
            min_gap_width (float): Minimum gap width in meters to be considered safe
        """
        self.safety_distance = safety_distance
        self.min_gap_width = min_gap_width
        
        # ROI parameters (same as vision gap follow)
        self.roi_top_fraction = 0.7
        self.roi_bottom_fraction = 1.0
        self.roi_left_fraction = 0.0
        self.roi_right_fraction = 1.0
    
    def get_roi_coordinates(self, height, width):
        """Get ROI coordinates"""
        top = int(height * self.roi_top_fraction)
        bottom = int(height * self.roi_bottom_fraction)
        left = int(width * self.roi_left_fraction)
        right = int(width * self.roi_right_fraction)
        return top, bottom, left, right
    
    def check_safety(self, rgb_image, depth_image):
        """
        Check if it's safe to continue driving
        
        Args:
            rgb_image (np.ndarray): RGB image
            depth_image (np.ndarray): Depth image in meters
            
        Returns:
            tuple: (is_safe, reason, safety_data)
        """
        if rgb_image is None or depth_image is None:
            return False, "Missing image data", {}
        
        # Convert RGB to grayscale for free space detection
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image
        
        # Get ROI
        height, width = gray.shape
        top, bottom, left, right = self.get_roi_coordinates(height, width)
        
        # Extract ROI from both images
        roi_gray = gray[top:bottom, left:right]
        roi_depth = depth_image[top:bottom, left:right]
        
        # Detect free space in ROI
        _, free_space = cv2.threshold(roi_gray, 80, 255, cv2.THRESH_BINARY)
        
        # Check depth in free space areas
        free_space_mask = free_space > 0
        
        if not np.any(free_space_mask):
            return False, "No free space detected", {
                'min_depth': 0.0,
                'safe_pixels': 0,
                'total_pixels': free_space_mask.size
            }
        
        # Get depth values only in free space areas
        free_space_depths = roi_depth[free_space_mask]
        
        # Remove invalid depth values (0 or very large values)
        valid_depths = free_space_depths[(free_space_depths > 0.1) & (free_space_depths < 10.0)]
        
        if len(valid_depths) == 0:
            return False, "No valid depth data in free space", {
                'min_depth': 0.0,
                'safe_pixels': 0,
                'total_pixels': free_space_mask.size
            }
        
        # Check minimum distance to obstacles in free space
        min_depth = np.min(valid_depths)
        safe_pixels = np.sum(valid_depths > self.safety_distance)
        total_free_pixels = len(valid_depths)
        
        # Safety criteria
        is_min_distance_safe = min_depth > self.safety_distance
        safe_percentage = safe_pixels / total_free_pixels if total_free_pixels > 0 else 0
        is_enough_safe_space = safe_percentage > 0.3  # At least 30% of free space should be safe
        
        is_safe = is_min_distance_safe and is_enough_safe_space
        
        if not is_safe:
            if not is_min_distance_safe:
                reason = f"Obstacle too close: {min_depth:.2f}m < {self.safety_distance}m"
            else:
                reason = f"Insufficient safe space: {safe_percentage:.1%} < 30%"
        else:
            reason = "Path is safe"
        
        safety_data = {
            'min_depth': float(min_depth),
            'safe_pixels': int(safe_pixels),
            'total_pixels': int(total_free_pixels),
            'safe_percentage': float(safe_percentage),
            'is_min_distance_safe': bool(is_min_distance_safe),
            'is_enough_safe_space': bool(is_enough_safe_space)
        }
        
        return is_safe, reason, safety_data
    
    def get_emergency_action(self, is_safe, safety_data):
        """
        Determine emergency action based on safety check
        
        Args:
            is_safe (bool): Result from safety check
            safety_data (dict): Detailed safety information
            
        Returns:
            dict: Emergency action with speed and steering commands
        """
        if is_safe:
            return {
                'action': 'continue',
                'speed': None,  # No speed override
                'steering': None,  # No steering override
                'brake': False
            }
        else:
            return {
                'action': 'emergency_stop',
                'speed': 0.0,  # Full stop
                'steering': 0.0,  # Keep current heading
                'brake': True
            }


def test_vision_aeb():
    """Test function for vision AEB system"""
    print("Vision AEB Safety System Test")
    print("=" * 40)
    
    # Create AEB system
    aeb = VisionAEBSafety(safety_distance=1.5, min_gap_width=0.5)
    
    # Test with dummy data
    test_rgb = np.ones((480, 960, 3), dtype=np.uint8) * 128  # Gray image
    test_depth = np.ones((480, 960), dtype=np.float32) * 2.0  # 2m distance
    
    # Add some "free space" (brighter area)
    test_rgb[300:400, 400:600] = 200  # Bright area = free space
    
    # Add close obstacle in free space
    test_depth[350:370, 450:550] = 0.8  # Close obstacle at 0.8m
    
    is_safe, reason, data = aeb.check_safety(test_rgb, test_depth)
    action = aeb.get_emergency_action(is_safe, data)
    
    print(f"Safety check: {'SAFE' if is_safe else 'UNSAFE'}")
    print(f"Reason: {reason}")
    print(f"Action: {action['action']}")
    print(f"Safety data: {data}")


if __name__ == "__main__":
    test_vision_aeb()