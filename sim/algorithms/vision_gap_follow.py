"""
Vision-Based Gap Following Algorithm for F1Tenth Racing

This algorithm:
1. Detects free space in RGB images using thresholding
2. Finds drivable gaps in the free space
3. Calculates steering angle toward the best gap
4. Mimics lidar gap-following behavior using vision

Key Features:
- Pure computer vision approach to gap detection
- Configurable parameters similar to lidar algorithm
- ROI-focused processing for computational efficiency
- Steering calculation based on gap geometry
"""

import numpy as np
import cv2
import os
from math import atan2, degrees, radians, pi


class VisionGapFollower:
    """
    Vision-based gap following algorithm using RGB camera data
    """
    
    def __init__(self, 
                 car_width=0.3,
                 lookahead_distance=1.0,
                 safety_buffer=0.1,
                 free_space_threshold=80,
                 min_gap_width_pixels=50,
                 smoothing_factor=0.05):
        """
        Initialize the vision gap follower
        
        Args:
            car_width (float): Width of the car in meters
            lookahead_distance (float): How far ahead to look (used for gap selection)
            safety_buffer (float): Additional safety margin around car
            free_space_threshold (int): Grayscale threshold for free space detection
            min_gap_width_pixels (int): Minimum gap width in pixels to be considered drivable
            smoothing_factor (float): Maximum steering angle change per cycle
        """
        self.car_width = car_width
        self.lookahead_distance = lookahead_distance
        self.safety_buffer = safety_buffer
        self.free_space_threshold = free_space_threshold
        self.min_gap_width_pixels = min_gap_width_pixels
        self.smoothing_factor = smoothing_factor
        
        # State tracking
        self.last_angle = 0.0
        
        # Output directories
        self.output_dir = "images/vision"
        self.test_dir = "images/vision/test"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Region of Interest parameters (as fraction of image dimensions)
        # Focus on road area for gap detection
        self.roi_top_fraction = 0.7     # Start ROI at 70% down from top (road area)
        self.roi_bottom_fraction = 1.0  # End ROI at 100% down from top
        self.roi_left_fraction = 0.0     # Full width for gap detection
        self.roi_right_fraction = 1.0    # Full width for gap detection
        
        # For observability and debugging
        self.debug_info = {}
    
    def get_roi_coordinates(self, image_height, image_width):
        """Calculate ROI coordinates based on image dimensions"""
        top = int(image_height * self.roi_top_fraction)
        bottom = int(image_height * self.roi_bottom_fraction)
        left = int(image_width * self.roi_left_fraction)
        right = int(image_width * self.roi_right_fraction)
        return top, bottom, left, right
        
    def process_image(self, image_array, output_filename=None, is_test=False):
        """
        Main processing function: detect gaps and calculate steering angle
        
        Args:
            image_array (np.ndarray): Input RGB image
            output_filename (str, optional): Name for output visualization file
            is_test (bool): If True, save to test/ subfolder
            
        Returns:
            dict: Contains steering_angle and debug information
        """
        if image_array is None:
            print("No image provided")
            return {'steering_angle': None, 'debug': {}}
        
        # Step 1: Detect free space
        free_space_mask = self.detect_free_space(image_array)
        
        # Step 2: Find gaps in the ROI
        gaps = self.find_gaps(free_space_mask)
        
        # Step 3: Calculate steering angle
        steering_angle = self.calculate_steering_angle(gaps, free_space_mask)
        
        # Step 4: Apply smoothing
        smoothed_angle = self.smooth_steering(steering_angle)
        
        # Store debug information
        self.debug_info = {
            'gaps_found': len(gaps),
            'gaps': gaps,
            'steering_angle_raw': steering_angle,
            'steering_angle_smoothed': smoothed_angle,
            'image_shape': image_array.shape
        }
        
        # Save visualization if requested
        if output_filename:
            self.save_visualization(image_array, free_space_mask, gaps, smoothed_angle, 
                                  output_filename, is_test)
        
        return {
            'steering_angle': smoothed_angle,
            'debug': self.debug_info
        }
    
    def detect_free_space(self, image_array):
        """
        Detect free space areas in the image
        
        Args:
            image_array (np.ndarray): Input RGB image
            
        Returns:
            np.ndarray: Binary mask where white=free space, black=obstacles
        """
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        
        # Apply threshold to separate free space (bright) from obstacles (dark)
        _, free_space = cv2.threshold(gray, self.free_space_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the detection
        kernel = np.ones((5,5), np.uint8)
        # Close small gaps in free space
        free_space = cv2.morphologyEx(free_space, cv2.MORPH_CLOSE, kernel)
        # Remove small noise
        free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel)
        
        return free_space
    
    def find_gaps(self, free_space_mask):
        """
        Find navigable gaps in the free space mask within ROI
        Analyzes entire ROI area, not just bottom scan line
        
        Args:
            free_space_mask (np.ndarray): Binary free space mask
            
        Returns:
            list: List of gaps, each gap is (start_x, end_x, center_x, width)
        """
        height, width = free_space_mask.shape
        top, bottom, left, right = self.get_roi_coordinates(height, width)
        
        # Extract ROI from free space mask
        roi_mask = free_space_mask[top:bottom, left:right]
        roi_height, roi_width = roi_mask.shape
        
        # Calculate free space percentage for each column in ROI
        # A column is considered "navigable" if it has sufficient free space
        min_free_space_ratio = 0.7  # Require 70% of column to be free space
        
        column_navigable = []
        for x in range(roi_width):
            column = roi_mask[:, x]
            free_pixels = np.sum(column > 0)
            free_ratio = free_pixels / roi_height
            column_navigable.append(free_ratio >= min_free_space_ratio)
        
        # Find continuous navigable regions (gaps)
        gaps = []
        current_gap_start = None
        
        for x in range(roi_width):
            if column_navigable[x]:  # Navigable column
                if current_gap_start is None:
                    current_gap_start = x
            else:  # Non-navigable column (obstacle)
                if current_gap_start is not None:
                    # End current gap
                    gap_width = x - current_gap_start
                    if gap_width >= self.min_gap_width_pixels:
                        gap_center = current_gap_start + gap_width // 2
                        # Convert back to full image coordinates
                        gap_start_full = current_gap_start + left
                        gap_end_full = x + left
                        gap_center_full = gap_center + left
                        gaps.append((gap_start_full, gap_end_full, gap_center_full, gap_width))
                    current_gap_start = None
        
        # Handle gap that extends to edge of ROI
        if current_gap_start is not None:
            gap_width = roi_width - current_gap_start
            if gap_width >= self.min_gap_width_pixels:
                gap_center = current_gap_start + gap_width // 2
                gap_start_full = current_gap_start + left
                gap_end_full = roi_width + left
                gap_center_full = gap_center + left
                gaps.append((gap_start_full, gap_end_full, gap_center_full, gap_width))
        
        return gaps
    
    def calculate_steering_angle(self, gaps, free_space_mask):
        """
        Calculate steering angle toward the best gap
        
        Args:
            gaps (list): List of detected gaps
            free_space_mask (np.ndarray): Free space mask for reference
            
        Returns:
            float: Steering angle in radians (None if no gaps)
        """
        if not gaps:
            return None
        
        # Select the largest gap (similar to lidar algorithm)
        best_gap = max(gaps, key=lambda gap: gap[3])  # gap[3] is width
        gap_center_x = best_gap[2]
        
        # Calculate steering angle based on gap center position
        image_width = free_space_mask.shape[1]
        image_center_x = image_width // 2
        
        # Convert pixel offset to steering angle
        # Assume camera FOV of ~60 degrees (similar to typical camera setup)
        camera_fov_deg = 60.0
        camera_fov_rad = radians(camera_fov_deg)
        
        # Calculate angle offset from center
        pixel_offset = gap_center_x - image_center_x
        angle_per_pixel = camera_fov_rad / image_width
        steering_angle = pixel_offset * angle_per_pixel
        
        # Apply scaling factor similar to lidar (0.5 factor for stability)
        steering_angle *= 0.5
        
        return steering_angle
    
    def smooth_steering(self, target_angle):
        """
        Apply smoothing to prevent sudden steering changes
        
        Args:
            target_angle (float): Desired steering angle
            
        Returns:
            float: Smoothed steering angle
        """
        if target_angle is None:
            return None
        
        # Limit change per cycle
        delta = target_angle - self.last_angle
        delta = max(min(delta, self.smoothing_factor), -self.smoothing_factor)
        smoothed_angle = self.last_angle + delta
        
        self.last_angle = smoothed_angle
        return smoothed_angle
    
    def save_visualization(self, original_image, free_space_mask, gaps, steering_angle, 
                          output_filename, is_test=False):
        """
        Save visualization with gap information logged at top of screen
        
        Args:
            original_image (np.ndarray): Original RGB image
            free_space_mask (np.ndarray): Binary free space mask
            gaps (list): Detected gaps
            steering_angle (float): Calculated steering angle
            output_filename (str): Output filename
            is_test (bool): Whether to save in test directory
        """
        # Create visualization image (just original image)
        vis_image = original_image.copy()
        height, width = vis_image.shape[:2]
        
        # Draw ROI rectangle
        top, bottom, left, right = self.get_roi_coordinates(height, width)
        cv2.rectangle(vis_image, (left, top), (right, bottom), (255, 0, 0), 2)  # Blue ROI
        
        # Draw gap boundaries and centers
        for i, (start_x, end_x, center_x, gap_width) in enumerate(gaps):
            # Draw gap boundaries as vertical lines
            cv2.line(vis_image, (start_x, top), (start_x, bottom), (0, 255, 255), 2)  # Yellow start
            cv2.line(vis_image, (end_x, top), (end_x, bottom), (0, 255, 255), 2)    # Yellow end
            
            # Draw gap center
            cv2.line(vis_image, (center_x, top), (center_x, bottom), (0, 255, 0), 2)  # Green center
        
        # Add text information at top of screen (similar to lidar)
        y_offset = 30
        line_height = 25
        
        # Add steering angle
        if steering_angle is not None:
            angle_deg = degrees(steering_angle)
            cv2.putText(vis_image, f'Steering: {angle_deg:.1f}°', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += line_height
        else:
            cv2.putText(vis_image, 'Steering: NO SOLUTION', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += line_height
        
        # Add gap count
        cv2.putText(vis_image, f'Gaps: {len(gaps)}', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height
        
        # Add gap details
        for i, (start_x, end_x, center_x, gap_width) in enumerate(gaps):
            gap_text = f'Gap{i+1}: center={center_x}px, width={gap_width}px'
            cv2.putText(vis_image, gap_text, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # Choose output directory and save
        save_dir = self.test_dir if is_test else self.output_dir
        output_path = os.path.join(save_dir, output_filename)
        cv2.imwrite(output_path, vis_image)
        print(f"Vision gap analysis saved: {output_path}")
    
    def get_debug_info(self):
        """
        Get debugging information from the last processing cycle
        
        Returns:
            dict: Debug information including gaps, angles, etc.
        """
        return self.debug_info.copy()
