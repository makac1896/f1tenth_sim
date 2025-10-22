"""
Simple Edge Detection Algorithm

This algorithm:
1. Loads image 
2. Applies OpenCV edge detection
3. Saves edge image to images/
"""

import numpy as np
import cv2
import os


class SimpleVisionGapFollower:
    """
    Simple edge detection using OpenCV
    """
    
    def __init__(self):
        """Initialize with basic edge detection parameters"""
        self.output_dir = "images/vision"
        self.test_dir = "images/vision/test"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Region of Interest parameters (as fraction of image dimensions)
        # Focus on road area similar to ROS vision algorithm
        self.roi_top_fraction = 0.7     # Start ROI at 70% down from top (road area)
        self.roi_bottom_fraction = 1.0  # End ROI at 100% down from top
        self.roi_left_fraction = 0.0     # Full width for gap detection
        self.roi_right_fraction = 1.0    # Full width for gap detection
    
    def get_roi_coordinates(self, image_height, image_width):
        """Calculate ROI coordinates based on image dimensions"""
        top = int(image_height * self.roi_top_fraction)
        bottom = int(image_height * self.roi_bottom_fraction)
        left = int(image_width * self.roi_left_fraction)
        right = int(image_width * self.roi_right_fraction)
        return top, bottom, left, right
        
    def process_image(self, image_array, output_filename, is_test=False):
        """
        Detect free space/gaps for navigation
        
        Args:
            image_array (np.ndarray): Input image
            output_filename (str): Name for output file
            is_test (bool): If True, save to test/ subfolder
        """
        if image_array is None:
            print("No image provided")
            return
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        
        # Apply threshold to separate free space (bright) from obstacles (dark)
        # Invert so free space becomes white (255) and obstacles become black (0)
        _, free_space = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the free space detection
        kernel = np.ones((5,5), np.uint8)
        # Close small gaps in free space
        free_space = cv2.morphologyEx(free_space, cv2.MORPH_CLOSE, kernel)
        # Remove small noise
        free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel)
        
        # Convert to color image for visualization
        free_space_color = cv2.cvtColor(free_space, cv2.COLOR_GRAY2BGR)
        
        # Color free space areas in green for better visualization
        free_space_color[free_space > 0] = [0, 255, 0]  # Green for free space
        
        # Get ROI coordinates and draw rectangle
        height, width = gray.shape
        top, bottom, left, right = self.get_roi_coordinates(height, width)
        cv2.rectangle(free_space_color, (left, top), (right, bottom), (255, 0, 0), 2)  # Blue ROI
        
        # Choose output directory and save
        save_dir = self.test_dir if is_test else self.output_dir
        output_path = os.path.join(save_dir, output_filename)
        cv2.imwrite(output_path, free_space_color)
        print(f"Free space image with ROI saved: {output_path}")
