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
                 free_space_threshold=80,
                 min_gap_width_pixels=50,
                 smoothing_factor=0.05,
                 min_free_space_ratio=0.7):
        """
        Initialize the vision gap follower
        
        Args:
            car_width (float): Width of the car in meters
            free_space_threshold (int): Grayscale threshold for free space detection
            min_gap_width_pixels (int): Minimum gap width in pixels to be considered drivable
            smoothing_factor (float): Maximum steering angle change per cycle
            min_free_space_ratio (float): Minimum ratio of free pixels in column to be navigable
        """
        self.car_width = car_width
        self.free_space_threshold = free_space_threshold
        self.min_gap_width_pixels = min_gap_width_pixels
        self.smoothing_factor = smoothing_factor
        self.min_free_space_ratio = min_free_space_ratio
        
        # State tracking
        self.last_angle = 0.0
        
        # Region of Interest parameters (as fraction of image dimensions)
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
        
    def process_image(self, image_array, depth_image=None):
        """
        Main processing function: detect gaps and calculate steering angle
        
        Args:
            image_array (np.ndarray): Input RGB image
            depth_image (np.ndarray, optional): Input depth image for enhanced gap selection
            
        Returns:
            dict: Contains steering_angle and debug information
        """
        if image_array is None:
            return {'steering_angle': None, 'debug': {}}
        
        # Step 1: Detect free space with intermediate stages
        free_space_result = self.detect_free_space(image_array)
        free_space_mask = free_space_result['free_space_mask']
        
        # Step 2: Find gaps in the ROI
        gap_result = self.find_gaps(free_space_mask)
        gaps = gap_result['gaps']
        
        # Step 3: Enhance gaps with depth information if available
        if depth_image is not None and len(gaps) > 0:
            height, width = free_space_mask.shape
            roi = self.get_roi_coordinates(height, width)
            # Convert simple gap tuples to enhanced gap dictionaries
            enhanced_gaps = self._analyze_gap_depths(gaps, depth_image, roi)
        else:
            # Convert simple gap tuples to dictionaries for consistency
            enhanced_gaps = []
            for gap_start, gap_end, gap_center, gap_width in gaps:
                enhanced_gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'center': gap_center,
                    'width': gap_width,
                    'median_depth': None,
                    'max_depth': None,
                    'min_depth': None,
                    'depth_variance': None
                })
        
        # Step 4: Calculate steering angle with depth consideration
        steering_angle = self.calculate_steering_angle(enhanced_gaps, free_space_mask, use_depth=depth_image is not None)
        
        # Step 5: Apply smoothing
        smoothed_angle = self.smooth_steering(steering_angle)
        
        # Debug information including processing stages
        debug_info = {
            'gaps_found': len(enhanced_gaps),
            'gaps': enhanced_gaps,
            'steering_angle_raw': steering_angle,
            'steering_angle_smoothed': smoothed_angle,
            'image_shape': image_array.shape,
            'processing_stages': free_space_result,
            'gap_debug': gap_result['debug_data'],
            'depth_available': depth_image is not None
        }
        
        return {
            'steering_angle': smoothed_angle,
            'debug': debug_info
        }
    
    def detect_free_space(self, image_array):
        """
        Detect free space using edge detection approach
        
        Args:
            image_array (np.ndarray): Input RGB image
            
        Returns:
            dict: Contains intermediate processing stages and final result
        """
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        
        # Step 1: Edge detection using Canny
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Step 2: Dilate edges to make obstacles thicker
        kernel = np.ones((3,3), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Step 3: Create free space mask (invert edges - no edges = free space)
        free_space_mask = cv2.bitwise_not(thick_edges)
        
        # Step 4: Apply morphological operations to clean up
        kernel = np.ones((5,5), np.uint8)
        # Close small gaps in free space
        free_space_mask = cv2.morphologyEx(free_space_mask, cv2.MORPH_CLOSE, kernel)
        # Remove small noise
        free_space_mask = cv2.morphologyEx(free_space_mask, cv2.MORPH_OPEN, kernel)
        
        # Step 5: Create green/black visualization
        green_black_image = np.zeros_like(image_array)
        green_black_image[free_space_mask > 0] = [0, 255, 0]  # Green for free space
        green_black_image[free_space_mask == 0] = [0, 0, 0]   # Black for obstacles
        
        return {
            'grayscale': gray,
            'edges': edges,
            'thick_edges': thick_edges,
            'free_space_mask': free_space_mask,
            'green_black': green_black_image
        }
    
    def find_gaps(self, free_space_mask):
        """
        Find navigable gaps in the free space mask within ROI
        Analyzes entire ROI area, not just bottom scan line
        
        Args:
            free_space_mask (np.ndarray): Binary free space mask
            
        Returns:
            dict: Contains gaps list and debug visualization data
        """
        height, width = free_space_mask.shape
        top, bottom, left, right = self.get_roi_coordinates(height, width)
        
        # Extract ROI from free space mask
        roi_mask = free_space_mask[top:bottom, left:right]
        roi_height, roi_width = roi_mask.shape
        
        # Calculate free space percentage for each column in ROI
        # A column is considered "navigable" if it has sufficient free space
        
        column_navigable = []
        column_ratios = []
        
        for x in range(roi_width):
            column = roi_mask[:, x]
            free_pixels = np.sum(column > 0)
            free_ratio = free_pixels / roi_height
            column_ratios.append(free_ratio)
            column_navigable.append(free_ratio >= self.min_free_space_ratio)
        
        # Create column navigability visualization
        column_debug_image = self._create_column_debug_visualization(
            free_space_mask, column_navigable, column_ratios, top, bottom, left, right)
        
        # Store debug data
        gap_debug_data = {
            'column_navigable': column_navigable,
            'column_ratios': column_ratios,
            'column_debug_image': column_debug_image,
            'roi_coords': (top, bottom, left, right)
        }
        
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
        
        return {
            'gaps': gaps,
            'debug_data': gap_debug_data
        }
    
    def _create_column_debug_visualization(self, free_space_mask, column_navigable, 
                                         column_ratios, top, bottom, left, right):
        """Create visualization showing column navigability analysis"""
        # Start with the free space mask as base
        debug_image = cv2.cvtColor(free_space_mask, cv2.COLOR_GRAY2BGR)
        
        # Draw column analysis in ROI
        roi_width = right - left
        
        for x in range(roi_width):
            full_x = x + left
            ratio = column_ratios[x]
            navigable = column_navigable[x]
            
            # Color code the column based on navigability
            if navigable:
                # Green vertical line for navigable columns
                cv2.line(debug_image, (full_x, top), (full_x, bottom), (0, 255, 0), 1)
            else:
                # Red vertical line for blocked columns  
                cv2.line(debug_image, (full_x, top), (full_x, bottom), (0, 0, 255), 1)
            
            # Add ratio visualization as a bar at bottom
            bar_height = int(ratio * 20)  # Scale to 20 pixels max
            bar_start_y = bottom - bar_height
            cv2.line(debug_image, (full_x, bottom), (full_x, bar_start_y), (255, 255, 0), 1)
        
        # Add legend text
        cv2.putText(debug_image, 'Green=Navigable, Red=Blocked', 
                   (10, top-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_image, f'Yellow bars=Free ratio (threshold: {self.min_free_space_ratio:.1f})', 
                   (10, top-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_image
    
    def _analyze_gap_depths(self, gaps, depth_image, roi):
        """
        Analyze depth information for each gap to find deepest free space
        
        Args:
            gaps (list): List of gap tuples (start, end, center, width)
            depth_image (np.ndarray): Depth image
            roi (tuple): ROI coordinates (top, bottom, left, right)
            
        Returns:
            list: Enhanced gaps with depth information
        """
        if not gaps or depth_image is None:
            return gaps
        
        enhanced_gaps = []
        top, bottom, left, right = roi
        
        for gap_start, gap_end, gap_center, gap_width in gaps:
            # Sample depth values from the entire gap region within ROI
            # Use full ROI height for comprehensive depth analysis
            gap_depth_region = depth_image[top:bottom, gap_start:gap_end]
            
            # Filter out invalid depth values (0 or very close)
            valid_depths = gap_depth_region[(gap_depth_region > 100) & (gap_depth_region < 5000)]
            
            if len(valid_depths) > 0:
                # Use median depth to avoid outliers
                median_depth = np.median(valid_depths)
                max_depth = np.max(valid_depths)
                min_depth = np.min(valid_depths)
            else:
                # No valid depth data - assign neutral values
                median_depth = 1000  # Assume moderate distance
                max_depth = 1000
                min_depth = 1000
            
            enhanced_gaps.append({
                'start': gap_start,
                'end': gap_end,
                'center': gap_center,
                'width': gap_width,
                'median_depth': median_depth,
                'max_depth': max_depth,
                'min_depth': min_depth,
                'depth_variance': max_depth - min_depth
            })
        
        return enhanced_gaps
    
    def calculate_steering_angle(self, gaps, free_space_mask, use_depth=False):
        """
        Calculate steering angle toward the best gap
        
        Args:
            gaps (list): List of detected gap dictionaries
            free_space_mask (np.ndarray): Free space mask for reference
            use_depth (bool): Whether to use depth information for gap selection
            
        Returns:
            float: Steering angle in radians (None if no gaps)
        """
        if not gaps:
            return None
        
        # Select the best gap based on available information
        if use_depth and any(gap['median_depth'] is not None for gap in gaps):
            # Depth-based selection: prefer deeper gaps at corners, wider gaps on straights
            best_gap = self._select_best_gap_with_depth(gaps)
        else:
            # Width-based selection (fallback)
            best_gap = max(gaps, key=lambda gap: gap['width'])
        
        gap_center_x = best_gap['center']
        
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
    
    def _select_best_gap_with_depth(self, gaps):
        """
        Select the best gap considering both width and depth information
        
        Args:
            gaps (list): List of gap dictionaries with depth information
            
        Returns:
            dict: Best gap for navigation
        """
        if len(gaps) == 1:
            return gaps[0]
        
        # Score each gap based on multiple criteria
        scored_gaps = []
        
        for gap in gaps:
            if gap['median_depth'] is None:
                # No depth data, score based on width only
                score = gap['width']
            else:
                # Combined scoring: width + depth + stability
                width_score = gap['width'] / 50.0  # Normalize by minimum gap width
                depth_score = min(gap['median_depth'] / 1000.0, 5.0)  # Cap at 5m for scoring
                
                # Penalize high depth variance (unstable depth readings)
                stability_penalty = gap['depth_variance'] / 1000.0 if gap['depth_variance'] else 0
                
                # For corners (where we want to move into free space early),
                # prefer deeper gaps. For straights, prefer wider gaps.
                # Use a balanced approach: 60% width, 40% depth
                score = (0.6 * width_score + 0.4 * depth_score) - (0.1 * stability_penalty)
            
            scored_gaps.append((score, gap))
        
        # Return the gap with the highest score
        return max(scored_gaps, key=lambda x: x[0])[1]
    
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
    

