"""
Vision Hybrid Gap Following Algorithm for F1Tenth Racing

Abstracted from ROS node for simulation testing. Supports:
1. RGB-only mode (uses edge detection)
2. Depth-only mode (uses depth thresholding)
3. Hybrid mode (prefers depth when available, falls back to RGB)

Key Features:
- Pure computer vision approach without ROS dependencies
- Configurable parameters for all modes
- Comprehensive debug information for visualization
- Processing stage tracking for analysis
"""

import numpy as np
import cv2
from math import atan2, degrees, radians, pi


class VisionHybridFollower:
    """
    Vision hybrid gap following algorithm supporting RGB, depth, and hybrid modes
    """
    
    def __init__(self,
                 # Original RGB parameters (from working vision_gap_follow.py)
                 car_width=0.3,
                 free_space_threshold=120,
                 min_gap_width_meters=0.5,
                 min_gap_width_pixels=30,
                 smoothing_factor=0.1,
                 min_free_space_ratio=0.3,
                 speed=0.8,
                 camera_fov_deg=87.0,
                 corner_threshold=0.5,
                 steering_offset=-0.05,
                 
                 # New depth parameters
                 driving_mode="hybrid",  # "rgb", "depth", or "hybrid"
                 depth_min_valid=0.1,
                 depth_max_valid=5.0,
                 lookahead_distance=1.0,
                 min_depth_gap_width_px=30,
                 
                 # Advanced depth processing parameters
                 close_obstacle_threshold=0.6,  # Closer threshold based on data analysis
                 obstacle_pixel_ratio=0.5,      # More permissive ratio
                 depth_smoothing_kernel=7):      # Larger smoothing to reduce streaks
        """
        Initialize the vision hybrid follower
        
        Args:
            driving_mode (str): "rgb", "depth", or "hybrid"
            All other parameters match the ROS node configuration
        """
        # Core parameters
        self.car_width = car_width
        self.free_space_threshold = free_space_threshold
        self.min_gap_width_meters = min_gap_width_meters
        self.min_gap_width_pixels = min_gap_width_pixels
        self.smoothing_factor = smoothing_factor
        self.min_free_space_ratio = min_free_space_ratio
        self.speed = speed
        self.camera_fov_deg = camera_fov_deg
        self.corner_threshold = corner_threshold
        self.steering_offset = steering_offset
        
        # Depth parameters
        self.driving_mode = driving_mode
        self.depth_min_valid = depth_min_valid
        self.depth_max_valid = depth_max_valid
        self.lookahead_distance = lookahead_distance
        self.min_depth_gap_width_px = min_depth_gap_width_px
        
        # Advanced depth processing parameters
        self.close_obstacle_threshold = close_obstacle_threshold
        self.obstacle_pixel_ratio = obstacle_pixel_ratio
        self.depth_smoothing_kernel = depth_smoothing_kernel
        
        # State tracking
        self.last_angle = 0.0
        self.is_corner = False
        
        # ROI parameters (same as working RGB)
        self.roi_top_fraction = 0.5
        self.roi_bottom_fraction = 1.0
        self.roi_left_fraction = 0.0
        self.roi_right_fraction = 1.0
    
    def get_roi_coordinates(self, image_height, image_width):
        """Calculate ROI coordinates based on image dimensions"""
        top = int(image_height * self.roi_top_fraction)
        bottom = int(image_height * self.roi_bottom_fraction)
        left = int(image_width * self.roi_left_fraction)
        right = int(image_width * self.roi_right_fraction)
        return top, bottom, left, right
    
    def process_image(self, rgb_image=None, depth_image=None):
        """
        Main processing function: detect gaps and calculate steering angle
        
        Args:
            rgb_image (np.ndarray, optional): Input RGB image
            depth_image (np.ndarray, optional): Input depth image
            
        Returns:
            dict: Contains steering_angle, mode used, and comprehensive debug information
        """
        if rgb_image is None and depth_image is None:
            return {'steering_angle': None, 'debug': {'error': 'No input images provided'}}
        
        debug_info = {
            'rgb_processing_stages': {},
            'depth_processing_stages': {},
            'mode': self.driving_mode.upper(),
            'actual_mode_used': None,
            'rgb_gaps': [],
            'depth_gaps': [],
            'selected_gaps': [],
            'corner_detected': False,
            'steering_angle_raw': None,
            'steering_angle_smoothed': None
        }
        
        rgb_gaps = []
        depth_gaps = []
        
        # --- RGB gap detection (identical to working version) ---
        if rgb_image is not None:
            rgb_result = self._process_rgb_pipeline(rgb_image)
            rgb_gaps = rgb_result['gaps']
            debug_info['rgb_processing_stages'] = rgb_result['debug_stages']
            debug_info['rgb_gaps'] = rgb_gaps
            debug_info['corner_detected'] = self.is_corner
        
        # --- Depth gap detection ---
        if depth_image is not None:
            depth_result = self._process_depth_pipeline(depth_image)
            depth_gaps = depth_result['gaps']
            debug_info['depth_processing_stages'] = depth_result['debug_stages']
            debug_info['depth_gaps'] = depth_gaps
        
        # --- Mode selection ---
        selected_gaps = []
        actual_mode = "NONE"
        
        if self.driving_mode == "rgb":
            selected_gaps = rgb_gaps
            actual_mode = "RGB"
        elif self.driving_mode == "depth":
            selected_gaps = depth_gaps
            actual_mode = "DEPTH"
        else:  # hybrid
            if len(depth_gaps) > 0:
                selected_gaps = depth_gaps
                actual_mode = "DEPTH"
            elif len(rgb_gaps) > 0:
                selected_gaps = rgb_gaps
                actual_mode = "RGB"
            else:
                actual_mode = "NONE"
        
        debug_info['actual_mode_used'] = actual_mode
        debug_info['selected_gaps'] = selected_gaps
        
        # --- Steering calculation ---
        steering_angle = self._calculate_steering_angle(selected_gaps, rgb_image)
        debug_info['steering_angle_raw'] = steering_angle
        
        smoothed_angle = self._smooth_steering(steering_angle, override_smoothing=self.is_corner)
        debug_info['steering_angle_smoothed'] = smoothed_angle
        
        return {
            'steering_angle': smoothed_angle,
            'debug': debug_info
        }
    
    def _process_rgb_pipeline(self, rgb_image):
        """
        Process RGB image for gap detection (identical to working version)
        
        Args:
            rgb_image (np.ndarray): Input RGB image
            
        Returns:
            dict: Contains gaps and debug stages
        """
        debug_stages = {}
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        debug_stages['grayscale'] = gray.copy()
        
        # Step 2: Thresholding for free space detection
        _, free_space_mask = cv2.threshold(
            gray, self.free_space_threshold, 255, cv2.THRESH_BINARY)
        debug_stages['threshold_mask'] = free_space_mask.copy()
        
        # Step 3: Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(free_space_mask, cv2.MORPH_CLOSE, kernel)
        debug_stages['closed_mask'] = closed_mask.copy()
        
        final_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
        debug_stages['final_mask'] = final_mask.copy()
        
        # Step 4: Find gaps in ROI
        gaps = self._find_rgb_gaps(final_mask)
        debug_stages['gaps_found'] = len(gaps)
        
        # Step 5: Create visualization
        debug_stages['gap_visualization'] = self._create_rgb_gap_visualization(
            rgb_image, final_mask, gaps)
        
        return {
            'gaps': gaps,
            'debug_stages': debug_stages
        }
    
    def _find_rgb_gaps(self, free_space_mask):
        """
        Find navigable gaps in RGB free space mask (identical to working version)
        
        Args:
            free_space_mask (np.ndarray): Binary free space mask
            
        Returns:
            list: List of gap tuples (start, end, center, width)
        """
        height, width = free_space_mask.shape
        top, bottom, left, right = self.get_roi_coordinates(height, width)
        roi_mask = free_space_mask[top:bottom, left:right]
        roi_height, roi_width = roi_mask.shape
        
        # Analyze column navigability
        column_navigable = []
        for x in range(roi_width):
            column = roi_mask[:, x]
            free_pixels = np.sum(column > 0)
            free_ratio = free_pixels / roi_height
            column_navigable.append(free_ratio >= self.min_free_space_ratio)
        
        # Detect corners based on overall navigability
        total_navigable = sum(column_navigable)
        navigable_ratio = total_navigable / roi_width
        self.is_corner = navigable_ratio < self.corner_threshold
        
        # Find continuous navigable regions (gaps)
        gaps = []
        current_gap_start = None
        
        for x in range(roi_width):
            if column_navigable[x]:  # Navigable column
                if current_gap_start is None:
                    current_gap_start = x
            else:  # Non-navigable column
                if current_gap_start is not None:
                    gap_width_pixels = x - current_gap_start
                    if gap_width_pixels >= self.min_gap_width_pixels:
                        gap_center = current_gap_start + gap_width_pixels // 2
                        # Convert back to full image coordinates
                        gaps.append((current_gap_start + left, x + left, 
                                   gap_center + left, gap_width_pixels))
                    current_gap_start = None
        
        # Handle gap that extends to edge of ROI
        if current_gap_start is not None:
            gap_width_pixels = roi_width - current_gap_start
            if gap_width_pixels >= self.min_gap_width_pixels:
                gap_center = current_gap_start + gap_width_pixels // 2
                gaps.append((current_gap_start + left, roi_width + left,
                           gap_center + left, gap_width_pixels))
        
        return gaps
    
    def _process_depth_pipeline(self, depth_image):
        """
        Process depth image for gap detection with improved obstacle detection
        
        Args:
            depth_image (np.ndarray): Input depth image (in meters)
            
        Returns:
            dict: Contains gaps and debug stages
        """
        debug_stages = {}
        
        # Step 1: Clip depth values to valid range
        clipped_depth = np.clip(depth_image, self.depth_min_valid, self.depth_max_valid)
        debug_stages['clipped_depth'] = clipped_depth.copy()
        
        # Step 2: Extract ROI
        H, W = clipped_depth.shape
        y_top = int(H * self.roi_top_fraction)
        y_bot = int(H * self.roi_bottom_fraction)
        roi_depth = clipped_depth[y_top:y_bot, :]
        debug_stages['roi_depth'] = roi_depth.copy()
        
        # Step 3: Calculate statistics per column with improved obstacle detection
        median_depth = np.zeros(W)
        min_depth = np.zeros(W)
        obstacle_ratio = np.zeros(W)
        
        # Use configurable obstacle detection thresholds
        close_obstacle_threshold = self.close_obstacle_threshold
        obstacle_pixel_ratio = self.obstacle_pixel_ratio
        
        for x in range(W):
            col = roi_depth[:, x]
            valid = col[(col > self.depth_min_valid) & (col < self.depth_max_valid)]
            
            if len(valid) > 0:
                median_depth[x] = np.median(valid)
                min_depth[x] = np.min(valid)
                
                # Count pixels that are close obstacles
                close_pixels = np.sum(valid < close_obstacle_threshold)
                obstacle_ratio[x] = close_pixels / len(valid)
            else:
                median_depth[x] = 0
                min_depth[x] = 0
                obstacle_ratio[x] = 1.0  # No valid data = blocked
        
        debug_stages['median_depth'] = median_depth.copy()
        debug_stages['min_depth'] = min_depth.copy()
        debug_stages['obstacle_ratio'] = obstacle_ratio.copy()
        
        # Step 4: Create improved navigability mask
        # A column is navigable if:
        # Based on data analysis: ROI median ~1.05m, most columns 0.8-2.0m range
        # Close obstacles are minority (~99/640 columns < 0.8m)
        navigable_raw = (
            (obstacle_ratio < 0.5) &  # Allow some close pixels (relaxed threshold)
            (min_depth > 0.3) &  # Min depth not too close (very permissive)
            (median_depth > 0.7)  # Median depth reasonable (data shows ~1.05m median)
        )
        
        # Step 5: Apply smoothing to reduce noise/streaks
        # Use morphological operations to clean up the mask
        navigable_smooth = self._smooth_navigability_mask(navigable_raw, self.depth_smoothing_kernel)
        debug_stages['navigable_raw'] = navigable_raw.copy()
        debug_stages['navigable_mask'] = navigable_smooth.copy()
        
        # Step 6: Find gaps
        gaps = self._find_depth_gaps(navigable_smooth)
        debug_stages['gaps_found'] = len(gaps)
        
        # Step 7: Create visualization
        debug_stages['gap_visualization'] = self._create_depth_gap_visualization(
            depth_image, median_depth, navigable_smooth, gaps)
        
        return {
            'gaps': gaps,
            'debug_stages': debug_stages
        }
    
    def _smooth_navigability_mask(self, navigable_mask, kernel_size=5):
        """
        Smooth the navigability mask to reduce noise and eliminate streaks
        
        Args:
            navigable_mask (np.ndarray): Raw boolean navigability mask
            kernel_size (int): Size of smoothing kernel
            
        Returns:
            np.ndarray: Smoothed boolean navigability mask
        """
        # Convert to uint8 for morphological operations
        mask_uint8 = navigable_mask.astype(np.uint8) * 255
        
        # Create kernel for morphological operations
        kernel = np.ones((1, kernel_size), np.uint8)  # Horizontal kernel to smooth columns
        
        # Apply morphological closing to fill small gaps
        closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Apply morphological opening to remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # Convert back to boolean
        return opened > 127
    
    def _find_depth_gaps(self, navigable):
        """
        Find navigable gaps in depth navigability mask
        
        Args:
            navigable (np.ndarray): Boolean array indicating navigable columns
            
        Returns:
            list: List of gap tuples (start, end, center, width)
        """
        gaps = []
        start = None
        W = len(navigable)
        
        for x in range(W):
            if navigable[x]:
                if start is None:
                    start = x
            else:
                if start is not None:
                    width_px = x - start
                    if width_px >= self.min_depth_gap_width_px:
                        center = start + width_px // 2
                        gaps.append((start, x, center, width_px))
                    start = None
        
        # Handle gap that extends to edge
        if start is not None:
            width_px = W - start
            if width_px >= self.min_depth_gap_width_px:
                center = start + width_px // 2
                gaps.append((start, W, center, width_px))
        
        return gaps
    
    def _calculate_steering_angle(self, gaps, reference_image):
        """
        Calculate steering angle toward the best gap (identical to ROS node)
        
        Args:
            gaps (list): List of detected gap tuples
            reference_image (np.ndarray): Reference image for width calculation
            
        Returns:
            float: Steering angle in radians (None if no gaps)
        """
        if not gaps:
            return None
        
        # Select best gap (widest)
        best_gap = max(gaps, key=lambda gap: gap[3])
        gap_center_x = best_gap[2]
        
        # Calculate steering angle based on gap center position
        if reference_image is not None:
            image_width = reference_image.shape[1]
        else:
            # Fallback - assume standard camera resolution
            image_width = 640
        
        image_center_x = image_width // 2
        camera_fov_rad = radians(self.camera_fov_deg)
        
        # Convert pixel offset to steering angle
        pixel_offset = gap_center_x - image_center_x
        angle_per_pixel = camera_fov_rad / image_width
        steering_angle = pixel_offset * angle_per_pixel
        
        # Apply scaling factor (same as ROS node)
        steering_angle *= 0.5
        
        # Apply steering offset based on corner detection
        if not self.is_corner:
            steering_angle += self.steering_offset * 1.5
        else:
            steering_angle += self.steering_offset * 2.7
        
        return steering_angle
    
    def _smooth_steering(self, target_angle, override_smoothing=False):
        """
        Apply smoothing to prevent sudden steering changes (identical to ROS node)
        
        Args:
            target_angle (float): Desired steering angle
            override_smoothing (bool): If True, skip smoothing (for corners)
            
        Returns:
            float: Smoothed steering angle
        """
        if target_angle is None:
            return None
        
        if override_smoothing:
            self.last_angle = target_angle
            return target_angle
        
        # Limit change per cycle
        delta = target_angle - self.last_angle
        delta = max(min(delta, self.smoothing_factor), -self.smoothing_factor)
        smoothed_angle = self.last_angle + delta
        
        self.last_angle = smoothed_angle
        return smoothed_angle
    
    def _create_rgb_gap_visualization(self, original_image, free_space_mask, gaps):
        """
        Create visualization of RGB gap detection
        
        Args:
            original_image (np.ndarray): Original RGB image
            free_space_mask (np.ndarray): Binary free space mask
            gaps (list): Detected gaps
            
        Returns:
            np.ndarray: Visualization image
        """
        vis_image = original_image.copy()
        height, width = vis_image.shape[:2]
        
        # Draw ROI rectangle
        top, bottom, left, right = self.get_roi_coordinates(height, width)
        cv2.rectangle(vis_image, (left, top), (right, bottom), (255, 0, 0), 2)
        
        # Draw gaps
        for i, gap in enumerate(gaps):
            start_x, end_x, center_x, gap_width = gap
            
            # Gap boundaries
            cv2.line(vis_image, (start_x, top), (start_x, bottom), (0, 255, 255), 2)
            cv2.line(vis_image, (end_x, top), (end_x, bottom), (0, 255, 255), 2)
            # Gap center
            cv2.line(vis_image, (center_x, top), (center_x, bottom), (0, 255, 0), 3)
            
            # Gap label
            cv2.putText(vis_image, f'G{i+1}', (center_x-10, top-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add info text
        mode_text = f"RGB Mode - Gaps: {len(gaps)}"
        if self.is_corner:
            mode_text += " (CORNER)"
        cv2.putText(vis_image, mode_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def _create_depth_gap_visualization(self, depth_image, median_depth, navigable, gaps):
        """
        Create visualization of depth gap detection
        
        Args:
            depth_image (np.ndarray): Original depth image
            median_depth (np.ndarray): Median depth per column
            navigable (np.ndarray): Navigability mask
            gaps (list): Detected gaps
            
        Returns:
            np.ndarray: Visualization image
        """
        # Convert depth to 8-bit for visualization
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vis_image = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        height, width = vis_image.shape[:2]
        
        # Draw ROI rectangle
        top, bottom, left, right = self.get_roi_coordinates(height, width)
        cv2.rectangle(vis_image, (left, top), (right, bottom), (255, 255, 255), 2)
        
        # Draw navigability analysis
        for x in range(width):
            if x < len(navigable):
                if navigable[x]:
                    # Green line for navigable columns
                    cv2.line(vis_image, (x, top), (x, bottom), (0, 255, 0), 1)
                else:
                    # Red line for blocked columns
                    cv2.line(vis_image, (x, top), (x, bottom), (0, 0, 255), 1)
        
        # Draw gaps
        for i, gap in enumerate(gaps):
            start_x, end_x, center_x, gap_width = gap
            
            # Gap boundaries
            cv2.line(vis_image, (start_x, top), (start_x, bottom), (255, 255, 0), 3)
            cv2.line(vis_image, (end_x, top), (end_x, bottom), (255, 255, 0), 3)
            # Gap center
            cv2.line(vis_image, (center_x, top), (center_x, bottom), (255, 0, 255), 4)
            
            # Gap label
            cv2.putText(vis_image, f'D{i+1}', (center_x-10, top-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Add info text
        mode_text = f"DEPTH Mode - Gaps: {len(gaps)} (Lookahead: {self.lookahead_distance}m)"
        cv2.putText(vis_image, mode_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image