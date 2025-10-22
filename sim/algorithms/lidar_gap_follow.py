"""
ROS-Free Lidar Gap Following Algorithm

Simplified, modular implementation of gap-following for F1Tenth racing.
Extracted from ROS2 node for use in simulation and testing.

Key Features:
- Pure Python implementation (no ROS dependencies)
- Configurable parameters for different scenarios
- Clear separation of concerns: detection, calculation, control
- Observable intermediate results for debugging
"""
import numpy as np
from math import atan2, degrees, radians, sin, pi


class LidarGapFollower:
    """
    ROS-Free Gap Following Algorithm for F1Tenth Racing
    
    This class implements the core gap detection and steering calculation
    without ROS dependencies, making it suitable for simulation and testing.
    """
    
    def __init__(self, 
                 car_width=0.3,
                 lookahead_distance=1.0,
                 safety_buffer=0.1,
                 disparity_threshold=3.0,
                 disparity_factor=2.5,
                 disparity_safety_buffer=15,
                 smoothing_factor=0.05):  # Original: max_delta = 0.05 rad per cycle
        """
        Initialize the gap follower with configurable parameters
        
        Args:
            car_width (float): Width of the car in meters
            lookahead_distance (float): How far ahead to look for gaps
            safety_buffer (float): Additional safety margin around car
            disparity_threshold (float): Minimum distance change to trigger disparity extension
            disparity_factor (float): Scaling factor for disparity extension
            disparity_safety_buffer (int): Additional angular buffer in degrees
            smoothing_factor (float): Maximum steering angle change per cycle
        """
        self.car_width = car_width
        self.lookahead_distance = lookahead_distance
        self.safety_buffer = safety_buffer
        self.disparity_threshold = disparity_threshold
        self.disparity_factor = disparity_factor
        self.disparity_safety_buffer = disparity_safety_buffer
        self.smoothing_factor = smoothing_factor
        
        # State tracking
        self.last_angle = 0.0
        
        # For observability and debugging
        self.debug_info = {}
    
    def process_lidar_data(self, ranges, angle_min=-2.356194496154785, angle_max=2.356194496154785):
        """
        Main processing function that takes lidar data and returns steering angle
        
        Args:
            ranges (list): List of range measurements from lidar
            angle_min (float): Minimum angle in radians (from lidar metadata)
            angle_max (float): Maximum angle in radians (from lidar metadata)
            
        Returns:
            dict: Contains steering_angle and debug information
        """
        # Step 1: Apply disparity extension
        extended_ranges = self.disparity_extender(ranges)
        
        # Step 2: Find gaps
        gaps = self.find_gaps(extended_ranges, angle_min, angle_max)
        
        # Step 3: Calculate steering angle
        steering_angle = self.calculate_steering_angle(gaps, extended_ranges, angle_min, angle_max)
        
        # Store debug information
        self.debug_info = {
            'raw_ranges_count': len(ranges),
            'gaps_found': len(gaps),
            'gaps': gaps,
            'extended_ranges': extended_ranges,
            'steering_angle_raw': steering_angle,
            'steering_angle_smoothed': None if steering_angle is None else self.smooth_steering(steering_angle)
        }
        
        return {
            'steering_angle': self.debug_info['steering_angle_smoothed'],
            'debug': self.debug_info
        }
    
    def disparity_extender(self, lidar_ranges):
        """
        Apply disparity extension to handle obstacles and sudden depth changes
        
        Args:
            lidar_ranges (list): Raw LIDAR range measurements
            
        Returns:
            list: Modified range data with extended obstacle regions
        """
        extended_ranges = list(lidar_ranges)
        
        for i in range(len(lidar_ranges) - 1):
            d1 = lidar_ranges[i]
            d2 = lidar_ranges[i + 1]
            
            # Check for sudden distance changes
            if abs(d1 - d2) > self.disparity_threshold:
                closer_distance = min(d1, d2)
                
                # Calculate angular width for extension
                angle_width = atan2(self.car_width / self.disparity_factor, closer_distance)
                num_beams_to_extend = int((angle_width * len(lidar_ranges)) / (270 * pi/180))
                
                # Add safety buffer
                num_beams_to_extend += int((self.disparity_safety_buffer * len(lidar_ranges)) / 270)
                
                # Extend obstacle based on which side is closer
                if d1 < d2:
                    # Obstacle on left side
                    for k in range(1, num_beams_to_extend + 1):
                        if i + k < len(extended_ranges):
                            extended_ranges[i + k] = min(extended_ranges[i + k], closer_distance)
                else:
                    # Obstacle on right side
                    for k in range(1, num_beams_to_extend + 1):
                        if i - k >= 0:
                            extended_ranges[i - k] = min(extended_ranges[i - k], closer_distance)
        
        return extended_ranges
    
    def find_gaps(self, lidar_ranges, angle_min, angle_max):
        """
        Find navigable gaps in the lidar data - EXACT COPY of original algorithm
        
        Original algorithm assumes 270° FOV over 1080 indices with -135° center
        """
        min_gap_width = self.car_width + 2 * self.safety_buffer
        gaps = []
        current_gap = []

        for angle_idx, distance in enumerate(lidar_ranges):
            if distance > self.lookahead_distance:
                # Store angle index of gap, not distance
                current_gap.append(angle_idx)
            else:
                if current_gap:  # If we have a gap
                    # Original calculation: 270 degrees over 1080 LIDAR indices - 135 (half 270) such that center is 0
                    start_angle = current_gap[0] * 270 / 1080 - 135 
                    end_angle = current_gap[-1] * 270 / 1080 - 135
                    gap_width = abs(end_angle - start_angle)

                    # Check if the gap is wide enough - original algorithm logic
                    if gap_width > degrees(atan2(min_gap_width, min(lidar_ranges[current_gap[0]], lidar_ranges[current_gap[-1]]))):
                        gaps.append(current_gap)
                    current_gap = []

        # Handle last gap if still open - original algorithm logic
        if current_gap:
            start_angle = current_gap[0] * 270 / 1080 - 135
            end_angle = current_gap[-1] * 270 / 1080 - 135
            gap_width = abs(end_angle - start_angle)

            # Checks that the gap width (in degrees) is large enough for the car
            if gap_width > degrees(atan2(min_gap_width, min(lidar_ranges[current_gap[0]], lidar_ranges[current_gap[-1]]))):
                gaps.append(current_gap)
        
        return gaps
    
    def calculate_steering_angle(self, gaps, lidar_ranges, angle_min, angle_range):
        """
        Calculate steering angle based on detected gaps - EXACT COPY of original algorithm
        """
        if not gaps:  # If no gaps found
            return None

        # Pick the largest gap - original algorithm
        best_gap = max(gaps, key=len)
        middle_idx = best_gap[len(best_gap) // 2]  # Target the middle of the largest gap

        # Convert index to angle in radians - ORIGINAL CALCULATION
        angle = (middle_idx * 270 / 1080 - 135) * pi / 180
        distance = lidar_ranges[middle_idx]
        lookahead_distance = min(distance, self.lookahead_distance)

        # Original steering angle calculation
        steering_angle = 0.5 * angle

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
    
    def get_debug_info(self):
        """
        Get debugging information from the last processing cycle
        
        Returns:
            dict: Debug information including gaps, ranges, angles
        """
        return self.debug_info.copy()