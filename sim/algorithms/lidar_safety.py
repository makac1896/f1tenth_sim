"""
ROS-Free Safety Algorithm

Collision detection and safety assessment for F1Tenth racing.
Extracted from ROS2 safety node for use in simulation and testing.

Key Features:
- Time-to-collision calculation
- Forward collision detection
- Speed-based safety assessment
- Observable safety metrics for debugging
"""
import math


class LidarSafetyChecker:
    """
    ROS-Free Safety Assessment for F1Tenth Racing
    
    This class implements collision detection and safety assessment
    without ROS dependencies, making it suitable for simulation.
    """
    
    def __init__(self, 
                 car_radius=0.30,
                 collision_threshold=0.4,
                 warning_threshold=0.5,
                 forward_angle_window=80):  # degrees total
        """
        Initialize the safety checker with configurable parameters
        
        Args:
            car_radius (float): Radius of the car for collision detection
            collision_threshold (float): Time-to-collision threshold for emergency stop
            warning_threshold (float): Time-to-collision threshold for warnings
            forward_angle_window (int): Angular window for forward collision detection (degrees)
        """
        self.car_radius = car_radius
        self.collision_threshold = collision_threshold
        self.warning_threshold = warning_threshold
        self.forward_angle_window = forward_angle_window
        
        # For observability and debugging
        self.debug_info = {}
    
    def assess_safety(self, ranges, current_speed, 
                     angle_min=-2.356194496154785, 
                     angle_max=2.356194496154785,
                     range_min=0.02, 
                     range_max=30.0):
        """
        Main safety assessment function
        
        Args:
            ranges (list): List of range measurements from lidar
            current_speed (float): Current vehicle speed in m/s
            angle_min (float): Minimum angle in radians
            angle_max (float): Maximum angle in radians
            range_min (float): Minimum valid range
            range_max (float): Maximum valid range
            
        Returns:
            dict: Safety assessment results with recommendations
        """
        # Calculate time to collision
        ttc = self.calculate_time_to_collision(ranges, current_speed, 
                                             angle_min, angle_max, 
                                             range_min, range_max)
        
        # Determine safety status
        if ttc < self.collision_threshold:
            safety_status = "EMERGENCY_STOP"
            recommended_action = "stop"
        elif ttc < self.warning_threshold:
            safety_status = "WARNING"
            recommended_action = "brake"
        else:
            safety_status = "CLEAR"
            recommended_action = "continue"
        
        # Store debug information
        self.debug_info = {
            'time_to_collision': ttc,
            'safety_status': safety_status,
            'recommended_action': recommended_action,
            'collision_threshold': self.collision_threshold,
            'warning_threshold': self.warning_threshold,
            'current_speed': current_speed,
            'car_radius': self.car_radius
        }
        
        return {
            'time_to_collision': ttc,
            'safety_status': safety_status,
            'recommended_action': recommended_action,
            'debug': self.debug_info
        }
    
    def calculate_time_to_collision(self, ranges, linear_speed, 
                                  angle_min, angle_max, 
                                  range_min, range_max):
        """
        Calculate minimum time to collision based on current trajectory
        
        Args:
            ranges (list): LIDAR range measurements
            linear_speed (float): Current forward speed
            angle_min (float): Minimum angle in radians
            angle_max (float): Maximum angle in radians
            range_min (float): Minimum valid range
            range_max (float): Maximum valid range
            
        Returns:
            float: Minimum time to collision in seconds
        """
        if linear_speed <= 0:
            return float('inf')
        
        min_ttc = float('inf')
        angle_range = angle_max - angle_min
        
        # Calculate center indices for forward-looking window
        center_idx = len(ranges) // 2
        half_window_angle = math.radians(self.forward_angle_window / 2)
        half_window_indices = int((half_window_angle / angle_range) * len(ranges))
        
        start_idx = max(0, center_idx - half_window_indices)
        end_idx = min(len(ranges), center_idx + half_window_indices)
        
        for i in range(start_idx, end_idx):
            current_range = ranges[i] - self.car_radius
            
            # Skip invalid ranges
            if current_range <= range_min or current_range >= range_max:
                continue
            
            # Calculate beam angle relative to vehicle center
            beam_angle_rad = (i / len(ranges)) * angle_range + angle_min
            beam_angle_deg = math.degrees(beam_angle_rad)
            
            # Calculate range rate (how fast we're approaching this point)
            range_rate = linear_speed * math.cos(beam_angle_rad)
            
            # Skip if we're not approaching this obstacle
            if range_rate <= 0:
                continue
            
            # Calculate time to collision for this beam
            ttc = current_range / range_rate
            
            # Track minimum time to collision
            if ttc < min_ttc:
                min_ttc = ttc
        
        return min_ttc
    
    def get_debug_info(self):
        """
        Get debugging information from the last safety assessment
        
        Returns:
            dict: Debug information including TTC, thresholds, and status
        """
        return self.debug_info.copy()