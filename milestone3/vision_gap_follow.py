"""
Vision-Based Driving Node using Intel RealSense D435i
"""

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
from math import *
from .vision_base import VisionBase


class VisionFollowing(VisionBase):
    """Vision-based navigation using depth camera gap detection"""

    min_depth = 0.3        # Ignore anything closer than this (too near / invalid)
    max_depth = 3.0        # Ignore anything beyond this (not relevant for local driving)
    
    nav_roi_params = {
        'y0_frac': 0.55,   # Top of navigation ROI (road area)
        'y1_frac': 0.95,   # Bottom of navigation ROI
        'x0_frac': 0.0,    # Full width for gap detection
        'x1_frac': 1.0
    }
    
    # Speed and control parameters
    speed = 1.0            # Base forward speed (m/s)
    max_speed = 1.5
    min_speed = 0.8
    last_angle = 0.0       # Last steering angle used (for smoothing)
    steer_smooth = 0.05    # Maximum change in steering per frame (radians)

    def __init__(self):
        """Initialize the VisionFollowing node using shared vision base."""
        super().__init__('vision_follow_node')

        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive_raw', 10)

        self.get_logger().info("VisionFollowing node initialized (using shared vision base)")

    def on_depth_received(self, depth_image, header):
        """
        Process the depth image to detect free space and decide steering.
        
        Steps:
        1. Extract navigation ROI from processed depth image
        2. Create binary mask of valid "free space" pixels
        3. Find widest contiguous gap of drivable columns
        4. Calculate steering angle toward gap center
        5. Apply smoothing and estimate forward clearance
        6. Publish drive command
        """

        roi, (y0, y1, x0, x1) = self.extract_roi(depth_image, self.nav_roi_params)
        free_mask = self.create_valid_depth_mask(roi, self.min_depth, self.max_depth)
        
        #count free pixels per column
        col_score = np.sum(free_mask, axis=0)
        
        #column is "open" if at least half its pixels are valid
        threshold = 0.5 * roi.shape[0]
        valid_cols = col_score > threshold
        
        if not np.any(valid_cols):
            self.get_logger().warn("No valid free-space detected!")
            self.vehicleControl(0.0, forward_clear=0.3)
            return
            
        #find the widest contiguous gap using gap detection algorithm
        best_start, best_end, width_best = self._find_widest_gap(valid_cols)
        
        gap_center_px = (best_start + best_end) / 2.0
        angle = self.pixel_to_angle(gap_center_px, roi.shape[1])
        smoothed_angle = self._apply_steering_smoothing(angle)
        forward_clear = self.calculate_forward_clearance(depth_image)
        self.vehicleControl(smoothed_angle, forward_clear)
        
        self.get_logger().info(
            f"Steer={degrees(smoothed_angle):.2f}°, gap_width={width_best}, "
            f"forward_clear={forward_clear:.2f}m"
        )
    
    def _find_widest_gap(self, valid_cols):
        best_start, best_end, width_best = 0, 0, 0
        W = len(valid_cols)
        
        i = 0
        # widest gap found from cols
        while i < W:
            if valid_cols[i]:
                j = i
                while j < W and valid_cols[j]:
                    j += 1
                width = j - i
                if width > width_best:
                    width_best, best_start, best_end = width, i, j
                i = j
            else:
                i += 1
                
        return best_start, best_end, width_best
    
    def _apply_steering_smoothing(self, target_angle):
        delta = target_angle - self.last_angle
        delta = max(min(delta, self.steer_smooth), -self.steer_smooth)
        smoothed_angle = self.last_angle + delta
        self.last_angle = smoothed_angle
        return smoothed_angle

    def vehicleControl(self, angle, forward_clear=1.0):
        max_angle = 0.4189  # rad (~24°)
        steer_cmd = max(min(angle, max_angle), -max_angle)
        angle_factor = max(0.5, 1.0 - abs(steer_cmd) / max_angle)
        distance_factor = min(1.5, forward_clear / 2.0)

        dynamic_speed = self.speed * angle_factor * distance_factor
        dynamic_speed = max(self.min_speed, min(dynamic_speed, self.max_speed))
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steer_cmd
        msg.drive.speed = dynamic_speed
        self.publisher.publish(msg)

        self.get_logger().info(
            f"Published → Steering: {steer_cmd:.2f} rad | Speed: {dynamic_speed:.2f} m/s"
        )
 

def main(args=None):
    rclpy.init(args=args)
    node = VisionFollowing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
