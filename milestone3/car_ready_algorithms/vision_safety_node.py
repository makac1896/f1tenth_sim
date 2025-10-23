"""
Vision AEB Safety System

Core production algorithm for F1Tenth autonomous vehicle.
Uses free space detection with optional depth enhancement for obstacle avoidance.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
import cv2
import numpy as np


class VisionSafetyNode(Node):
    min_free_space_percentage = 0.3  # minimum required free space fraction in ROI
    safety_buffer = 0.1  # additional safety margin (0.0 to 1.0)
    min_safe_distance = 1.5  # minimum safe distance in meters (for depth)
    critical_distance = 0.3  # critical distance triggering emergency stop (for depth)
    aeb_fov_fraction = 0.4  # fraction of image width to check for AEB (0.4 = center 40%)
    depth_min_valid = 0.1  # minimum valid depth reading in meters (filters sensor noise)
    depth_max_valid = 10.0  # maximum valid depth reading in meters (filters unreliable readings)
    depth_percentile = 5.0  # percentile to use for depth safety check (10.0 = 10th percentile)
    
    roi_top_fraction = 0.7
    roi_bottom_fraction = 1.0
    roi_left_fraction = 0.0
    roi_right_fraction = 1.0

    def __init__(self):
        super().__init__('vision_safety_node')
        
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.latest_drive_cmd = None
        
        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.rgb_callback,
            10
        )
        
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        
        self.drive_subscription = self.create_subscription(
            AckermannDriveStamped,
            '/drive_raw',
            self.drive_callback,
            10
        )
        
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        self.timer = self.create_timer(0.05, self.check_safety_conditions)
        
        self.get_logger().info("VisionSafetyNode initialized - monitoring for obstacles")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")

    def depth_callback(self, msg):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.depth_image = depth_raw.astype(np.float32) / 1000.0  # mm to meters
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def drive_callback(self, msg):
        self.latest_drive_cmd = msg

    def check_safety_conditions(self):
        if self.rgb_image is None or self.latest_drive_cmd is None:
            return
            
        is_safe, reason, safety_data = self.check_safety(self.rgb_image, self.depth_image)
        
        if not is_safe:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.speed = 0.0
            self.publisher.publish(drive_msg)
            self.get_logger().warn(
                f"EMERGENCY STOP: {reason}, "
                f"Free space: {safety_data.get('free_space_percentage', 0):.1%}, "
                f"Method: {safety_data.get('method', 'unknown')}, "
                f"Min depth: {safety_data.get('min_depth', 'N/A')}"
            )
        else:
            self.publisher.publish(self.latest_drive_cmd)
            self.get_logger().info(
                f"Safe: {reason}, "
                f"Free space: {safety_data.get('free_space_percentage', 0):.1%}, "
                f"Method: {safety_data.get('method', 'unknown')}, "
                f"Original cmd: steer={self.latest_drive_cmd.drive.steering_angle:.2f}, speed={self.latest_drive_cmd.drive.speed:.2f}"
            )

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
        
        # depth enhancement if available
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

def main(args=None):
    rclpy.init(args=args)
    vision_safety_node = VisionSafetyNode()
    rclpy.spin(vision_safety_node)
    vision_safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()