import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import numpy as np
import math
from .vision_base import VisionBase


class VisionSafetyNode(VisionBase):
    collision_threshold = 0.4    # Time threshold for emergency stop (seconds)
    brake_threshold = 0.5        # Time threshold for braking (seconds)
    car_radius = 0.30           # Vehicle safety radius (meters)
    
    safety_roi_params = {
        'y0_frac': 0.4,    
        'y1_frac': 0.9,    
        'x0_frac': 0.3,    
        'x1_frac': 0.7
    }

    def __init__(self):
        super().__init__('vision_safety_node')

        self.latest_odom = None
        self.latest_drive_raw = None
        self.linear_x_speed = 0.0
        
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',  # Vehicle odometry
            self.odom_callback,
            10
        )
        
        self.drive_subscription = self.create_subscription(
            AckermannDriveStamped,
            '/drive_raw',
            self.drive_raw_callback,
            10
        )
        
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )
        
        self.get_logger().info("Vision Safety Node initialized (using shared vision base)")

    def on_depth_received(self, depth_image, header):
        self.check_for_collision()
    
    def on_rgb_received(self, rgb_image, header):
        pass

    def odom_callback(self, msg):
        self.latest_odom = msg
        self.linear_x_speed = msg.twist.twist.linear.x
        self.check_for_collision()

    def drive_raw_callback(self, msg):
        self.latest_drive_raw = msg
        self.check_for_collision()

    def check_for_collision(self):
        if self.latest_depth is None:
            self.get_logger().debug("No depth data available for safety check")
            return
        
        if self.latest_odom is None:
            self.get_logger().info('No odometry data - using fallback speed')
            self.linear_x_speed = 1.0 
        else:
            self.linear_x_speed = self.latest_odom.twist.twist.linear.x
            
        time_to_collision = self.calculate_time_to_collision_vision(
            self.latest_odom, self.latest_depth
        )
        
        if time_to_collision < self.collision_threshold:  # < 0.4s
            self.publish_stop()
            self.get_logger().warn(f"EMERGENCY STOP - Collision in {time_to_collision:.2f}s!")
            
        elif time_to_collision < self.brake_threshold:  # < 0.5s
            self.get_logger().warn(f"Obstacle detected - Braking (TTC: {time_to_collision:.2f}s)")
            self.publish_brake()
            
        elif self.latest_drive_raw is None:
            self.get_logger().debug("No navigation commands received yet")
            
        else:
            # Safe to pass through navigation commands
            self.drive_publisher.publish(self.latest_drive_raw)
            self.get_logger().debug(f"Path clear - TTC: {time_to_collision:.2f}s")

    def calculate_time_to_collision_vision(self, odom_msg, depth_img):
        if depth_img is None:
            return float('inf')

        roi, (y0, y1, x0, x1) = self.extract_roi(depth_img, self.safety_roi_params)
        roi_safe = roi - self.car_radius
        valid_mask = self.create_valid_depth_mask(roi_safe)
        
        if not np.any(valid_mask):
            self.get_logger().debug("No valid obstacles detected in safety ROI")
            return float('inf')
            
        # valid distances
        valid_distances = roi_safe[valid_mask]
        min_ttc = float('inf')
        valid_coords = np.where(valid_mask)
        
        for i, (row, col) in enumerate(zip(valid_coords[0], valid_coords[1])):
            distance = valid_distances[i]
            # convert the pixels to an angle and determine where an approaching object is relative to car centerline
            pixel_angle_rad = self.pixel_to_angle(col, roi.shape[1])
            closure_rate = self.linear_x_speed * math.cos(pixel_angle_rad)
            
            if closure_rate <= 0:
                continue
    
            ttc = distance / closure_rate
        
            if ttc < min_ttc:
                min_ttc = ttc
        
        self.get_logger().info(f'Vision TTC: {min_ttc:.3f}s (speed: {self.linear_x_speed:.2f}m/s)')
        return min_ttc

    def publish_stop(self):
        self.get_logger().fatal("EMERGENCY STOP ACTIVATED - SHUTTING DOWN SYSTEM")
        rclpy.shutdown()

    def publish_brake(self):
        brake_msg = AckermannDriveStamped()
        brake_msg.drive.speed = 0.0
        brake_msg.drive.steering_angle = 0.0
        
        self.drive_publisher.publish(brake_msg)
        self.get_logger().warn("Emergency braking applied - vehicle stopped")


def main(args=None):
    rclpy.init(args=args)
    vision_safety = VisionSafetyNode()
    
    try:
        rclpy.spin(vision_safety)
    except KeyboardInterrupt:
        vision_safety.get_logger().info("Vision Safety Node interrupted by user")
    finally:
        vision_safety.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
