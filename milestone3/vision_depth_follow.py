"""
Depth-Only Vision Gap Following Node (car-ready)

This node implements the depth processing pipeline (noise reduction,
streak cleaning, depth-aware gap selection) extracted from the hybrid
implementation and packaged as a standalone ROS2 node suitable for
running on the car. It keeps the same control and steering logic as the
hybrid node but focuses on depth input and uses a depth-specific ROI.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from math import radians, degrees


class VisionDepthFollowing(Node):
    # ===== Parameters (tunable) =====
    car_width = 0.3
    camera_fov_deg = 87.0
    steering_offset = -0.05
    smoothing_factor = 0.1

    # Depth-specific parameters
    depth_min_valid = 0.1
    depth_max_valid = 5.0
    lookahead_distance = 1.0
    min_depth_gap_width_px = 30

    # Speeds
    speed = 0.8

    # ROI parameters (two sets: RGB kept for compatibility, depth used by this node)
    rgb_roi_top_fraction = 0.5
    rgb_roi_bottom_fraction = 1.0
    rgb_roi_left_fraction = 0.0
    rgb_roi_right_fraction = 1.0

    depth_roi_top_fraction = 0.3
    depth_roi_bottom_fraction = 0.8

    # State
    last_angle = 0.0
    is_corner = False

    def __init__(self):
        super().__init__('vision_depth_follow_node')
        self.bridge = CvBridge()
        self.depth_image = None

        # Subscribe to depth (required)
        self.depth_subscription = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)

        # Optional RGB subscription for visualization/debug (kept inactive)
        # self.rgb_subscription = self.create_subscription(
        #     Image, '/camera/color/image_raw', self.rgb_callback, 10)

        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive_raw', 10)

        self.get_logger().info(
            f"VisionDepthFollowing node initialized (depth-only)")
        self.get_logger().info(
            f"Depth ROI: {self.depth_roi_top_fraction}-{self.depth_roi_bottom_fraction}")
        self.get_logger().info(
            f"Lookahead distance: {self.lookahead_distance}m, Min gap width: {self.min_depth_gap_width_px}px")

    # Depth callback
    def depth_callback(self, msg):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.depth_image = depth_raw.astype(np.float32) / 1000.0
            self.process_vision_data()
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def process_vision_data(self):
        if self.depth_image is None:
            return

        # Run depth pipeline
        result = self.find_gaps_from_depth(self.depth_image)
        gaps = result['gaps']

        # Create a dummy reference mask to provide image width to steering calc
        W = self.depth_image.shape[1]
        dummy_mask = np.zeros((1, W), dtype=np.uint8)

        steering_angle = self.calculate_steering_angle(gaps, dummy_mask)
        smoothed = self.smooth_steering(steering_angle, override_smoothing=self.is_corner)

        if smoothed is not None:
            # Use same sign convention as hybrid node (invert if necessary)
            self.vehicle_control(-smoothed)
            self.get_logger().info(
                f"Steering: {degrees(smoothed):.2f}° | Gaps: {len(gaps)}")
        else:
            self.vehicle_control(0.0, emergency=True)
            self.get_logger().warn("No valid gap found - emergency stop")

    # ---------------- Depth pipeline (copied from hybrid) ----------------
    def find_gaps_from_depth(self, depth_image):
        """
        Enhanced depth processing with noise reduction and streak cleaning
        """
        # Step 1: Apply noise reduction to raw depth image
        filtered_depth = cv2.medianBlur((depth_image * 1000).astype(np.uint16), 5).astype(np.float32) / 1000.0
        smooth_depth = cv2.GaussianBlur(filtered_depth, (3, 3), 0)

        # Step 2: Clip depth values to valid range
        clipped_depth = np.clip(smooth_depth, self.depth_min_valid, self.depth_max_valid)

        # Step 3: Extract ROI using depth-specific parameters
        H, W = clipped_depth.shape
        y_top = int(H * self.depth_roi_top_fraction)
        y_bot = int(H * self.depth_roi_bottom_fraction)
        roi_depth = clipped_depth[y_top:y_bot, :]

        # Step 4: Calculate robust statistics per column
        median_depth = np.zeros(W)
        valid_pixel_count = np.zeros(W)

        for x in range(W):
            col = roi_depth[:, x]
            valid = col[(col > self.depth_min_valid) & (col < self.depth_max_valid)]

            if len(valid) > 0:
                median_depth[x] = np.median(valid)
                valid_pixel_count[x] = len(valid)
            else:
                median_depth[x] = 0
                valid_pixel_count[x] = 0

        # Step 5: Apply spatial smoothing to reduce streaks
        try:
            from scipy.ndimage import gaussian_filter1d
            smoothed_median = gaussian_filter1d(median_depth, sigma=1.5)
        except ImportError:
            kernel_size = 3
            kernel = np.ones(kernel_size) / kernel_size
            smoothed_median = np.convolve(median_depth, kernel, mode='same')

        # Step 6: Create improved navigability mask
        min_valid_pixels = roi_depth.shape[0] * 0.2  # 20% of ROI height must be valid
        lookahead_threshold = self.lookahead_distance * 0.7  # 70% of lookahead distance

        navigable_raw = (
            (smoothed_median >= lookahead_threshold) &
            (valid_pixel_count >= min_valid_pixels)
        )

        # Step 7: Apply morphological operations to clean up streaks
        navigable_clean = self._clean_navigability_streaks(navigable_raw)

        # Step 8: Find gaps with depth information
        gaps = self._find_depth_gaps(navigable_clean, smoothed_median)

        return {'gaps': gaps}

    def _clean_navigability_streaks(self, navigable_raw):
        """Clean up streaky navigability mask using morphological operations"""
        mask_uint8 = navigable_raw.astype(np.uint8) * 255
        mask_2d = mask_uint8.reshape(1, -1)

        kernel_horizontal = np.ones((1, 5), np.uint8)
        closed = cv2.morphologyEx(mask_2d, cv2.MORPH_CLOSE, kernel_horizontal)

        kernel_clean = np.ones((1, 2), np.uint8)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_clean)

        return (opened.flatten() > 127)

    def _find_depth_gaps(self, navigable_mask, median_depths=None):
        """Find gaps in the navigable mask with depth information"""
        gaps = []

        diff = np.diff(np.concatenate(([False], navigable_mask, [False])))
        starts = np.where(diff)[0]

        for i in range(0, len(starts), 2):
            if i + 1 < len(starts):
                start = starts[i]
                end = starts[i + 1] - 1
                width = end - start + 1

                # Filter by minimum width
                if width >= self.min_depth_gap_width_px:
                    center = (start + end) / 2

                    # Calculate average depth of the gap
                    avg_depth = 0.0
                    if median_depths is not None:
                        gap_depths = median_depths[start:end+1]
                        avg_depth = np.mean(gap_depths) if len(gap_depths) > 0 else 0.0

                    gaps.append({
                        'start': start,
                        'end': end,
                        'center': center,
                        'width': width,
                        'avg_depth': avg_depth
                    })

        gaps.sort(key=lambda g: g['width'], reverse=True)
        return gaps

    # ---------------- Steering & Control (copied) ----------------
    def calculate_steering_angle(self, gaps, reference_mask):
        """
        Calculate steering angle toward the best gap (identical to sim algorithm)
        """
        if not gaps:
            return None
        
        # Select best gap - prefer deepest gap among those that meet minimum width
        if isinstance(gaps[0], dict) and 'avg_depth' in gaps[0]:
            # For depth gaps, prefer the deepest one for safety
            best_gap = max(gaps, key=lambda gap: gap['avg_depth'])
        else:
            # For RGB gaps, use widest
            best_gap = max(gaps, key=lambda gap: gap['width'] if isinstance(gap, dict) else gap[3])
        
        gap_center_x = best_gap['center'] if isinstance(best_gap, dict) else best_gap[2]
        
        # Calculate steering angle based on gap center position
        if reference_mask is not None:
            image_width = reference_mask.shape[1]
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

    def smooth_steering(self, target_angle, override_smoothing=False):
        if target_angle is None:
            return None
        if override_smoothing:
            self.last_angle = target_angle
            return target_angle
        delta = target_angle - self.last_angle
        delta = max(min(delta, self.smoothing_factor), -self.smoothing_factor)
        smoothed_angle = self.last_angle + delta
        self.last_angle = smoothed_angle
        return smoothed_angle

    def vehicle_control(self, steering_angle, emergency=False):
        max_angle = 0.4189
        if emergency:
            final_steering = 0.0
            final_speed = 0.0
        else:
            final_steering = max(min(steering_angle, max_angle), -max_angle)
            final_speed = self.speed
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = final_steering
        msg.drive.speed = final_speed
        self.publisher.publish(msg)
        self.get_logger().info(
            f"Control → Steering: {degrees(final_steering):.2f}°, Speed: {final_speed:.2f} m/s"
        )


def main(args=None):
    rclpy.init(args=args)
    node = VisionDepthFollowing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
