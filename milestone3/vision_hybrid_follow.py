"""
Vision-Based Gap Following Algorithm for F1Tenth Racing
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from math import radians, degrees


class VisionGapFollowing(Node):
    # ===== Original confirmed parameters =====
    car_width = 0.3
    free_space_threshold = 120
    min_gap_width_meters = 0.5
    min_gap_width_pixels = 30
    smoothing_factor = 0.1
    min_free_space_ratio = 0.3
    speed = 0.8
    max_speed = 1.5
    min_speed = 0.8
    camera_fov_deg = 87.0
    corner_threshold = 0.5
    steering_offset = -0.05

    # ===== New depth plug-in parameters =====
    driving_mode = "hybrid"  # "rgb", "depth", or "hybrid"
    depth_min_valid = 0.1
    depth_max_valid = 5.0
    lookahead_distance = 1.0
    min_depth_gap_width_px = 30

    # ===== State tracking =====
    last_angle = 0.0
    is_corner = False

    # ===== ROI (same as working RGB) =====
    roi_top_fraction = 0.5
    roi_bottom_fraction = 1.0
    roi_left_fraction = 0.0
    roi_right_fraction = 1.0

    def __init__(self):
        super().__init__('vision_follow_node')

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None

        self.rgb_subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_subscription = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)

        self.publisher = self.create_publisher(
            AckermannDriveStamped, '/drive_raw', 10)

        self.get_logger().info(
            f"VisionGapFollowing node initialized in {self.driving_mode.upper()} mode")

    # =====================================================
    # RGB and Depth Callbacks
    # =====================================================
    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_vision_data()
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")

    def depth_callback(self, msg):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.depth_image = depth_raw.astype(np.float32) / 1000.0
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    # =====================================================
    # Main Vision Processing
    # =====================================================
    def process_vision_data(self):
        if self.rgb_image is None:
            return

        result = self.process_image(self.rgb_image, self.depth_image)
        steering_angle = result['steering_angle']
        if steering_angle is not None:
            self.vehicle_control(-steering_angle)
            debug = result['debug']
            self.get_logger().info(
                f"Steering: {degrees(steering_angle):.2f}°, "
                f"Mode: {debug['mode']}, "
                f"RGB gaps: {debug['rgb_gaps']}, Depth gaps: {debug['depth_gaps']}"
            )
        else:
            self.vehicle_control(0.0, emergency=True)
            self.get_logger().warn("No valid gap found - emergency stop")

    # =====================================================
    # Combined Logic (keeps RGB pipeline unchanged)
    # =====================================================
    def process_image(self, image_array, depth_image=None):
        # --- RGB gap detection (identical to working version) ---
        free_space_result = self.detect_free_space(image_array)
        free_space_mask = free_space_result['free_space_mask']
        rgb_gap_result = self.find_gaps(free_space_mask, depth_image)
        rgb_gaps = rgb_gap_result['gaps']

        # --- Depth plug-in ---
        depth_gaps = []
        if depth_image is not None:
            depth_result = self.find_gaps_from_depth(depth_image)
            depth_gaps = depth_result['gaps']

        # --- Mode selection ---
        if self.driving_mode == "rgb":
            selected_gaps = rgb_gaps
            mode = "RGB"
        elif self.driving_mode == "depth":
            selected_gaps = depth_gaps
            mode = "DEPTH"
        else:  # hybrid
            selected_gaps = depth_gaps if len(depth_gaps) > 0 else rgb_gaps
            mode = "DEPTH" if len(depth_gaps) > 0 else "RGB"

        # --- Steering ---
        steering_angle = self.calculate_steering_angle(
            selected_gaps, free_space_mask)
        smoothed_angle = self.smooth_steering(steering_angle, override_smoothing=self.is_corner)

        debug = {
            'mode': mode,
            'rgb_gaps': len(rgb_gaps),
            'depth_gaps': len(depth_gaps)
        }

        return {'steering_angle': smoothed_angle, 'debug': debug}

    # =====================================================
    # RGB Free Space (unchanged)
    # =====================================================
    def detect_free_space(self, image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        _, free_space_mask = cv2.threshold(
            gray, self.free_space_threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        free_space_mask = cv2.morphologyEx(free_space_mask, cv2.MORPH_CLOSE, kernel)
        free_space_mask = cv2.morphologyEx(free_space_mask, cv2.MORPH_OPEN, kernel)
        return {'free_space_mask': free_space_mask}

    # =====================================================
    # RGB Gap Detection (identical)
    # =====================================================
    def find_gaps(self, free_space_mask, depth_image=None):
        height, width = free_space_mask.shape
        top, bottom, left, right = self.get_roi_coordinates(height, width)
        roi_mask = free_space_mask[top:bottom, left:right]
        roi_height, roi_width = roi_mask.shape

        column_navigable = []
        for x in range(roi_width):
            column = roi_mask[:, x]
            free_pixels = np.sum(column > 0)
            free_ratio = free_pixels / roi_height
            column_navigable.append(free_ratio >= self.min_free_space_ratio)

        total_navigable = sum(column_navigable)
        navigable_ratio = total_navigable / roi_width
        self.is_corner = navigable_ratio < self.corner_threshold

        # Find continuous navigable regions (gaps)
        gaps = []
        current_gap_start = None
        for x in range(roi_width):
            if column_navigable[x]:
                if current_gap_start is None:
                    current_gap_start = x
            else:
                if current_gap_start is not None:
                    gap_width_pixels = x - current_gap_start
                    if gap_width_pixels >= self.min_gap_width_pixels:
                        gap_center = current_gap_start + gap_width_pixels // 2
                        gaps.append((current_gap_start + left, x + left, gap_center + left, gap_width_pixels))
                    current_gap_start = None
        if current_gap_start is not None:
            gap_width_pixels = roi_width - current_gap_start
            if gap_width_pixels >= self.min_gap_width_pixels:
                gap_center = current_gap_start + gap_width_pixels // 2
                gaps.append((current_gap_start + left, roi_width + left, gap_center + left, gap_width_pixels))

        return {'gaps': gaps}

    # =====================================================
    # Depth Gap Detection
    # =====================================================
    def find_gaps_from_depth(self, depth_image):
        depth_image = np.clip(depth_image, self.depth_min_valid, self.depth_max_valid)
        H, W = depth_image.shape
        y_top = int(H * self.roi_top_fraction)
        y_bot = int(H * self.roi_bottom_fraction)
        roi = depth_image[y_top:y_bot, :]

        median_depth = np.zeros(W)
        for x in range(W):
            col = roi[:, x]
            valid = col[(col > self.depth_min_valid) & (col < self.depth_max_valid)]
            median_depth[x] = np.median(valid) if len(valid) > 0 else 0

        navigable = median_depth >= self.lookahead_distance
        gaps = []
        start = None
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
        if start is not None:
            width_px = W - start
            if width_px >= self.min_depth_gap_width_px:
                center = start + width_px // 2
                gaps.append((start, W, center, width_px))
        return {'gaps': gaps}

    # =====================================================
    # Steering and Control (unchanged)
    # =====================================================
    def calculate_steering_angle(self, gaps, free_space_mask):
        if not gaps:
            return None
        best_gap = max(gaps, key=lambda gap: gap[3])
        gap_center_x = best_gap[2]
        image_width = free_space_mask.shape[1]
        image_center_x = image_width // 2
        camera_fov_rad = radians(self.camera_fov_deg)
        pixel_offset = gap_center_x - image_center_x
        angle_per_pixel = camera_fov_rad / image_width
        steering_angle = pixel_offset * angle_per_pixel
        steering_angle *= 0.5  # same scaling
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

    def get_roi_coordinates(self, image_height, image_width):
        return (
            int(image_height * self.roi_top_fraction),
            int(image_height * self.roi_bottom_fraction),
            int(image_width * self.roi_left_fraction),
            int(image_width * self.roi_right_fraction),
        )


def main(args=None):
    rclpy.init(args=args)
    vision_gap_following = VisionGapFollowing()
    rclpy.spin(vision_gap_following)
    vision_gap_following.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
