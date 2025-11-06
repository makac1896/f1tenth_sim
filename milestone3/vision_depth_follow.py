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
from datetime import datetime
import json
import os


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

        # Setup enhanced logging with JSON output and parameter logging
        self._setup_logging()

        # Subscribe to depth (required)
        self.depth_subscription = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)

        # Optional RGB subscription for visualization/debug (kept inactive)
        # self.rgb_subscription = self.create_subscription(
        #     Image, '/camera/color/image_raw', self.rgb_callback, 10)

        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive_raw', 10)

        # Log parameters at startup
        self._log_parameters()

        self.log_message(f"VisionDepthFollowing node initialized (depth-only)")
        self.log_message(f"Depth ROI: {self.depth_roi_top_fraction}-{self.depth_roi_bottom_fraction}")
        self.log_message(f"Lookahead distance: {self.lookahead_distance}m, Min gap width: {self.min_depth_gap_width_px}px")

    # =====================================================
    # Enhanced Logging System  
    # =====================================================
    def _setup_logging(self):
        """Setup timestamped log files for both text and JSON data"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs('logs', exist_ok=True)
        self.log_file = f'logs/vision_depth_follow_{timestamp}.log'
        self.json_log_file = f'logs/vision_depth_data_{timestamp}.json'
        self.log_data = []

    def _log_parameters(self):
        """Log all algorithm parameters at startup for reproducibility"""
        params = {
            'timestamp': datetime.now().isoformat(),
            'node_type': 'VisionDepthFollowing',
            'parameters': {
                # Core parameters
                'car_width': self.car_width,
                'camera_fov_deg': self.camera_fov_deg,
                'steering_offset': self.steering_offset,
                'smoothing_factor': self.smoothing_factor,
                'speed': self.speed,
                
                # Depth-specific parameters
                'depth_min_valid': self.depth_min_valid,
                'depth_max_valid': self.depth_max_valid,
                'lookahead_distance': self.lookahead_distance,
                'min_depth_gap_width_px': self.min_depth_gap_width_px,
                
                # ROI parameters
                'rgb_roi_top_fraction': self.rgb_roi_top_fraction,
                'rgb_roi_bottom_fraction': self.rgb_roi_bottom_fraction,
                'rgb_roi_left_fraction': self.rgb_roi_left_fraction,
                'rgb_roi_right_fraction': self.rgb_roi_right_fraction,
                'depth_roi_top_fraction': self.depth_roi_top_fraction,
                'depth_roi_bottom_fraction': self.depth_roi_bottom_fraction,
                
                # Algorithm-specific parameters
                'noise_reduction': {
                    'median_blur_kernel': 5,
                    'gaussian_blur_kernel': (3, 3),
                    'spatial_smoothing_sigma': 1.5,
                    'min_valid_pixels_ratio': 0.2,
                    'lookahead_threshold_factor': 0.7
                },
                'morphological_ops': {
                    'horizontal_kernel_size': (1, 5),
                    'clean_kernel_size': (1, 2)
                }
            }
        }
        
        # Write parameter log
        try:
            with open(self.json_log_file, 'w') as f:
                json.dump([params], f, indent=2)
            self.log_message(f"Parameters logged to {self.json_log_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to write parameter log: {e}")

    def log_message(self, message, level='INFO'):
        """
        Enhanced logging to both ROS logger and timestamped file
        
        Args:
            message: The message to log
            level: Log level ('INFO', 'WARN', 'ERROR', 'DEBUG')
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # Write to text log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            self.get_logger().error(f"Failed to write to log file: {e}")
        
        # Write to ROS logger
        if level == 'INFO':
            self.get_logger().info(message)
        elif level == 'WARN':
            self.get_logger().warn(message)
        elif level == 'ERROR':
            self.get_logger().error(message)
        elif level == 'DEBUG':
            self.get_logger().debug(message)

    def log_driving_data(self, steering_angle, gaps_data, additional_info=None):
        """
        Log structured driving data for analysis
        
        Args:
            steering_angle: Final steering angle in radians
            gaps_data: List of detected gaps
            additional_info: Additional debug information
        """
        data_entry = {
            'timestamp': datetime.now().isoformat(),
            'steering_angle_rad': float(steering_angle) if steering_angle is not None else None,
            'steering_angle_deg': float(degrees(steering_angle)) if steering_angle is not None else None,
            'driving_mode': 'DEPTH_ONLY',
            'gaps_count': len(gaps_data) if gaps_data else 0,
            'corner_detected': self.is_corner,
        }
        
        # Add gap details
        if gaps_data:
            gap_details = []
            for gap in gaps_data:
                if isinstance(gap, dict):
                    gap_details.append({
                        'center': float(gap['center']),
                        'width': int(gap['width']),
                        'avg_depth': float(gap.get('avg_depth', 0)),
                        'start': int(gap['start']),
                        'end': int(gap['end'])
                    })
            data_entry['gap_details'] = gap_details
            
            # Log best gap selection
            if gap_details:
                best_gap = max(gaps_data, key=lambda g: g.get('avg_depth', 0))
                data_entry['selected_gap'] = {
                    'center': float(best_gap['center']),
                    'width': int(best_gap['width']),
                    'avg_depth': float(best_gap.get('avg_depth', 0))
                }
        
        if additional_info:
            data_entry.update(additional_info)
        
        self.log_data.append(data_entry)
        
        # Write to JSON file every 10 entries to avoid losing data
        if len(self.log_data) % 10 == 0:
            try:
                with open(self.json_log_file, 'r') as f:
                    existing_data = json.load(f)
                existing_data.extend(self.log_data)
                with open(self.json_log_file, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                self.log_data = []  # Clear buffer
            except Exception as e:
                self.get_logger().error(f"Failed to write JSON log: {e}")

    # =====================================================
    # Depth Callback
    # =====================================================
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
            
            # Enhanced logging with gap details
            gap_info = ""
            if gaps:
                best_gap = max(gaps, key=lambda g: g.get('avg_depth', 0))
                gap_info = f" | Best gap: center={best_gap['center']:.1f}, depth={best_gap['avg_depth']:.2f}m, width={best_gap['width']}px"
            
            corner_status = "Corner" if self.is_corner else "Normal"
            
            self.log_message(
                f"Steering: {degrees(smoothed):.2f}° | "
                f"Gaps: {len(gaps)} | "
                f"{corner_status}{gap_info}"
            )
            
            # Log structured driving data
            self.log_driving_data(smoothed, gaps, {
                'raw_steering_angle': steering_angle,
                'smoothed_steering_angle': smoothed
            })
        else:
            self.vehicle_control(0.0, emergency=True)
            self.log_message("No valid gap found - emergency stop", 'WARN')
            
            # Log failed steering attempt
            self.log_driving_data(None, gaps)

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
