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
from datetime import datetime
import json
import os


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

    # ===== ROI Parameters =====
    # RGB ROI (keep original working parameters)
    rgb_roi_top_fraction = 0.5
    rgb_roi_bottom_fraction = 1.0
    rgb_roi_left_fraction = 0.0
    rgb_roi_right_fraction = 1.0
    
    # Depth ROI (optimized parameters from testing)
    depth_roi_top_fraction = 0.3
    depth_roi_bottom_fraction = 0.8

    def __init__(self):
        super().__init__('vision_follow_node')

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None

        # Setup enhanced logging with JSON output and parameter logging
        self._setup_logging()

        self.rgb_subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_subscription = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)

        self.publisher = self.create_publisher(
            AckermannDriveStamped, '/drive_raw', 10)

        # Log parameters at startup
        self._log_parameters()
        
        self.log_message(f"VisionGapFollowing node initialized in {self.driving_mode.upper()} mode")
        self.log_message(f"RGB ROI: {self.rgb_roi_top_fraction}-{self.rgb_roi_bottom_fraction}, "
                        f"Depth ROI: {self.depth_roi_top_fraction}-{self.depth_roi_bottom_fraction}")
        self.log_message(f"Lookahead distance: {self.lookahead_distance}m, "
                        f"Min gap width: {self.min_depth_gap_width_px}px")

    # =====================================================
    # Enhanced Logging System
    # =====================================================
    def _setup_logging(self):
        """Setup timestamped log files for both text and JSON data"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs('logs', exist_ok=True)
        self.log_file = f'logs/vision_hybrid_follow_{timestamp}.log'
        self.json_log_file = f'logs/vision_hybrid_data_{timestamp}.json'
        self.log_data = []

    def _log_parameters(self):
        """Log all algorithm parameters at startup for reproducibility"""
        params = {
            'timestamp': datetime.now().isoformat(),
            'node_type': 'VisionHybridFollowing',
            'parameters': {
                # Core parameters
                'car_width': self.car_width,
                'free_space_threshold': self.free_space_threshold,
                'min_gap_width_meters': self.min_gap_width_meters,
                'min_gap_width_pixels': self.min_gap_width_pixels,
                'smoothing_factor': self.smoothing_factor,
                'min_free_space_ratio': self.min_free_space_ratio,
                'speed': self.speed,
                'max_speed': self.max_speed,
                'min_speed': self.min_speed,
                'camera_fov_deg': self.camera_fov_deg,
                'corner_threshold': self.corner_threshold,
                'steering_offset': self.steering_offset,
                
                # Driving mode
                'driving_mode': self.driving_mode,
                
                # Depth parameters
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
                'depth_roi_bottom_fraction': self.depth_roi_bottom_fraction
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
        Enhanced logging to both ROS logger and timestamped file with structured data
        
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

    def log_driving_data(self, steering_angle, gaps_data, debug_info):
        """
        Log structured driving data for analysis
        
        Args:
            steering_angle: Final steering angle in radians
            gaps_data: Dictionary with gap detection results
            debug_info: Additional debug information
        """
        data_entry = {
            'timestamp': datetime.now().isoformat(),
            'steering_angle_rad': float(steering_angle) if steering_angle is not None else None,
            'steering_angle_deg': float(degrees(steering_angle)) if steering_angle is not None else None,
            'driving_mode': debug_info.get('actual_mode_used', 'UNKNOWN'),
            'rgb_gaps_count': debug_info.get('rgb_gaps', 0),
            'depth_gaps_count': debug_info.get('depth_gaps', 0),
            'selected_gaps_count': len(debug_info.get('selected_gaps', [])),
            'corner_detected': debug_info.get('corner_detected', False),
            'raw_steering_angle': float(debug_info.get('steering_angle_raw')) if debug_info.get('steering_angle_raw') is not None else None,
            'smoothed_steering_angle': float(debug_info.get('steering_angle_smoothed')) if debug_info.get('steering_angle_smoothed') is not None else None,
        }
        
        # Add gap details if available
        if debug_info.get('selected_gaps'):
            gap_details = []
            for gap in debug_info['selected_gaps']:
                if isinstance(gap, dict):
                    gap_details.append({
                        'center': float(gap['center']),
                        'width': int(gap['width']),
                        'avg_depth': float(gap.get('avg_depth', 0))
                    })
                else:
                    # Tuple format (start, end, center, width)
                    gap_details.append({
                        'center': float(gap[2]),
                        'width': int(gap[3]),
                        'start': int(gap[0]),
                        'end': int(gap[1])
                    })
            data_entry['gap_details'] = gap_details
        
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
    # RGB and Depth Callbacks
    # ====================================================="
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
        debug = result['debug']
        
        if steering_angle is not None:
            self.vehicle_control(-steering_angle)
            
            # Enhanced logging with gap details
            gap_info = ""
            if debug['selected_gaps']:
                if isinstance(debug['selected_gaps'][0], dict):
                    # Depth gaps with avg_depth
                    best_gap = debug['selected_gaps'][0]
                    gap_info = f" | Best gap: center={best_gap['center']:.1f}, depth={best_gap['avg_depth']:.2f}m"
                else:
                    # RGB gaps
                    best_gap = debug['selected_gaps'][0]
                    gap_info = f" | Best gap: center={best_gap[2]}, width={best_gap[3]}px"
            
            corner_status = "Corner" if debug.get('corner_detected', False) else "Normal"
            
            self.log_message(
                f"Steering: {degrees(steering_angle):.2f}° | "
                f"Mode: {debug['actual_mode_used']} | "
                f"RGB: {debug['rgb_gaps']}, Depth: {debug['depth_gaps']} | "
                f"{corner_status}{gap_info}"
            )
            
            # Log structured driving data
            self.log_driving_data(steering_angle, {}, debug)
        else:
            self.vehicle_control(0.0, emergency=True)
            self.log_message("No valid gap found - emergency stop", 'WARN')
            
            # Log failed steering attempt
            self.log_driving_data(None, {}, debug)

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
            'depth_gaps': len(depth_gaps),
            'selected_gaps': selected_gaps,
            'corner_detected': self.is_corner
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
    # Enhanced Depth Gap Detection with Noise Reduction
    # =====================================================
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
            # Fallback to moving average
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
        
        # Create horizontal kernel to smooth column-wise streaks  
        kernel_horizontal = np.ones((1, 5), np.uint8)
        closed = cv2.morphologyEx(mask_2d, cv2.MORPH_CLOSE, kernel_horizontal)
        
        # Use smaller opening kernel to preserve more navigable areas
        kernel_clean = np.ones((1, 2), np.uint8)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_clean)
        
        return (opened.flatten() > 127)
    
    def _find_depth_gaps(self, navigable_mask, median_depths=None):
        """Find gaps in the navigable mask with depth information"""
        gaps = []
        
        # Find connected components of True values
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
                    
                    # Return as dictionary format for enhanced gap selection
                    gaps.append({
                        'start': start,
                        'end': end, 
                        'center': center,
                        'width': width,
                        'avg_depth': avg_depth
                    })
        
        # Sort by width (largest first) initially
        gaps.sort(key=lambda g: g['width'], reverse=True)
        return gaps

    # =====================================================
    # Steering and Control (unchanged)
    # =====================================================
    def calculate_steering_angle(self, gaps, free_space_mask):
        if not gaps:
            return None
        
        # Select best gap - prefer deepest gap for depth mode, widest for RGB
        if isinstance(gaps[0], dict) and 'avg_depth' in gaps[0]:
            # For depth gaps, prefer the deepest one for safety
            best_gap = max(gaps, key=lambda gap: gap['avg_depth'])
            gap_center_x = best_gap['center']
        else:
            # For RGB gaps (tuple format), use widest
            best_gap = max(gaps, key=lambda gap: gap[3] if isinstance(gap, tuple) else gap['width'])
            gap_center_x = best_gap[2] if isinstance(best_gap, tuple) else best_gap['center']
        
        image_width = free_space_mask.shape[1]
        image_center_x = image_width // 2
        camera_fov_rad = radians(self.camera_fov_deg)
        pixel_offset = gap_center_x - image_center_x
        angle_per_pixel = camera_fov_rad / image_width
        steering_angle = pixel_offset * angle_per_pixel
        steering_angle *= 0.5  # same scaling
        
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

    def get_roi_coordinates(self, image_height, image_width):
        """Get RGB ROI coordinates (keep original working parameters)"""
        return (
            int(image_height * self.rgb_roi_top_fraction),
            int(image_height * self.rgb_roi_bottom_fraction),
            int(image_width * self.rgb_roi_left_fraction),
            int(image_width * self.rgb_roi_right_fraction),
        )


def main(args=None):
    rclpy.init(args=args)
    vision_gap_following = VisionGapFollowing()
    rclpy.spin(vision_gap_following)
    vision_gap_following.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
