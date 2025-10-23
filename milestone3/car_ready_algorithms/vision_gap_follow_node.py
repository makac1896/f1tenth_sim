"""
Vision-Based Gap Following Algorithm for F1Tenth Racing
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from math import atan2, degrees, radians, pi


class VisionGapFollowing(Node):
    car_width = 0.3
    free_space_threshold = 120
    min_gap_width_meters = 0.5
    min_gap_width_pixels = 50  # fallback when no depth data available
    smoothing_factor = 0.05
    min_free_space_ratio = 0.55
    speed = 1.0  # base speed in m/s
    max_speed = 1.5
    min_speed = 0.8
    camera_fov_deg = 87.0  # RealSense D435i horizontal FOV
    
    # State tracking
    last_angle = 0.0
    
    # Region of Interest parameters (as fraction of image dimensions)
    roi_top_fraction = 0.7     # Start ROI at 70% down from top (road area)
    roi_bottom_fraction = 1.0  # End ROI at 100% down from top
    roi_left_fraction = 0.0     # Full width for gap detection
    roi_right_fraction = 1.0    # Full width for gap detection

    def __init__(self):
        super().__init__('vision_follow_node')
        
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        
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
        
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive_raw', 10)
        
        self.get_logger().info("VisionGapFollowing node initialized")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_vision_data()
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")

    def depth_callback(self, msg):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.depth_image = depth_raw.astype(np.float32) / 1000.0  # mm to meters
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def process_vision_data(self):
        if self.rgb_image is None:
            return
            
        result = self.process_image(self.rgb_image, self.depth_image)
        steering_angle = result['steering_angle']
        
        if steering_angle is not None:
            self.vehicle_control(steering_angle)
            debug = result['debug']
            self.get_logger().info(
                f"Steering: {degrees(steering_angle):.2f}°, Gaps: {debug['gaps_found']}, "
                f"Raw: {degrees(debug['steering_angle_raw']):.2f}°, "
                f"Depth: {debug['depth_available']}, "
                f"Best gap: {debug['gaps'][0] if debug['gaps'] else 'None'}"
            )
        else:
            self.vehicle_control(0.0, emergency=True)
            self.get_logger().warn("No valid gap found - emergency stop")

    def process_image(self, image_array, depth_image=None):
        if image_array is None:
            return {'steering_angle': None, 'debug': {}}
        
        free_space_result = self.detect_free_space(image_array)
        free_space_mask = free_space_result['free_space_mask']
        
        gap_result = self.find_gaps(free_space_mask, depth_image)
        gaps = gap_result['gaps']
        
        # if we have depth data further analyze gaps by finding deepest/widest gap
        if depth_image is not None and len(gaps) > 0:
            height, width = free_space_mask.shape
            roi = self.get_roi_coordinates(height, width)
            enhanced_gaps = self._analyze_gap_depths(gaps, depth_image, roi)
        else:
            enhanced_gaps = []
            for gap_start, gap_end, gap_center, gap_width in gaps:
                enhanced_gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'center': gap_center,
                    'width': gap_width,
                    'width_meters': None,  # No depth data available
                    'median_depth': None,
                    'max_depth': None,
                    'min_depth': None,
                    'depth_variance': None
                })
        
        steering_angle = self.calculate_steering_angle(enhanced_gaps, free_space_mask, use_depth=depth_image is not None)

        smoothed_angle = self.smooth_steering(steering_angle)
        
        # debug info
        debug_info = {
            'gaps_found': len(enhanced_gaps),
            'gaps': enhanced_gaps,
            'steering_angle_raw': steering_angle,
            'steering_angle_smoothed': smoothed_angle,
            'image_shape': image_array.shape,
            'processing_stages': free_space_result,
            'gap_debug': gap_result['debug_data'],
            'depth_available': depth_image is not None
        }
        
        return {
            'steering_angle': smoothed_angle,
            'debug': debug_info
        }

    def get_roi_coordinates(self, image_height, image_width):
        top = int(image_height * self.roi_top_fraction)
        bottom = int(image_height * self.roi_bottom_fraction)
        left = int(image_width * self.roi_left_fraction)
        right = int(image_width * self.roi_right_fraction)
        return top, bottom, left, right
    
    def pixels_to_distance(self, pixel_width, depth_m, camera_fov_deg=87.0):
        if depth_m <= 0:
            return 0.0
        
        # pixel_width / image_width = real_width / (2 * depth * tan(fov/2))
        camera_fov_rad = radians(camera_fov_deg)
        image_width_pixels = 960
        fov_width_at_depth = 2.0 * depth_m * np.tan(camera_fov_rad / 2.0)
        meters_per_pixel = fov_width_at_depth / image_width_pixels
        real_width_m = pixel_width * meters_per_pixel
        
        return real_width_m
    
    def detect_free_space(self, image_array):
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        
        # anything above threshold is marked as free space (bright = road, dark = obstacles)
        _, free_space_mask = cv2.threshold(gray, self.free_space_threshold, 255, cv2.THRESH_BINARY)
        
        # clean up noise using morphological operations (same as AEB system, https://www.geeksforgeeks.org/python/python-opencv-morphological-operations/)
        kernel = np.ones((5,5), np.uint8)
        free_space_mask = cv2.morphologyEx(free_space_mask, cv2.MORPH_CLOSE, kernel)
        free_space_mask = cv2.morphologyEx(free_space_mask, cv2.MORPH_OPEN, kernel)
        
        # visualize free space for debugging
        green_black_image = np.zeros_like(image_array)
        green_black_image[free_space_mask > 0] = [0, 255, 0]  # green for free space
        green_black_image[free_space_mask == 0] = [0, 0, 0]   # black for obstacles
        
        # debug info for simulation scripts
        return {
            'grayscale': gray,
            'edges': free_space_mask,
            'thick_edges': free_space_mask,
            'thresholded': free_space_mask,
            'free_space_mask': free_space_mask,
            'green_black': green_black_image
        }
    
    def find_gaps(self, free_space_mask, depth_image=None):
        height, width = free_space_mask.shape
        top, bottom, left, right = self.get_roi_coordinates(height, width)
        
        roi_mask = free_space_mask[top:bottom, left:right]
        roi_height, roi_width = roi_mask.shape
        
        column_navigable = []
        column_ratios = []
        
        for x in range(roi_width):
            column = roi_mask[:, x] # take all rows at idx x
            free_pixels = np.sum(column > 0)
            free_ratio = free_pixels / roi_height
            column_ratios.append(free_ratio)
            column_navigable.append(free_ratio >= self.min_free_space_ratio)
        
        # debug info for simulator
        column_debug_image = self._create_column_debug_visualization(
            free_space_mask, column_navigable, column_ratios, top, bottom, left, right)
        gap_debug_data = {
            'column_navigable': column_navigable,
            'column_ratios': column_ratios,
            'column_debug_image': column_debug_image,
            'roi_coords': (top, bottom, left, right)
        }
        
        # Find continuous navigable regions (gaps)
        gaps = []
        current_gap_start = None
        
        for x in range(roi_width):
            if column_navigable[x]:
                if current_gap_start is None:
                    current_gap_start = x
            else:
                if current_gap_start is not None:
                    # end gap
                    gap_width_pixels = x - current_gap_start
                    gap_center = current_gap_start + gap_width_pixels // 2
                    
                    # convert back to full image coordinates
                    gap_start_full = current_gap_start + left
                    gap_end_full = x + left
                    gap_center_full = gap_center + left
                    
                    # check if gap is wide enough (0.5m minimum)
                    is_gap_valid = False
                    if depth_image is not None:
                        gap_center_depth = depth_image[top + roi_height//2, gap_center_full]
                        if gap_center_depth > 0.1:  # valid depth reading (0.1m minimum)
                            gap_width_meters = self.pixels_to_distance(gap_width_pixels, gap_center_depth)
                            is_gap_valid = gap_width_meters >= self.min_gap_width_meters
                        else:
                            # fallback to pixel-based
                            is_gap_valid = gap_width_pixels >= self.min_gap_width_pixels
                    else:
                        # no depth data, use pixel-based fallback
                        is_gap_valid = gap_width_pixels >= self.min_gap_width_pixels
                    
                    if is_gap_valid:
                        gaps.append((gap_start_full, gap_end_full, gap_center_full, gap_width_pixels))
                    current_gap_start = None
        
        # handle gap that extends to edge of ROI
        if current_gap_start is not None:
            gap_width_pixels = roi_width - current_gap_start
            gap_center = current_gap_start + gap_width_pixels // 2
            gap_start_full = current_gap_start + left
            gap_end_full = roi_width + left
            gap_center_full = gap_center + left
            is_gap_valid = False

            if depth_image is not None:
                gap_center_depth = depth_image[top + roi_height//2, gap_center_full]
                if gap_center_depth > 0.1:
                    gap_width_meters = self.pixels_to_distance(gap_width_pixels, gap_center_depth)
                    is_gap_valid = gap_width_meters >= self.min_gap_width_meters
                else:
                    is_gap_valid = gap_width_pixels >= self.min_gap_width_pixels
            else:
                is_gap_valid = gap_width_pixels >= self.min_gap_width_pixels
            
            if is_gap_valid:
                gaps.append((gap_start_full, gap_end_full, gap_center_full, gap_width_pixels))
        
        return {
            'gaps': gaps,
            'debug_data': gap_debug_data
        }

    # debug function
    def _create_column_debug_visualization(self, free_space_mask, column_navigable, 
                                         column_ratios, top, bottom, left, right):
        debug_image = cv2.cvtColor(free_space_mask, cv2.COLOR_GRAY2BGR)
        roi_width = right - left
        
        for x in range(roi_width):
            full_x = x + left
            ratio = column_ratios[x]
            navigable = column_navigable[x]
        
            if navigable:
                # green vertical line for navigable columns
                cv2.line(debug_image, (full_x, top), (full_x, bottom), (0, 255, 0), 1)
            else:
                # red vertical line for blocked columns
                cv2.line(debug_image, (full_x, top), (full_x, bottom), (0, 0, 255), 1)
            
            bar_height = int(ratio * 20)
            bar_start_y = bottom - bar_height
            cv2.line(debug_image, (full_x, bottom), (full_x, bar_start_y), (255, 255, 0), 1)
        
        cv2.putText(debug_image, 'Green=Navigable, Red=Blocked', 
                   (10, top-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_image, f'Yellow bars=Free ratio (threshold: {self.min_free_space_ratio:.1f})', 
                   (10, top-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_image
    
    def _analyze_gap_depths(self, gaps, depth_image, roi):
        if not gaps or depth_image is None:
            return gaps
        
        enhanced_gaps = []
        top, bottom, left, right = roi
        
        for gap_start, gap_end, gap_center, gap_width in gaps:
            gap_depth_region = depth_image[top:bottom, gap_start:gap_end]
            valid_depths = gap_depth_region[(gap_depth_region > 0.1) & (gap_depth_region < 5.0)]
            
            if len(valid_depths) > 0:
                median_depth = np.median(valid_depths)
                max_depth = np.percentile(valid_depths, 95)  
                min_depth = np.percentile(valid_depths, 5)
            else:
                median_depth = 1.0
                max_depth = 1.0
                min_depth = 1.0

            gap_width_meters = self.pixels_to_distance(gap_width, median_depth) if median_depth else None
            
            enhanced_gaps.append({
                'start': gap_start,
                'end': gap_end,
                'center': gap_center,
                'width': gap_width,
                'width_meters': gap_width_meters,
                'median_depth': median_depth,
                'max_depth': max_depth,
                'min_depth': min_depth,
                'depth_variance': max_depth - min_depth
            })
        
        return enhanced_gaps
    
    def calculate_steering_angle(self, gaps, free_space_mask, use_depth=False):
        if not gaps:
            return None
        
        if use_depth and any(gap['median_depth'] is not None for gap in gaps):
            best_gap = self._select_best_gap_with_depth(gaps)
        else:
            best_gap = max(gaps, key=lambda gap: gap['width'])
        
        gap_center_x = best_gap['center']
        image_width = free_space_mask.shape[1]
        image_center_x = image_width // 2
        camera_fov_deg = 87.0
        camera_fov_rad = radians(camera_fov_deg)
        
        pixel_offset = gap_center_x - image_center_x
        angle_per_pixel = camera_fov_rad / image_width
        steering_angle = pixel_offset * angle_per_pixel
    
        steering_angle *= 0.5
        
        return steering_angle
    
    def _select_best_gap_with_depth(self, gaps):
        if len(gaps) == 1:
            return gaps[0]
        
        scored_gaps = []
        
        for gap in gaps:
            # find the most suitable gap based on width and depth: useful on corners where a deeper gap is more preferable to a wider gap
            if gap['median_depth'] is None:
                score = gap['width']
            else:
                width_score = gap['width'] / 50.0
                depth_score = min(gap['median_depth'], 5.0)  # cap at 5m for scoring
                stability_penalty = gap['depth_variance'] if gap['depth_variance'] else 0
                # score: 60% width, 40% depth
                score = (0.6 * width_score + 0.4 * depth_score) - (0.1 * stability_penalty)
            
            scored_gaps.append((score, gap))
        
        return max(scored_gaps, key=lambda x: x[0])[1]
    
    def smooth_steering(self, target_angle):
        if target_angle is None:
            return None
        
        delta = target_angle - self.last_angle
        delta = max(min(delta, self.smoothing_factor), -self.smoothing_factor)
        smoothed_angle = self.last_angle + delta
        
        self.last_angle = smoothed_angle
        return smoothed_angle

    def vehicle_control(self, steering_angle, emergency=False):
        max_angle = 0.4189  # ~24 degrees max steering
        
        if emergency:
            final_steering = 0.0
            final_speed = 0.0
        else:
            final_steering = max(min(steering_angle, max_angle), -max_angle)
            
            # Speed control: slow down for sharp turns
            angle_factor = max(0.5, 1.0 - abs(final_steering) / max_angle)
            
            forward_clearance = self.get_forward_clearance()
            distance_factor = min(1.5, forward_clearance / 2.0) if forward_clearance else 1.0
            
            dynamic_speed = self.speed * angle_factor * distance_factor
            final_speed = max(self.min_speed, min(dynamic_speed, self.max_speed))
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = final_steering
        drive_msg.drive.speed = final_speed
        self.publisher.publish(drive_msg)
        
        self.get_logger().info(
            f"Control → Steering: {degrees(final_steering):.2f}°, Speed: {final_speed:.2f} m/s"
        )

    def get_forward_clearance(self):
        if self.depth_image is None:
            return 2.0  # default safe distance
            
        height, width = self.depth_image.shape
        center_y = int(height * 0.8)  # look ahead on road
        center_x = width // 2
        window_size = 20  # pixels around center
        
        y_start = max(0, center_y - window_size // 2)
        y_end = min(height, center_y + window_size // 2)
        x_start = max(0, center_x - window_size)
        x_end = min(width, center_x + window_size)
        
        forward_region = self.depth_image[y_start:y_end, x_start:x_end]
        valid_depths = forward_region[(forward_region > 0.1) & (forward_region < 10.0)]
        
        if len(valid_depths) > 0:
            return float(np.percentile(valid_depths, 10))  # 10th percentile for safety
        else:
            return 2.0  # default


def main(args=None):
    rclpy.init(args=args)
    vision_gap_following = VisionGapFollowing()
    rclpy.spin(vision_gap_following)
    vision_gap_following.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()