"""
Shared Vision Processing Base Class for F1/10 Autonomous Racing

This module provides common vision processing functionality shared between
vision-based navigation and safety nodes. It eliminates code duplication
and provides a unified interface for RealSense depth camera processing.

Shared Components:
- Depth image processing and conversion
- ROI (Region of Interest) extraction 
- Camera calibration parameters
- Pixel-to-angle coordinate transformations
- Common ROS2 subscription management
- Depth validation and filtering utilities

Usage:
Inherit from VisionBase to create vision-enabled nodes:
- VisionFollowing extends VisionBase (navigation)
- VisionSafetyNode extends VisionBase (collision avoidance)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from math import *


class VisionBase(Node):
    # RealSense D435i Camera Parameters
    hfov_deg = 87.0             # Horizontal field of view (degrees)
    vfov_deg = 58.0             # Vertical field of view (degrees)  
    
    # Depth Processing Parameters  
    min_depth = 0.1             # Minimum valid depth (meters)
    max_depth = 5.0             # Maximum processing depth (meters)
    depth_scale = 1000.0        # Scale factor for depth conversion (mm to m)
    
    # Image Processing Parameters
    median_filter_size = 5      # Kernel size for noise reduction
    gaussian_blur_size = 3      # Kernel size for smoothing
    
    def __init__(self, node_name):
        super().__init__(node_name)
        
        # OpenCV bridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()
        
        # Latest sensor data storage
        self.latest_depth = None
        self.latest_rgb = None
        
        # Set up camera subscriptions
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',  # RealSense depth topic
            self._depth_callback_internal,
            10  # QoS depth
        )
        
        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # RealSense RGB topic  
            self._rgb_callback_internal,
            10  # QoS depth
        )
        
        self.get_logger().info(f"Vision Base initialized for {node_name}")
        
    
    def _depth_callback_internal(self, msg):
        try:
            # convert image we get from car to opencv
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # convert depths to meters, not sure if the depth image contains this??
            processed_depth = self._process_raw_depth(depth_img)
            self.latest_depth = processed_depth
            self.on_depth_received(processed_depth, msg.header)
            
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")


    # base class func, inherited classes can try get this
    def _rgb_callback_internal(self, msg):
        try:
            rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb = rgb_img
            self.on_rgb_received(rgb_img, msg.header)
            
        except Exception as e:
            self.get_logger().error(f"RGB processing error: {e}")
    
    def _process_raw_depth(self, raw_depth):
        # Convert to float and scale to meters
        if raw_depth.dtype != np.float32:
            depth_m = raw_depth.astype(np.float32) / self.depth_scale
        else:
            depth_m = raw_depth
            
        # Apply noise filtering
        depth_filtered = cv2.medianBlur(
            (depth_m * 1000).astype(np.uint16), 
            self.median_filter_size
        ).astype(np.float32) / 1000.0
        
        # Gaussian smoothing for stability
        depth_smooth = cv2.GaussianBlur( #refer to the slides for this cz I they mention some good examples
            depth_filtered,
            (self.gaussian_blur_size, self.gaussian_blur_size),
            0
        )
        
        return depth_smooth
    
    # ROI is region of interest
    def extract_roi(self, image, roi_params):
        H, W = image.shape[:2]
        
        y0 = int(roi_params.get('y0_frac', 0.0) * H)
        y1 = int(roi_params.get('y1_frac', 1.0) * H)
        x0 = int(roi_params.get('x0_frac', 0.0) * W)
        x1 = int(roi_params.get('x1_frac', 1.0) * W)
        y0, y1 = max(0, y0), min(H, y1)
        x0, x1 = max(0, x0), min(W, x1)
        roi = image[y0:y1, x0:x1]
        
        return roi, (y0, y1, x0, x1)
    
    # need to test this cz was not sure on how to get this to work properly
    def pixel_to_angle(self, pixel_x, image_width, horizontal_fov_deg=None):
        if horizontal_fov_deg is None:
            horizontal_fov_deg = self.hfov_deg
            
        # Normalize to [-1, 1]
        center_x = (image_width - 1) / 2.0
        normalized = (pixel_x - center_x) / center_x
        
        # Convert to angle
        angle_rad = normalized * radians(horizontal_fov_deg / 2.0)
        
        return angle_rad
    
    def create_valid_depth_mask(self, depth_image, min_depth=None, max_depth=None):
        
        if min_depth is None:
            min_depth = self.min_depth
        if max_depth is None:
            max_depth = self.max_depth
            
        # Check for finite values and depth range
        valid_mask = np.logical_and(
            np.isfinite(depth_image),
            np.logical_and(
                depth_image > min_depth,
                depth_image < max_depth
            )
        )
        
        return valid_mask
    
    def calculate_forward_clearance(self, depth_image, center_width_frac=0.1):
        H, W = depth_image.shape
        
        # define center slice
        center_w = int(center_width_frac * W)
        x_start = (W - center_w) // 2
        x_end = x_start + center_w
        
        # extract center region
        center_slice = depth_image[:, x_start:x_end]
        
        # calculate mean of valid pixels
        valid_mask = self.create_valid_depth_mask(center_slice)
        
        if np.any(valid_mask):
            forward_clear = np.mean(center_slice[valid_mask])
        else:
            forward_clear = self.min_depth 
            
        return forward_clear
    
    # since we have both gap follow and safety using safety we make a base class cz they can implement these depending on use case
    def on_depth_received(self, depth_image, header):
        pass
    
    def on_rgb_received(self, rgb_image, header):
        pass
