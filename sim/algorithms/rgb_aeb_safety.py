"""
RGB-Only Vision AEB Safety System

Since we only have RGB data, this uses free space detection as a safety proxy.
Assumes that areas with insufficient free space are unsafe to navigate.
"""

import numpy as np
import cv2
import os


class VisionAEBSafety:
    """AEB safety system using RGB vision data with optional depth enhancement"""
    
    def __init__(self, min_free_space_percentage=0.3, safety_buffer=0.2, 
                 min_safe_distance=1.5, critical_distance=0.8, aeb_fov_fraction=0.4):
        """
        Initialize AEB safety parameters
        
        Args:
            min_free_space_percentage (float): Minimum percentage of ROI that must be free space
            safety_buffer (float): Additional safety margin (0.0 to 1.0)
            min_safe_distance (float): Minimum safe distance in meters (for depth)
            critical_distance (float): Critical distance triggering emergency stop (for depth)
            aeb_fov_fraction (float): Fraction of image width to check for AEB (0.4 = center 40%)
        """
        self.min_free_space_percentage = min_free_space_percentage
        self.safety_buffer = safety_buffer
        self.min_safe_distance = min_safe_distance
        self.critical_distance = critical_distance
        self.aeb_fov_fraction = aeb_fov_fraction
        
        # ROI parameters for general vision processing
        self.roi_top_fraction = 0.7
        self.roi_bottom_fraction = 1.0
        self.roi_left_fraction = 0.0
        self.roi_right_fraction = 1.0
    
    def get_roi_coordinates(self, height, width):
        """Get ROI coordinates"""
        top = int(height * self.roi_top_fraction)
        bottom = int(height * self.roi_bottom_fraction)
        left = int(width * self.roi_left_fraction)
        right = int(width * self.roi_right_fraction)
        return top, bottom, left, right
    
    def get_aeb_safety_zone(self, height, width):
        """
        Get AEB-specific safety zone coordinates (narrower FOV directly in front)
        
        Args:
            height (int): Image height
            width (int): Image width
            
        Returns:
            tuple: (top, bottom, left, right) coordinates for AEB safety zone
        """
        # Vertical: Same as general ROI (road area)
        top = int(height * self.roi_top_fraction)
        bottom = int(height * self.roi_bottom_fraction)
        
        # Horizontal: Center portion only (e.g., center 40% of image)
        center_x = width // 2
        half_aeb_width = int(width * self.aeb_fov_fraction / 2)
        left = center_x - half_aeb_width
        right = center_x + half_aeb_width
        
        # Ensure boundaries are within image
        left = max(0, left)
        right = min(width - 1, right)
        
        return top, bottom, left, right
    
    def check_safety(self, rgb_image, depth_image=None):
        """
        Check if it's safe to continue driving based on RGB and optional depth data
        
        Args:
            rgb_image (np.ndarray): RGB image
            depth_image (np.ndarray, optional): Depth image in meters
            
        Returns:
            tuple: (is_safe, reason, safety_data, annotated_image)
        """
        if rgb_image is None:
            return False, "Missing image data", {}, None
        
        # Create annotated image copy
        annotated_image = rgb_image.copy()
        
        # Convert RGB to grayscale for free space detection
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image
        
        # Get AEB safety zone (narrower FOV directly in front of car)
        height, width = gray.shape
        top, bottom, left, right = self.get_aeb_safety_zone(height, width)
        
        # Extract AEB safety zone
        roi_gray = gray[top:bottom, left:right]
        roi_height, roi_width = roi_gray.shape
        total_roi_pixels = roi_height * roi_width
        
        # Detect free space in ROI (same as vision gap follow)
        _, free_space = cv2.threshold(roi_gray, 80, 255, cv2.THRESH_BINARY)
        
        # Clean up free space detection
        kernel = np.ones((5,5), np.uint8)
        free_space = cv2.morphologyEx(free_space, cv2.MORPH_CLOSE, kernel)
        free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel)
        
        # Calculate free space percentage
        free_space_pixels = np.sum(free_space > 0)
        free_space_percentage = free_space_pixels / total_roi_pixels
        
        # Apply safety buffer
        required_free_space = self.min_free_space_percentage + self.safety_buffer
        
        # Safety check
        is_safe = free_space_percentage >= required_free_space
        
        if is_safe:
            reason = f"Sufficient free space: {free_space_percentage:.1%} >= {required_free_space:.1%}"
        else:
            reason = f"Insufficient free space: {free_space_percentage:.1%} < {required_free_space:.1%}"
        
        # Check for forward path (center area of ROI)
        center_left = roi_width // 4
        center_right = 3 * roi_width // 4
        center_area = free_space[:, center_left:center_right]
        center_free_percentage = np.sum(center_area > 0) / center_area.size
        
        # Additional safety check for forward path
        min_forward_space = 0.2  # Need at least 20% free space in center
        has_forward_path = center_free_percentage >= min_forward_space
        
        if is_safe and not has_forward_path:
            is_safe = False
            reason = f"No forward path: center area only {center_free_percentage:.1%} free"
        
        # Perform depth safety check if depth data available
        depth_check = self._perform_depth_safety_check(depth_image, top, bottom, left, right)
        
        # Make final safety decision combining RGB and depth
        final_is_safe, final_reason = self._combine_safety_decisions(
            is_safe, reason, depth_check, free_space_percentage, center_free_percentage
        )
        
        # Annotate the image with final decision
        self._annotate_safety_image(annotated_image, top, bottom, left, right, 
                                   free_space_percentage, center_free_percentage, 
                                   final_is_safe, final_reason)
        
        safety_data = {
            'rgb_analysis': {
                'free_space_percentage': float(free_space_percentage),
                'required_free_space': float(required_free_space),
                'center_free_percentage': float(center_free_percentage),
                'has_forward_path': bool(has_forward_path),
                'total_roi_pixels': int(total_roi_pixels),
                'free_space_pixels': int(free_space_pixels),
                'is_safe': bool(is_safe),
                'reason': reason
            },
            'depth_check': depth_check,
            'final_decision': {
                'is_safe': bool(final_is_safe),
                'reason': final_reason,
                'method': 'rgb_only' if depth_check['status'] in ['missing', 'invalid'] else 'rgb_plus_depth'
            }
        }
        
        return final_is_safe, final_reason, safety_data, annotated_image
    
    def _annotate_safety_image(self, image, top, bottom, left, right, 
                              free_space_percentage, center_free_percentage, 
                              is_safe, reason):
        """Add safety annotations to the image"""
        # Choose colors based on safety status
        roi_color = (0, 255, 0) if is_safe else (0, 0, 255)  # Green if safe, red if unsafe
        text_color = (0, 255, 0) if is_safe else (0, 0, 255)
        
        # Draw AEB Safety Zone rectangle  
        cv2.rectangle(image, (left, top), (right, bottom), roi_color, 3)
        
        # Add AEB zone label
        cv2.putText(image, 'AEB Zone', (left+10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
        
        # Draw center area rectangle (forward path check)
        center_left = left + (right - left) // 4
        center_right = right - (right - left) // 4
        center_color = (0, 255, 0) if center_free_percentage >= 0.2 else (0, 0, 255)
        cv2.rectangle(image, (center_left, top), (center_right, bottom), center_color, 1)
        
        # Add safety status text
        status_text = "SAFE" if is_safe else "UNSAFE"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        
        # Add free space percentage
        free_space_text = f"Free Space: {free_space_percentage:.1%}"
        cv2.putText(image, free_space_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add center path percentage
        center_text = f"Forward Path: {center_free_percentage:.1%}"
        cv2.putText(image, center_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add reason text (split into multiple lines if too long)
        reason_words = reason.split()
        line_length = 0
        current_line = []
        y_offset = 120
        
        for word in reason_words:
            if line_length + len(word) > 40:  # Start new line
                if current_line:
                    cv2.putText(image, ' '.join(current_line), (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    y_offset += 25
                current_line = [word]
                line_length = len(word)
            else:
                current_line.append(word)
                line_length += len(word) + 1
        
        # Add final line
        if current_line:
            cv2.putText(image, ' '.join(current_line), (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # If unsafe, add warning overlay
        if not is_safe:
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (image.shape[1], 50), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            cv2.putText(image, "EMERGENCY STOP", (image.shape[1]//2 - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def _perform_depth_safety_check(self, depth_image, roi_top, roi_bottom, roi_left, roi_right):
        """Perform depth-based safety check when depth data is available"""
        if depth_image is None:
            return {
                'status': 'missing',
                'reason': 'No depth data available'
            }
        
        # Extract depth ROI
        roi_depth = depth_image[roi_top:roi_bottom, roi_left:roi_right]
        
        # Filter valid depth values
        valid_depths = roi_depth[(roi_depth > 0.1) & (roi_depth < 10.0)]
        
        if len(valid_depths) == 0:
            return {
                'status': 'invalid',
                'reason': 'No valid depth readings in ROI'
            }
        
        # Calculate depth statistics
        min_depth = np.min(valid_depths)
        mean_depth = np.mean(valid_depths)
        
        # Count pixels at different distance thresholds
        critical_pixels = np.sum(valid_depths < self.critical_distance)
        unsafe_pixels = np.sum(valid_depths < self.min_safe_distance)
        total_valid_pixels = len(valid_depths)
        
        critical_percentage = critical_pixels / total_valid_pixels
        unsafe_percentage = unsafe_pixels / total_valid_pixels
        
        # Check forward path (center area)
        center_left = roi_depth.shape[1] // 4
        center_right = 3 * roi_depth.shape[1] // 4
        forward_depth = roi_depth[:, center_left:center_right]
        forward_valid = forward_depth[(forward_depth > 0.1) & (forward_depth < 10.0)]
        
        forward_clear = True
        forward_min_depth = 10.0
        if len(forward_valid) > 0:
            forward_min_depth = np.min(forward_valid)
            forward_clear = forward_min_depth > self.min_safe_distance
        
        # Determine safety status
        if critical_percentage > 0.1:
            status = 'critical'
            reason = f'Critical obstacle: {critical_percentage:.1%} of ROI < {self.critical_distance}m'
        elif unsafe_percentage > 0.3:
            status = 'unsafe'
            reason = f'Unsafe proximity: {unsafe_percentage:.1%} of ROI < {self.min_safe_distance}m'
        elif min_depth < self.critical_distance:
            status = 'critical'
            reason = f'Closest obstacle at {min_depth:.2f}m < {self.critical_distance}m'
        elif min_depth < self.min_safe_distance:
            status = 'warning'
            reason = f'Closest obstacle at {min_depth:.2f}m < {self.min_safe_distance}m'
        else:
            status = 'safe'
            reason = f'All obstacles beyond safe distance (closest: {min_depth:.2f}m)'
        
        return {
            'status': status,
            'reason': reason,
            'min_depth': float(min_depth),
            'mean_depth': float(mean_depth),
            'critical_percentage': float(critical_percentage),
            'unsafe_percentage': float(unsafe_percentage),
            'forward_clear': bool(forward_clear),
            'forward_min_depth': float(forward_min_depth)
        }
    
    def _combine_safety_decisions(self, rgb_safe, rgb_reason, depth_check, 
                                 free_space_percentage, center_free_percentage):
        """Combine RGB and depth safety decisions"""
        if depth_check['status'] in ['missing', 'invalid']:
            # No depth data - use RGB decision
            return rgb_safe, f"RGB-only: {rgb_reason}"
        
        depth_safe = depth_check['status'] in ['safe', 'warning']
        
        if depth_check['status'] == 'critical':
            # Critical depth situation overrides everything
            return False, f"Depth critical: {depth_check['reason']}"
        elif not rgb_safe and not depth_safe:
            # Both indicate unsafe
            return False, f"RGB + Depth unsafe: {rgb_reason} AND {depth_check['reason']}"
        elif not rgb_safe and depth_safe:
            # RGB unsafe but depth safe - use depth to refine
            if depth_check['status'] == 'safe':
                return True, f"Depth override: {depth_check['reason']} (RGB warned: {rgb_reason})"
            else:
                return False, f"RGB unsafe: {rgb_reason}"
        else:
            # RGB safe - depth confirms or overrides
            return depth_safe, f"Combined: {depth_check['reason']}"
    
    def get_emergency_action(self, is_safe, safety_data):
        """
        Determine emergency action based on safety check
        
        Args:
            is_safe (bool): Result from safety check
            safety_data (dict): Detailed safety information
            
        Returns:
            dict: Emergency action with speed and steering commands
        """
        if is_safe:
            return {
                'action': 'continue',
                'speed': None,  # No speed override
                'steering': None,  # No steering override
                'brake': False
            }
        else:
            return {
                'action': 'emergency_stop',
                'speed': 0.0,  # Full stop
                'steering': 0.0,  # Keep current heading
                'brake': True
            }


def test_vision_aeb():
    """Test function for Vision AEB system"""
    print("Vision AEB Safety System Test")
    print("=" * 40)
    
    # Create AEB system
    aeb = VisionAEBSafety(min_free_space_percentage=0.3, safety_buffer=0.2)
    
    # Test with dummy data
    test_rgb = np.ones((480, 960, 3), dtype=np.uint8) * 50  # Dark image (obstacles)
    
    # Add some "free space" (brighter area)
    test_rgb[300:400, 200:800] = 150  # Large bright area = free space
    
    is_safe, reason, data, annotated_image = aeb.check_safety(test_rgb)
    action = aeb.get_emergency_action(is_safe, data)
    
    print(f"Safety check: {'SAFE' if is_safe else 'UNSAFE'}")
    print(f"Reason: {reason}")
    print(f"Action: {action['action']}")
    print(f"Safety data: {data}")
    
    # Save test image
    if annotated_image is not None:
        os.makedirs("images/vision/aeb", exist_ok=True)
        cv2.imwrite("images/vision/aeb/test_aeb.png", annotated_image)
        print("Test annotated image saved to: images/vision/aeb/test_aeb.png")


if __name__ == "__main__":
    test_vision_aeb()