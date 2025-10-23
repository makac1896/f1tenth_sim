"""
AEB Safety Debug Utilities

Debug and visualization extensions for the core VisionAEBSafety class.
Used for testing, analysis, and visualization during development and simulation.
"""

import numpy as np
import cv2
import os
from sim.algorithms.rgb_aeb_safety import VisionAEBSafety


class VisionAEBSafetyDebug(VisionAEBSafety):
    """Debug extension of VisionAEBSafety with visualization and testing capabilities"""
    
    def __init__(self, *args, **kwargs):
        """Initialize debug AEB system with same parameters as core system"""
        super().__init__(*args, **kwargs)
    
    def check_safety_with_visualization(self, rgb_image, depth_image=None):
        """
        Check safety with full visualization and debug output
        
        Args:
            rgb_image (np.ndarray): RGB image
            depth_image (np.ndarray, optional): Depth image in meters
            
        Returns:
            tuple: (is_safe, reason, safety_data, annotated_image)
        """
        # Get core safety analysis
        is_safe, reason, safety_data = self.check_safety(rgb_image, depth_image)
        
        # Create annotated visualization
        annotated_image = self._create_debug_visualization(
            rgb_image, depth_image, is_safe, reason, safety_data
        )
        
        return is_safe, reason, safety_data, annotated_image
    
    def _create_debug_visualization(self, rgb_image, depth_image, is_safe, reason, safety_data):
        """Create annotated debug visualization"""
        if rgb_image is None:
            return None
        
        # Create annotated image copy
        annotated_image = rgb_image.copy()
        
        # Convert RGB to grayscale for processing
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image
        
        # Get AEB safety zone coordinates
        height, width = gray.shape
        top, bottom, left, right = self.get_aeb_safety_zone(height, width)
        
        # Get analysis data
        free_space_percentage = safety_data.get('free_space_percentage', 0.0)
        center_free_percentage = safety_data.get('center_free_percentage', 0.0)
        has_forward_path = safety_data.get('has_forward_path', False)
        min_depth = safety_data.get('min_depth')
        
        # Add annotations
        self._annotate_safety_zones(annotated_image, top, bottom, left, right, 
                                   center_free_percentage, is_safe)
        self._annotate_safety_status(annotated_image, is_safe, reason, 
                                    free_space_percentage, center_free_percentage, min_depth)
        self._annotate_emergency_overlay(annotated_image, is_safe)
        
        return annotated_image
    
    def _annotate_safety_zones(self, image, top, bottom, left, right, 
                              center_free_percentage, is_safe):
        """Annotate safety zones on the image"""
        # Choose colors based on safety status
        roi_color = (0, 255, 0) if is_safe else (0, 0, 255)  # Green if safe, red if unsafe
        
        # Draw AEB Safety Zone rectangle  
        cv2.rectangle(image, (left, top), (right, bottom), roi_color, 3)
        
        # Add AEB zone label
        cv2.putText(image, 'AEB Zone', (left+10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
        
        # Draw center area rectangle (forward path check)
        center_left = left + (right - left) // 4
        center_right = right - (right - left) // 4
        center_color = (0, 255, 0) if center_free_percentage >= 0.2 else (0, 0, 255)
        cv2.rectangle(image, (center_left, top), (center_right, bottom), center_color, 1)
    
    def _annotate_safety_status(self, image, is_safe, reason, free_space_percentage, 
                               center_free_percentage, min_depth):
        """Add safety status text annotations"""
        text_color = (0, 255, 0) if is_safe else (0, 0, 255)
        
        # Add safety status text
        status_text = "SAFE" if is_safe else "UNSAFE"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        
        # Add free space percentage
        free_space_text = f"Free Space: {free_space_percentage:.1%}"
        cv2.putText(image, free_space_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add center path percentage
        center_text = f"Forward Path: {center_free_percentage:.1%}"
        cv2.putText(image, center_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add depth information if available
        if min_depth is not None:
            depth_text = f"Min Depth: {min_depth:.2f}m"
            cv2.putText(image, depth_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset = 150
        else:
            y_offset = 120
        
        # Add reason text (split into multiple lines if too long)
        self._add_multiline_text(image, reason, (10, y_offset), text_color)
    
    def _add_multiline_text(self, image, text, start_pos, color):
        """Add multi-line text to image"""
        words = text.split()
        line_length = 0
        current_line = []
        x, y_offset = start_pos
        
        for word in words:
            if line_length + len(word) > 40:  # Start new line
                if current_line:
                    cv2.putText(image, ' '.join(current_line), (x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 25
                current_line = [word]
                line_length = len(word)
            else:
                current_line.append(word)
                line_length += len(word) + 1
        
        # Add final line
        if current_line:
            cv2.putText(image, ' '.join(current_line), (x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _annotate_emergency_overlay(self, image, is_safe):
        """Add emergency warning overlay if unsafe"""
        if not is_safe:
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (image.shape[1], 50), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            cv2.putText(image, "EMERGENCY STOP", (image.shape[1]//2 - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def save_debug_image(self, annotated_image, filename, output_dir="images/vision/aeb"):
        """Save debug annotated image"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, annotated_image)
        return filepath


def test_vision_aeb_debug():
    """Test function for Vision AEB debug system"""
    print("Vision AEB Safety Debug System Test")
    print("=" * 50)
    
    # Create debug AEB system
    aeb_debug = VisionAEBSafetyDebug(min_free_space_percentage=0.3, safety_buffer=0.2)
    
    # Test with dummy data
    test_rgb = np.ones((480, 960, 3), dtype=np.uint8) * 50  # Dark image (obstacles)
    
    # Add some "free space" (brighter area)
    test_rgb[300:400, 200:800] = 150  # Large bright area = free space
    
    # Test RGB-only safety check
    is_safe, reason, data, annotated_image = aeb_debug.check_safety_with_visualization(test_rgb)
    
    print(f"Safety check: {'SAFE' if is_safe else 'UNSAFE'}")
    print(f"Reason: {reason}")
    print(f"Action: {'continue' if is_safe else 'emergency_stop'}")
    print(f"Safety data: {data}")
    
    # Save test image
    if annotated_image is not None:
        filepath = aeb_debug.save_debug_image(annotated_image, "test_aeb_debug.png")
        print(f"Debug visualization saved to: {filepath}")
    
    # Test with depth data
    print("\n" + "=" * 30)
    print("Testing with depth data...")
    
    test_depth = np.ones((480, 960), dtype=np.float32) * 2.5  # 2.5m distance
    # Add close obstacle in center
    test_depth[350:370, 450:550] = 0.6  # Close obstacle at 0.6m
    
    is_safe_depth, reason_depth, data_depth, annotated_depth = aeb_debug.check_safety_with_visualization(test_rgb, test_depth)
    
    print(f"Safety check with depth: {'SAFE' if is_safe_depth else 'UNSAFE'}")
    print(f"Reason: {reason_depth}")
    print(f"Method: {data_depth.get('method', 'unknown')}")
    print(f"Min depth: {data_depth.get('min_depth', 'N/A')}")
    
    if annotated_depth is not None:
        filepath_depth = aeb_debug.save_debug_image(annotated_depth, "test_aeb_debug_with_depth.png")
        print(f"Depth debug visualization saved to: {filepath_depth}")


if __name__ == "__main__":
    test_vision_aeb_debug()