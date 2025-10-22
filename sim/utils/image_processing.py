"""
Image Processing Utilities for F1Tenth Gap Visualization

Handles loading raw camera images, converting to PNG, and overlaying
lidar gap detection results as shaded areas on the camera view.

Key Features:
- Raw image loading and PNG conversion using existing converter
- Lidar-to-camera coordinate mapping
- Gap visualization overlays
- Batch processing for analysis runs
"""
import os
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from math import degrees, radians, tan, atan2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from raw_to_png_converter import raw_to_png


class ImageProcessor:
    """
    Process camera images and overlay lidar gap detection results
    """
    
    def __init__(self, output_dir="images/lidar"):
        """
        Initialize image processor
        
        Args:
            output_dir (str): Directory to save processed images
        """
        self.output_dir = output_dir
        
        # Camera parameters (adjust based on actual image size from converter)
        self.image_width = 960  # Updated to match actual converted image size
        self.image_height = 480
        self.camera_fov_horizontal = 69.4  # degrees (Intel D435i typical)
        self.camera_fov_vertical = 42.5    # degrees
        
        # Lidar parameters
        self.lidar_fov = 270  # degrees
        self.lidar_range_count = 1081  # points
        
        # Visual styling
        self.gap_color = (0, 255, 0, 100)      # Green with transparency
        self.selected_gap_color = (255, 0, 0, 150)  # Red with transparency
        self.steering_color = (255, 255, 0)     # Yellow
        self.text_color = (0, 255, 255)        # Cyan - more visible on grayscale
        
    def clear_output_directory(self):
        """Clear the output directory to save space"""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Cleared output directory: {self.output_dir}")
    
    def load_raw_image(self, image_path):
        """
        Load raw image file using the existing tested converter
        
        Args:
            image_path (str): Path to .raw image file
            
        Returns:
            np.ndarray: Image as numpy array (H, W, 3) or None if failed
        """
        try:
            # Create temporary directory for conversion
            temp_dir = os.path.join(self.output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Convert raw to PNG using the existing tested converter
            png_path = raw_to_png(image_path, temp_dir)
            
            if png_path and os.path.exists(png_path):
                # Load the converted PNG image as-is, no additional processing
                image = cv2.imread(png_path)
                if image is not None:
                    # Clean up temporary file
                    os.remove(png_path)
                    
                    return image
                else:
                    print(f"Failed to load converted PNG: {png_path}")
                    return None
            else:
                print(f"Failed to convert raw image: {image_path}")
                return None
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def lidar_angle_to_image_x(self, lidar_angle_deg):
        """
        Convert lidar angle to image x coordinate
        
        Args:
            lidar_angle_deg (float): Lidar angle in degrees (-135 to +135)
            
        Returns:
            int: X coordinate in image (0 to image_width)
        """
        # Map lidar FOV to camera FOV
        # Lidar: -135° to +135° (270° total)
        # Camera: -34.7° to +34.7° (69.4° total) - assuming center-aligned
        
        camera_half_fov = self.camera_fov_horizontal / 2
        
        # Clamp lidar angle to camera FOV
        if lidar_angle_deg < -camera_half_fov:
            lidar_angle_deg = -camera_half_fov
        elif lidar_angle_deg > camera_half_fov:
            lidar_angle_deg = camera_half_fov
        
        # Convert to image coordinates (0 = left, image_width = right)
        # Camera center (0°) maps to image center (image_width/2)
        x = int((lidar_angle_deg + camera_half_fov) * self.image_width / self.camera_fov_horizontal)
        
        return max(0, min(x, self.image_width - 1))
    
    def lidar_gap_to_image_region(self, gap_indices, lidar_ranges):
        """
        Convert lidar gap indices to image region coordinates
        
        Args:
            gap_indices (list): List of lidar point indices in the gap
            lidar_ranges (list): Full lidar range array
            
        Returns:
            tuple: (x1, x2, y1, y2) image region coordinates
        """
        if not gap_indices:
            return None
        
        # Convert indices to angles using original algorithm conversion
        start_angle_deg = gap_indices[0] * 270 / 1080 - 135
        end_angle_deg = gap_indices[-1] * 270 / 1080 - 135
        
        # Convert to image x coordinates
        x1 = self.lidar_angle_to_image_x(start_angle_deg)
        x2 = self.lidar_angle_to_image_x(end_angle_deg)
        
        # Ensure proper ordering
        if x1 > x2:
            x1, x2 = x2, x1
        
        # Y coordinates: use full height for now (could be refined with distance info)
        y1 = 0
        y2 = self.image_height
        
        return (x1, x2, y1, y2)
    
    def calculate_steering_arrow_position(self, steering_angle_deg):
        """
        Calculate position for steering direction arrow
        
        Args:
            steering_angle_deg (float): Steering angle in degrees
            
        Returns:
            tuple: (x, y) position for arrow
        """
        x = self.lidar_angle_to_image_x(steering_angle_deg)
        y = int(self.image_height * 0.8)  # 80% down the image
        return (x, y)
    
    def process_frame(self, image_path, gaps, selected_gap_idx, steering_angle_deg, 
                     frame_info, save_path):
        """
        Process a single frame: load image, overlay gaps, save result
        
        Args:
            image_path (str): Path to raw image file
            gaps (list): List of gap indices from lidar analysis
            selected_gap_idx (int): Index of the selected gap
            steering_angle_deg (float): Calculated steering angle
            frame_info (dict): Frame metadata for overlay text
            save_path (str): Path to save processed image
            
        Returns:
            bool: True if successful
        """
        # Load raw image
        image_array = self.load_raw_image(image_path)
        if image_array is None:
            return False
        
        # Convert to PIL Image for easier drawing
        image = Image.fromarray(image_array)
        
        # Create overlay for transparency effects
        overlay = Image.new('RGBA', (self.image_width, self.image_height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_image = ImageDraw.Draw(image)
        
        # No gap coloring - just display the algorithm's calculated steering direction
        
        # No arrow - just metadata text overlay
        
        # Add text overlay with frame information
        try:
            # Try to use a better font if available
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        text_lines = [
            f"Frame: {frame_info.get('frame_index', 'N/A')}",
            f"Gaps: {len(gaps)}",
            f"Steering: {steering_angle_deg:.2f}°" if steering_angle_deg else "Steering: N/A",
            f"Time: {frame_info.get('timestamp', 'N/A')}"
        ]
        
        y_offset = 10
        for line in text_lines:
            draw_image.text((10, y_offset), line, fill=self.text_color, font=font)
            y_offset += 20
        
        # No overlay compositing needed since we're not adding colored gaps
        
        # Save the image with just the metadata text
        image.save(save_path)
        
        return True
    
    def process_analysis_results(self, results, data_loader):
        """
        Process all frames from analysis results and create visualizations
        
        Args:
            results (GapAnalysisResults): Analysis results container
            data_loader (F1TenthDataLoader): Data loader for accessing images
            
        Returns:
            int: Number of images successfully processed
        """
        self.clear_output_directory()
        
        processed_count = 0
        total_frames = len(results.frame_results)
        
        print(f"Processing {total_frames} frames for visualization...")
        
        for i, result in enumerate(results.frame_results):
            if i % 10 == 0 or i == total_frames - 1:
                print(f"  Processing frame {i+1}/{total_frames}")
            
            # Get synchronized data
            pair = data_loader.get_synchronized_pair(result['frame_index'])
            if not pair or not pair['vision_data']:
                continue
            
            vision_data = pair['vision_data']
            if not vision_data['image'] is not None:
                continue
            
            # Prepare frame info
            frame_info = {
                'frame_index': result['frame_index'],
                'timestamp': result['timestamp'].strftime("%H:%M:%S.%f")[:-3],
                'lidar_ranges': pair['lidar_data']['ranges']
            }
            
            # Get gap data from stored results
            gaps = result.get('_gaps', [])
            selected_gap_idx = result.get('_selected_gap_idx', -1)
            
            # Generate output filename
            output_filename = f"frame_{result['frame_index']:04d}_{result['timestamp'].strftime('%H%M%S%f')[:-3]}.png"
            save_path = os.path.join(self.output_dir, output_filename)
            
            # Create image from raw array
            if vision_data['image'] is not None:
                temp_path = "temp_raw_image.raw"
                with open(temp_path, 'wb') as f:
                    f.write(vision_data['image'].tobytes())
                
                success = self.process_frame(
                    temp_path,
                    gaps,
                    selected_gap_idx,
                    result['our_steering_angle_deg'],
                    frame_info,
                    save_path
                )
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if success:
                    processed_count += 1
        
        print(f"Successfully processed {processed_count}/{total_frames} frames")
        print(f"Images saved to: {self.output_dir}")
        
        return processed_count