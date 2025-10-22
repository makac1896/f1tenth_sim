#!/usr/bin/env python3
"""
Raw to PNG Converter

Converts existing .raw depth files to visualizable PNG images.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import argparse

def auto_detect_resolution(file_size, dtype=np.uint16):
    """
    Auto-detect image resolution based on file size
    """
    bytes_per_pixel = np.dtype(dtype).itemsize
    total_pixels = file_size // bytes_per_pixel
    
    # Common resolutions to check
    common_resolutions = [
        (640, 480),   # 307200 pixels, 614400 bytes
        (960, 480),   # 460800 pixels, 921600 bytes
        (848, 480),   # 407040 pixels, 814080 bytes
        (800, 600),   # 480000 pixels, 960000 bytes
        (1024, 768),  # 786432 pixels, 1572864 bytes
    ]
    
    for width, height in common_resolutions:
        if width * height == total_pixels:
            return width, height
    
    # If no common resolution found, try to find square or close-to-square
    import math
    sqrt_pixels = int(math.sqrt(total_pixels))
    if sqrt_pixels * sqrt_pixels == total_pixels:
        return sqrt_pixels, sqrt_pixels
    
    # Default fallback
    return 640, 480

def raw_to_png(raw_file_path, output_dir=None, image_width=None, image_height=None, dtype=np.uint16):
    """
    Convert a .raw depth file to a PNG image with auto-resolution detection
    """
    raw_path = Path(raw_file_path)
    
    if output_dir is None:
        output_dir = raw_path.parent / "converted"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Read raw depth data
        with open(raw_path, 'rb') as f:
            data = f.read()
        
        if len(data) == 0:
            print(f"Skipping {raw_path.name}: Empty file")
            return None
        
        # Auto-detect resolution if not provided
        if image_width is None or image_height is None:
            detected_width, detected_height = auto_detect_resolution(len(data), dtype)
            actual_width = image_width if image_width else detected_width
            actual_height = image_height if image_height else detected_height
        else:
            actual_width, actual_height = image_width, image_height
        
        expected_size = actual_width * actual_height * np.dtype(dtype).itemsize
        
        if len(data) != expected_size:
            print(f"Warning: {raw_path.name} size mismatch. Expected {expected_size} for {actual_width}x{actual_height}, got {len(data)}")
            # Try auto-detection again
            actual_width, actual_height = auto_detect_resolution(len(data), dtype)
            expected_size = actual_width * actual_height * np.dtype(dtype).itemsize
            
            if len(data) != expected_size:
                print(f"Error: Could not determine correct resolution for {raw_path.name}")
                return None
        
        # Convert to numpy array
        depth_array = np.frombuffer(data, dtype=dtype)
        depth_image = depth_array.reshape((actual_height, actual_width))
        
        print(f"Processing {raw_path.name}: {actual_width}x{actual_height}")
        
        # Normalize for visualization (0-255)
        if depth_image.max() > 0:
            normalized = ((depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(depth_image, dtype=np.uint8)
        
        # Create output filename
        png_filename = raw_path.stem + '.png'
        png_path = output_dir / png_filename
        
        # Save PNG image
        cv2.imwrite(str(png_path), normalized)
        
        print(f"Converted: {raw_path.name} -> {png_path.name}")
        return str(png_path)
        
    except Exception as e:
        print(f"Error converting {raw_path.name}: {e}")
        return None

def process_raw_directory(input_dir, output_dir=None, width=640, height=480):
    """
    Process all .raw files in a directory
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    # Set default output directory
    if output_dir is None:
        output_dir = input_path / "converted"
    
    # Find all .raw files
    raw_files = list(input_path.glob("*.raw"))
    
    if not raw_files:
        print(f"No .raw files found in {input_dir}")
        return
    
    print(f"Found {len(raw_files)} .raw files to convert")
    
    converted_count = 0
    
    for raw_file in raw_files:
        # Use None for auto-detection, or pass specific dimensions if provided
        w = width if width != 640 else None  # Use auto-detection unless specifically set
        h = height if height != 480 else None
        png_file = raw_to_png(raw_file, output_dir, w, h)
        if png_file:
            converted_count += 1
    
    print(f"\nConversion complete:")
    print(f"- Converted {converted_count} files to PNG format")

def main():
    parser = argparse.ArgumentParser(description="Convert .raw depth files to PNG images")
    parser.add_argument("input_dir", help="Directory containing .raw files")
    parser.add_argument("-o", "--output", help="Output directory (default: input_dir/converted)")
    parser.add_argument("-w", "--width", type=int, default=640, help="Image width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Image height (default: 480)")
    
    args = parser.parse_args()
    
    process_raw_directory(
        input_dir=args.input_dir,
        output_dir=args.output,
        width=args.width,
        height=args.height
    )

if __name__ == "__main__":
    main()