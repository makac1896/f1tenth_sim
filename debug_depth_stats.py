#!/usr/bin/env python3
"""
Debug Depth Values - Analyze actual depth statistics to calibrate thresholds
"""

import numpy as np
import os
from pathlib import Path
from datetime import datetime

def load_depth_image(depth_file_path):
    """Load depth image from raw file"""
    try:
        with open(depth_file_path, 'rb') as f:
            data = f.read()
        
        expected_size = 640 * 480 * 2
        if len(data) == expected_size:
            depth_array = np.frombuffer(data, dtype=np.uint16).reshape((480, 640))
            depth_array = depth_array.astype(np.float32) / 1000.0
            return depth_array
        else:
            return None
    except Exception as e:
        print(f"Error loading depth image: {e}")
        return None

def analyze_depth_statistics(depth_dir="all_logs/depth/depth_2"):
    """Analyze depth statistics to understand value ranges"""
    
    # Find depth files
    depth_files = {}
    for file_path in Path(depth_dir).glob("depth_*.raw"):
        filename = file_path.name
        timestamp_str = filename.replace("depth_", "").replace(".raw", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            depth_files[timestamp] = str(file_path)
        except ValueError:
            continue
    
    print(f"Found {len(depth_files)} depth files")
    
    if len(depth_files) == 0:
        return
    
    # Analyze first few files
    sorted_timestamps = sorted(depth_files.keys())
    
    for i in range(min(3, len(sorted_timestamps))):
        timestamp = sorted_timestamps[i]
        depth_file = depth_files[timestamp]
        
        print(f"\n--- File {i+1}: {Path(depth_file).name} ---")
        
        depth_image = load_depth_image(depth_file)
        if depth_image is None:
            continue
        
        # Extract ROI (bottom half)
        height, width = depth_image.shape
        roi_top = int(height * 0.5)
        roi_depth = depth_image[roi_top:, :]
        
        # Overall statistics
        valid_mask = (depth_image > 0.1) & (depth_image < 5.0)
        valid_depths = depth_image[valid_mask]
        
        print(f"Image size: {height}x{width}")
        print(f"Valid pixels: {np.sum(valid_mask)} / {height*width}")
        print(f"Depth range: {np.min(valid_depths):.2f} - {np.max(valid_depths):.2f} meters")
        print(f"Median depth: {np.median(valid_depths):.2f} meters")
        print(f"Mean depth: {np.mean(valid_depths):.2f} meters")
        
        # ROI statistics
        roi_valid_mask = (roi_depth > 0.1) & (roi_depth < 5.0)
        roi_valid_depths = roi_depth[roi_valid_mask]
        
        print(f"\nROI statistics:")
        print(f"ROI size: {roi_depth.shape[0]}x{roi_depth.shape[1]}")
        print(f"ROI valid pixels: {np.sum(roi_valid_mask)}")
        print(f"ROI depth range: {np.min(roi_valid_depths):.2f} - {np.max(roi_valid_depths):.2f} meters")
        print(f"ROI median: {np.median(roi_valid_depths):.2f} meters")
        
        # Column analysis
        print(f"\nColumn analysis:")
        close_columns = 0  # < 0.8m
        medium_columns = 0  # 0.8-2.0m  
        far_columns = 0    # > 2.0m
        
        for x in range(width):
            col = roi_depth[:, x]
            valid = col[(col > 0.1) & (col < 5.0)]
            if len(valid) > 0:
                median = np.median(valid)
                if median < 0.8:
                    close_columns += 1
                elif median < 2.0:
                    medium_columns += 1
                else:
                    far_columns += 1
        
        print(f"Close columns (<0.8m): {close_columns}")
        print(f"Medium columns (0.8-2.0m): {medium_columns}")
        print(f"Far columns (>2.0m): {far_columns}")
        
        # Histogram of depths
        bins = [0.1, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
        hist, _ = np.histogram(roi_valid_depths, bins)
        print(f"\nDepth distribution in ROI:")
        for i in range(len(bins)-1):
            print(f"  {bins[i]:.1f} - {bins[i+1]:.1f}m: {hist[i]} pixels")

if __name__ == "__main__":
    analyze_depth_statistics()