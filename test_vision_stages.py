#!/usr/bin/env python3
"""
Vision Processing Stages Test

Shows the intermediate processing stages of the vision gap following algorithm:
1. Original image
2. Grayscale
3. Edge detection
4. Thick edges (dilated)
5. Green/Black free space visualization
6. Final gap detection with steering

Usage:
    python test_vision_stages.py --vision_dir all_logs/vision/vision_1 --frames 5
    python test_vision_stages.py --vision_dir all_logs/vision/vision_2 --start 10 --frames 3

This helps debug and understand how the algorithm processes images.
"""

import argparse
import cv2
import numpy as np
from sim.algorithms.vision_gap_follow import VisionGapFollower
from raw_to_png_converter import raw_to_png
from math import degrees
import os
from pathlib import Path
from datetime import datetime

def save_processing_stages(image, result, frame_name, output_dir="images/vision/stages"):
    """Save all processing stages as separate images with ROI clearly marked"""
    os.makedirs(output_dir, exist_ok=True)
    
    stages = result['debug']['processing_stages']
    height, width = image.shape[:2]
    
    # Get ROI coordinates (same as algorithm)
    roi_top_fraction = 0.7
    top = int(height * roi_top_fraction)
    bottom = height
    left = 0
    right = width
    
    # 1. Original image with ROI
    original_with_roi = image.copy()
    cv2.rectangle(original_with_roi, (left, top), (right, bottom), (0, 0, 255), 3)  # Red ROI
    cv2.putText(original_with_roi, 'ROI', (left+10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(f"{output_dir}/{frame_name}_1_original.png", original_with_roi)
    
    # 2. Grayscale with ROI
    grayscale_with_roi = cv2.cvtColor(stages['grayscale'], cv2.COLOR_GRAY2BGR)
    cv2.rectangle(grayscale_with_roi, (left, top), (right, bottom), (0, 0, 255), 3)  # Red ROI
    cv2.putText(grayscale_with_roi, 'ROI', (left+10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(f"{output_dir}/{frame_name}_2_grayscale.png", grayscale_with_roi)
    
    # 3. Raw edges with ROI
    edges_with_roi = cv2.cvtColor(stages['edges'], cv2.COLOR_GRAY2BGR)
    cv2.rectangle(edges_with_roi, (left, top), (right, bottom), (0, 0, 255), 3)  # Red ROI
    cv2.putText(edges_with_roi, 'ROI', (left+10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(f"{output_dir}/{frame_name}_3_edges.png", edges_with_roi)
    
    # 4. Thick edges with ROI
    thick_edges_with_roi = cv2.cvtColor(stages['thick_edges'], cv2.COLOR_GRAY2BGR)
    cv2.rectangle(thick_edges_with_roi, (left, top), (right, bottom), (0, 0, 255), 3)  # Red ROI
    cv2.putText(thick_edges_with_roi, 'ROI', (left+10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(f"{output_dir}/{frame_name}_4_thick_edges.png", thick_edges_with_roi)
    
    # 5. Green/Black free space with ROI
    green_black_with_roi = stages['green_black'].copy()
    cv2.rectangle(green_black_with_roi, (left, top), (right, bottom), (255, 0, 0), 3)  # Blue ROI (visible on green/black)
    cv2.putText(green_black_with_roi, 'ROI', (left+10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imwrite(f"{output_dir}/{frame_name}_5_green_black.png", green_black_with_roi)
    
    # 6. Column navigability analysis
    gap_debug = result['debug']['gap_debug']
    column_debug_image = gap_debug['column_debug_image']
    cv2.imwrite(f"{output_dir}/{frame_name}_6_column_analysis.png", column_debug_image)
    
    # 7. Final result with gap detection and ROI
    final_image = create_final_visualization(image, result)
    cv2.imwrite(f"{output_dir}/{frame_name}_7_final_gaps.png", final_image)
    
    print(f"Processing stages saved for {frame_name} (ROI: {top}-{bottom} pixels)")


def create_final_visualization(original_image, result):
    """Create final visualization with gaps and steering info"""
    vis_image = original_image.copy()
    height, width = vis_image.shape[:2]
    
    debug_info = result['debug']
    gaps = debug_info['gaps']
    steering_angle = result['steering_angle']
    
    # Get ROI coordinates
    roi_top_fraction = 0.7
    top = int(height * roi_top_fraction)
    bottom = height
    left = 0
    right = width
    
    # Draw ROI rectangle with label
    cv2.rectangle(vis_image, (left, top), (right, bottom), (255, 0, 0), 3)  # Blue ROI, thicker
    cv2.putText(vis_image, 'ROI', (left+10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Draw gaps
    for i, gap in enumerate(gaps):
        start_x = gap['start']
        end_x = gap['end']
        center_x = gap['center']
        gap_width = gap['width']
        
        # Gap boundaries
        cv2.line(vis_image, (start_x, top), (start_x, bottom), (0, 255, 255), 2)
        cv2.line(vis_image, (end_x, top), (end_x, bottom), (0, 255, 255), 2)
        # Gap center
        cv2.line(vis_image, (center_x, top), (center_x, bottom), (0, 255, 0), 2)
    
    # Add text info
    y_offset = 30
    line_height = 25
    
    if steering_angle is not None:
        angle_deg = degrees(steering_angle)
        cv2.putText(vis_image, f'Steering: {angle_deg:.1f}°', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height
    
    cv2.putText(vis_image, f'Gaps: {len(gaps)}', 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis_image


def find_vision_files(vision_dir):
    """Find all vision files with timestamps"""
    vision_files = {}
    
    for file_path in Path(vision_dir).glob("image_*.raw"):
        filename = file_path.name
        timestamp_str = filename.replace("image_", "").replace(".raw", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            vision_files[timestamp] = str(file_path)
        except ValueError:
            continue
    
    return vision_files


def test_processing_stages(vision_dir, start_frame=0, num_frames=3):
    """Test the vision algorithm and show all processing stages"""
    print("Vision Processing Stages Test")
    print("=" * 50)
    print(f"Vision Directory: {vision_dir}")
    
    # Validate vision directory
    if not os.path.exists(vision_dir):
        print(f"Error: Vision directory {vision_dir} does not exist")
        return
    
    # Load vision files
    print("Loading vision files...")
    vision_files = find_vision_files(vision_dir)
    print(f"Found {len(vision_files)} vision files")
    
    if len(vision_files) == 0:
        print("No vision files found!")
        return
    
    # Sort by timestamp and select range
    sorted_timestamps = sorted(vision_files.keys())
    end_frame = min(start_frame + num_frames, len(sorted_timestamps))
    
    print(f"Processing frames {start_frame} to {end_frame-1} ({end_frame-start_frame} total)")
    
    # Initialize algorithm
    vision_follower = VisionGapFollower()
    
    for i in range(start_frame, end_frame):
        timestamp = sorted_timestamps[i]
        image_file = vision_files[timestamp]
        frame_name = f"frame_{i+1:03d}"
        
        print(f"\n--- {frame_name}: {image_file.split('/')[-1]} ---")
        
        # Convert and load image
        png_path = raw_to_png(image_file, "temp_stages_test")
        if not png_path:
            print(f"[ERROR] Failed to convert {image_file}")
            continue
            
        image = cv2.imread(png_path)
        print(f"Image loaded: {image.shape}")
        
        # Process with algorithm
        result = vision_follower.process_image(image)
        
        # Save all processing stages
        save_processing_stages(image, result, frame_name)
        
        # Show results
        steering_angle = result['steering_angle']
        debug = result['debug']
        
        if steering_angle is not None:
            print(f"[OK] Steering: {degrees(steering_angle):.1f}°")
            print(f"     Gaps found: {debug['gaps_found']}")
            
            for j, gap in enumerate(debug['gaps']):
                center_x = gap['center']
                width = gap['width']
                depth_info = f", depth={gap['median_depth']:.1f}mm" if gap['median_depth'] else ""
                print(f"     Gap {j+1}: center={center_x}px, width={width}px{depth_info}")
        else:
            print(f"[ERROR] No steering solution")
            print(f"         Gaps detected: {debug['gaps_found']}")
    
    print(f"\n[OK] All processing stages saved to images/vision/stages/")
    print("Check the numbered files to see each processing step!")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Vision Processing Stages Test')
    
    # Directory arguments
    parser.add_argument('--vision_dir', required=True,
                       help='Vision data directory (e.g., all_logs/vision/vision_1)')
    
    # Frame selection arguments
    parser.add_argument('--frames', type=int, default=3,
                       help='Number of frames to process (default: 3)')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting frame index (default: 0)')
    
    # Algorithm parameters
    parser.add_argument('--car_width', type=float, default=0.3,
                       help='Car width in meters (default: 0.3)')
    parser.add_argument('--min_gap_pixels', type=int, default=50,
                       help='Minimum gap width in pixels (default: 50)')
    parser.add_argument('--threshold', type=int, default=80,
                       help='Free space threshold (default: 80)')
    
    args = parser.parse_args()
    
    # Run test with parameters
    test_processing_stages(args.vision_dir, args.start, args.frames)


if __name__ == "__main__":
    main()