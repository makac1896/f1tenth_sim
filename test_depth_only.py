#!/usr/bin/env python3
"""
Depth-Only Algorithm Test Script

Tests the depth algorithm using ONLY depth images - no synchronization needed.
Generates visualization stages showing how the depth algorithm processes images.

Usage:
    python test_depth_only.py --frames 10
    python test_depth_only.py --depth_dir all_logs/depth/depth_2 --frames 5
"""

import argparse
import cv2
import numpy as np
import os
from pathlib import Path
from math import degrees
from datetime import datetime

# Import our abstracted algorithm
from sim.algorithms.vision_hybrid_follow import VisionHybridFollower


def load_depth_image(depth_file_path):
    """Load depth image from raw file"""
    try:
        # Load raw depth data (assume 640x480, 16-bit depth)
        with open(depth_file_path, 'rb') as f:
            data = f.read()
        
        # Convert to numpy array (640x480 uint16 depth values in mm)
        expected_size = 640 * 480 * 2  # 16-bit = 2 bytes per pixel
        if len(data) == expected_size:
            depth_array = np.frombuffer(data, dtype=np.uint16).reshape((480, 640))
            # Convert from mm to meters
            depth_array = depth_array.astype(np.float32) / 1000.0
            return depth_array
        else:
            print(f"Warning: Unexpected depth file size {len(data)}, expected {expected_size}")
            return None
    except Exception as e:
        print(f"Error loading depth image {depth_file_path}: {e}")
        return None


def find_depth_files(depth_dir):
    """Find all depth files with timestamps"""
    depth_files = {}
    
    for file_path in Path(depth_dir).glob("depth_*.raw"):
        filename = file_path.name
        timestamp_str = filename.replace("depth_", "").replace(".raw", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            depth_files[timestamp] = str(file_path)
        except ValueError:
            continue
    
    return depth_files


def save_depth_visualization_stages(depth_image, result, frame_name, output_dir="images/depth_only"):
    """Save depth processing stages for debugging"""
    os.makedirs(output_dir, exist_ok=True)
    
    if depth_image is None:
        return
    
    debug = result['debug']
    height, width = depth_image.shape
    
    # Get ROI coordinates
    roi_top_fraction = 0.5  # Same as algorithm
    top = int(height * roi_top_fraction)
    bottom = height
    
    # 1a. Original depth image as grayscale
    depth_normalized = np.clip(depth_image, 0, 5.0) / 5.0
    depth_grayscale = (depth_normalized * 255).astype(np.uint8)
    depth_gray_rgb = cv2.cvtColor(depth_grayscale, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(depth_gray_rgb, (0, top), (width, bottom), (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/{frame_name}_1a_depth_grayscale.png", depth_gray_rgb)
    
    # 1b. Original depth image with color mapping
    depth_colored = cv2.applyColorMap(depth_grayscale, cv2.COLORMAP_JET)
    cv2.rectangle(depth_colored, (0, top), (width, bottom), (255, 255, 255), 2)
    cv2.imwrite(f"{output_dir}/{frame_name}_1b_depth_colored.png", depth_colored)
    
    # 2a. Depth ROI as grayscale
    depth_roi = depth_image[top:bottom, :]
    depth_roi_norm = np.clip(depth_roi, 0, 5.0) / 5.0
    depth_roi_gray = (depth_roi_norm * 255).astype(np.uint8)
    depth_roi_gray_rgb = cv2.cvtColor(depth_roi_gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(f"{output_dir}/{frame_name}_2a_depth_roi_gray.png", depth_roi_gray_rgb)
    
    # 2b. Depth ROI as color
    depth_roi_colored = cv2.applyColorMap(depth_roi_gray, cv2.COLORMAP_JET)
    cv2.imwrite(f"{output_dir}/{frame_name}_2b_depth_roi_color.png", depth_roi_colored)
    
    # Column navigability analysis - pure overlay (if debug info available)
    if 'depth_processing_stages' in debug and 'navigable_mask' in debug['depth_processing_stages']:
        navigable_mask = debug['depth_processing_stages']['navigable_mask']
        
        # Create clean black background with only column lines
        column_viz = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw ROI boundary
        cv2.rectangle(column_viz, (0, top), (width, bottom), (100, 100, 100), 1)
        
        # Draw vertical lines for each column within ROI only
        for x in range(width):
            if x < len(navigable_mask):
                if navigable_mask[x]:
                    # Green for navigable columns
                    cv2.line(column_viz, (x, top), (x, bottom), (0, 255, 0), 1)
                else:
                    # Red for blocked columns
                    cv2.line(column_viz, (x, top), (x, bottom), (0, 0, 255), 1)
        
        cv2.imwrite(f"{output_dir}/{frame_name}_column_navigability.png", column_viz)
    
    # 3a. Depth gaps on grayscale
    if 'depth_gaps' in debug:
        nav_viz_gray = depth_gray_rgb.copy()
        
        # Show where gaps were detected on grayscale
        depth_gaps = debug['depth_gaps']
        for i, gap in enumerate(depth_gaps):
            if isinstance(gap, tuple) and len(gap) >= 4:
                start_x, end_x, center_x, width = gap
                # Draw gap boundaries (bright green on grayscale)
                cv2.line(nav_viz_gray, (start_x, top), (start_x, bottom), (0, 255, 0), 3)
                cv2.line(nav_viz_gray, (end_x, top), (end_x, bottom), (0, 255, 0), 3)
                # Draw gap center (bright yellow)
                cv2.line(nav_viz_gray, (center_x, top), (center_x, bottom), (0, 255, 255), 2)
                # Label gap
                cv2.putText(nav_viz_gray, f'G{i+1}', (center_x-10, top+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        gap_count = len(depth_gaps)

        cv2.imwrite(f"{output_dir}/{frame_name}_3a_gaps_grayscale.png", nav_viz_gray)
        
        # 3b. Depth gaps on color map
        nav_viz_color = depth_colored.copy()
        
        for i, gap in enumerate(depth_gaps):
            if isinstance(gap, tuple) and len(gap) >= 4:
                start_x, end_x, center_x, width = gap
                # Draw gap boundaries
                cv2.line(nav_viz_color, (start_x, top), (start_x, bottom), (0, 255, 0), 3)
                cv2.line(nav_viz_color, (end_x, top), (end_x, bottom), (0, 255, 0), 3)
                # Draw gap center
                cv2.line(nav_viz_color, (center_x, top), (center_x, bottom), (255, 255, 255), 2)
                # Label gap
                cv2.putText(nav_viz_color, f'Gap{i+1}', (center_x-20, top+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        

        cv2.imwrite(f"{output_dir}/{frame_name}_3b_gaps_colored.png", nav_viz_color)
    
    # 4a. Final result with steering - Grayscale
    final_image_gray = depth_gray_rgb.copy()
    
    # Add minimal steering info - just the arrow
    if result['steering_angle'] is not None:
        angle_deg = degrees(result['steering_angle'])
        # Draw steering direction arrow
        center_x = width // 2
        arrow_end_x = center_x + int(angle_deg * 5)  # Scale for visibility
        cv2.arrowedLine(final_image_gray, (center_x, top-30), (arrow_end_x, top-30), (0, 255, 0), 3)
    cv2.imwrite(f"{output_dir}/{frame_name}_4a_final_grayscale.png", final_image_gray)
    
    # 4b. Final result with steering - Color
    final_image_color = depth_colored.copy()
    
    # Add minimal steering info - just the arrow
    if result['steering_angle'] is not None:
        angle_deg = degrees(result['steering_angle'])
        # Draw steering direction arrow
        center_x = width // 2
        arrow_end_x = center_x + int(angle_deg * 5)  # Scale for visibility
        cv2.arrowedLine(final_image_color, (center_x, top-30), (arrow_end_x, top-30), (0, 255, 0), 3)
    cv2.imwrite(f"{output_dir}/{frame_name}_4b_final_colored.png", final_image_color)
    
    print(f"  Depth visualization stages saved: {frame_name}_1a-1b, _2a-2c, _3a-3b, _4a-4b")


def test_depth_algorithm(depth_dir, start_frame=0, num_frames=5):
    """Test depth algorithm on pure depth images"""
    
    print(f"\n{'='*60}")
    print(f"Testing DEPTH-ONLY Algorithm")
    print(f"{'='*60}")
    print(f"Depth Directory: {depth_dir}")
    
    # Find depth files
    depth_files = find_depth_files(depth_dir)
    print(f"Found {len(depth_files)} depth files")
    
    if len(depth_files) == 0:
        print("No depth files found!")
        return
    
    # Initialize algorithm in depth-only mode
    algorithm = VisionHybridFollower(driving_mode="depth")
    
    # Sort by timestamp and select range
    sorted_timestamps = sorted(depth_files.keys())
    end_frame = min(start_frame + num_frames, len(sorted_timestamps))
    
    print(f"Processing frames {start_frame} to {end_frame-1} ({end_frame-start_frame} total)")
    
    results = []
    successful_frames = 0
    
    for i in range(start_frame, end_frame):
        timestamp = sorted_timestamps[i]
        depth_file = depth_files[timestamp]
        frame_name = f"frame_{i+1:03d}_depth"
        
        print(f"\n--- Frame {i+1}: {Path(depth_file).name} ---")
        
        # Load depth image
        depth_image = load_depth_image(depth_file)
        if depth_image is None:
            print(f"[ERROR] Failed to load depth image")
            continue
        
        print(f"Depth image loaded: {depth_image.shape}")
        print(f"Depth range: {depth_image.min():.2f}m to {depth_image.max():.2f}m")
        
        # Create dummy RGB image (algorithm expects both but will ignore RGB in depth mode)
        dummy_rgb = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
        
        try:
            # Process with depth algorithm
            result = algorithm.process_image(dummy_rgb, depth_image)
            
            # Save debug visualizations
            save_depth_visualization_stages(depth_image, result, frame_name)
            
            # Print results
            steering_angle = result['steering_angle']
            debug = result['debug']
            
            if steering_angle is not None:
                print(f"[OK] Steering: {degrees(steering_angle):.2f}°")
                successful_frames += 1
            else:
                print(f"[ERROR] No steering solution")
            
            mode_used = debug.get('actual_mode_used', 'UNKNOWN')
            print(f"     Mode used: {mode_used}")
            
            depth_gaps = debug.get('depth_gaps', [])
            print(f"     Depth gaps found: {len(depth_gaps)}")
            
            # Print gap details
            for j, gap in enumerate(depth_gaps):
                if isinstance(gap, tuple) and len(gap) >= 4:
                    start_x, end_x, center_x, width = gap
                    print(f"       Gap {j+1}: center={center_x}px, width={width}px")
            
            results.append({
                'frame': i,
                'steering_angle': steering_angle,
                'mode_used': mode_used,
                'depth_gaps': len(depth_gaps),
                'success': steering_angle is not None
            })
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"DEPTH Algorithm Summary:")
    print(f"Successful frames: {successful_frames}/{num_frames}")
    print(f"Success rate: {(successful_frames/num_frames)*100:.1f}%")
    
    if results:
        successful_results = [r for r in results if r['success']]
        if successful_results:
            angles = [abs(degrees(r['steering_angle'])) for r in successful_results]
            print(f"Average steering angle: {np.mean(angles):.2f}°")
            print(f"Max steering angle: {max(angles):.2f}°")
            
            total_gaps = sum(r['depth_gaps'] for r in successful_results)
            print(f"Total gaps detected: {total_gaps}")
    
    print(f"Visualizations saved to: images/depth_only/")
    return results


def main():
    parser = argparse.ArgumentParser(description='Test Depth-Only Algorithm')
    
    # Data source
    parser.add_argument('--depth_dir', default='all_logs/depth/depth_2',
                       help='Depth data directory')
    
    # Frame selection
    parser.add_argument('--frames', type=int, default=5,
                       help='Number of frames to process')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting frame index')
    
    args = parser.parse_args()
    
    print("Depth-Only Algorithm Test")
    print("="*30)
    print(f"Depth Dir: {args.depth_dir}")
    print(f"Frames: {args.start} to {args.start + args.frames - 1}")
    
    # Validate directory
    if not os.path.exists(args.depth_dir):
        print(f"ERROR: Depth directory does not exist: {args.depth_dir}")
        return
    
    # Create output directory
    os.makedirs("images/depth_only", exist_ok=True)
    
    # Run test
    test_depth_algorithm(args.depth_dir, args.start, args.frames)
    
    print(f"\n[OK] Testing complete!")
    print(f"Check images/depth_only/ for detailed visualizations")


if __name__ == "__main__":
    main()