#!/usr/bin/env python3
"""
Vision Gap Following Analysis

Comprehensive analysis script for vision-based gap following algorithm.
Similar to gap_analysis.py but focused on vision processing with RGB camera data.

Usage:
    python vision_gap_analysis.py --vision_dir all_logs/vision/vision_1 --frames 50
    python vision_gap_analysis.py --vision_dir all_logs/vision/vision_2 --start_time "2025-10-21T16:52:47" --end_time "2025-10-21T16:52:50"
"""

import argparse
import json
import os
import pandas as pd
import cv2
from datetime import datetime
from pathlib import Path
from math import degrees

from sim.algorithms.vision_gap_follow import VisionGapFollower
from sim.utils.image_processing import VisionGapVisualizer
from raw_to_png_converter import raw_to_png


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


def filter_by_timeframe(vision_files, start_time=None, end_time=None):
    """Filter vision files by time range"""
    if not start_time and not end_time:
        return vision_files
    
    filtered_files = {}
    
    # Parse time strings if provided
    start_dt = None
    end_dt = None
    
    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        except ValueError:
            print(f"Warning: Invalid start_time format: {start_time}")
    
    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        except ValueError:
            print(f"Warning: Invalid end_time format: {end_time}")
    
    # Filter files
    for timestamp, file_path in vision_files.items():
        include = True
        
        if start_dt and timestamp < start_dt:
            include = False
        if end_dt and timestamp > end_dt:
            include = False
            
        if include:
            filtered_files[timestamp] = file_path
    
    return filtered_files


def analyze_vision_frame(vision_file, vision_follower, visualizer, frame_idx, save_viz=False):
    """Analyze a single vision frame"""
    try:
        # Convert raw image to PNG
        png_path = raw_to_png(vision_file, "temp_vision_analysis")
        if not png_path:
            return None
        
        # Load image
        image = cv2.imread(png_path)
        if image is None:
            return None
        
        # Process with gap following algorithm
        result = vision_follower.process_image(image)
        
        # Create visualization if requested
        if save_viz and visualizer:
            visualizer.visualize_gaps(image, result, f"vision_frame_{frame_idx:04d}.png")
        
        # Extract results
        steering_angle = result['steering_angle']
        debug_info = result['debug']
        
        return {
            'frame_idx': frame_idx,
            'steering_angle_rad': steering_angle,
            'steering_angle_deg': degrees(steering_angle) if steering_angle is not None else None,
            'gaps_found': debug_info['gaps_found'],
            'gaps': debug_info['gaps'],
            'has_solution': steering_angle is not None
        }
        
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")
        return None


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Vision Gap Following Analysis')
    
    # Directory arguments
    parser.add_argument('--vision_dir', required=True, 
                       help='Vision data directory (e.g., all_logs/vision/vision_1)')
    
    # Frame selection arguments
    parser.add_argument('--frames', type=int, default=50,
                       help='Number of frames to process (default: 50)')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting frame index (default: 0)')
    parser.add_argument('--end', type=int,
                       help='Ending frame index (overrides --frames)')
    
    # Time filtering arguments
    parser.add_argument('--start_time', type=str,
                       help='Start time filter (ISO format: 2025-10-21T16:52:47Z)')
    parser.add_argument('--end_time', type=str,
                       help='End time filter (ISO format: 2025-10-21T16:52:50Z)')
    
    # Algorithm parameters
    parser.add_argument('--car_width', type=float, default=0.3,
                       help='Car width in meters (default: 0.3)')
    parser.add_argument('--min_gap_pixels', type=int, default=50,
                       help='Minimum gap width in pixels (default: 50)')
    parser.add_argument('--threshold', type=int, default=80,
                       help='Free space threshold (default: 80)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='vision_analysis_results.json',
                       help='Output file for results (default: vision_analysis_results.json)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization images')
    
    args = parser.parse_args()
    
    print("Vision Gap Following Analysis")
    print("=" * 50)
    print(f"Vision Directory: {args.vision_dir}")
    
    # Validate vision directory
    if not os.path.exists(args.vision_dir):
        print(f"Error: Vision directory {args.vision_dir} does not exist")
        return
    
    # Load vision files
    print("Loading vision files...")
    vision_files = find_vision_files(args.vision_dir)
    print(f"Found {len(vision_files)} vision files")
    
    if len(vision_files) == 0:
        print("No vision files found!")
        return
    
    # Apply time filtering if specified
    if args.start_time or args.end_time:
        print(f"Filtering by time range: {args.start_time} to {args.end_time}")
        vision_files = filter_by_timeframe(vision_files, args.start_time, args.end_time)
        print(f"After time filtering: {len(vision_files)} files")
    
    # Sort by timestamp
    sorted_timestamps = sorted(vision_files.keys())
    
    # Determine frame range
    start_frame = args.start
    if args.end is not None:
        end_frame = min(args.end, len(sorted_timestamps))
    else:
        end_frame = min(start_frame + args.frames, len(sorted_timestamps))
    
    print(f"Processing frames {start_frame} to {end_frame-1} ({end_frame-start_frame} total)")
    
    # Initialize vision algorithm
    vision_follower = VisionGapFollower(
        car_width=args.car_width,
        min_gap_width_pixels=args.min_gap_pixels,
        free_space_threshold=args.threshold
    )
    
    # Initialize visualizer if needed
    visualizer = VisionGapVisualizer() if args.visualize else None
    
    # Process frames
    results = []
    successful_frames = 0
    
    for i in range(start_frame, end_frame):
        timestamp = sorted_timestamps[i]
        vision_file = vision_files[timestamp]
        
        print(f"Processing frame {i+1}/{end_frame}: {timestamp}")
        
        # Analyze frame
        frame_result = analyze_vision_frame(vision_file, vision_follower, visualizer, i, args.visualize)
        
        if frame_result:
            frame_result['timestamp'] = timestamp.isoformat()
            frame_result['vision_file'] = vision_file
            results.append(frame_result)
            
            if frame_result['has_solution']:
                successful_frames += 1
            
            # Print frame summary
            if frame_result['has_solution']:
                print(f"  [OK] Steering: {frame_result['steering_angle_deg']:.1f}°, Gaps: {frame_result['gaps_found']}")
            else:
                print(f"  [ERROR] No solution, Gaps: {frame_result['gaps_found']}")
        else:
            print(f"  [ERROR] Processing failed")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary statistics
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    
    if results:
        df = pd.DataFrame(results)
        
        print(f"Frames processed: {len(results)}")
        print(f"Successful solutions: {successful_frames}/{len(results)} ({successful_frames/len(results)*100:.1f}%)")
        
        # Steering statistics
        valid_steering = df[df['has_solution']]
        if len(valid_steering) > 0:
            print(f"\nSteering Analysis:")
            print(f"  Mean angle: {valid_steering['steering_angle_deg'].mean():.1f}°")
            print(f"  Std deviation: {valid_steering['steering_angle_deg'].std():.1f}°")
            print(f"  Min angle: {valid_steering['steering_angle_deg'].min():.1f}°")
            print(f"  Max angle: {valid_steering['steering_angle_deg'].max():.1f}°")
        
        # Gap statistics
        print(f"\nGap Analysis:")
        print(f"  Mean gaps per frame: {df['gaps_found'].mean():.1f}")
        print(f"  Max gaps in frame: {df['gaps_found'].max()}")
        print(f"  Frames with 0 gaps: {(df['gaps_found'] == 0).sum()}")
        print(f"  Frames with 1+ gaps: {(df['gaps_found'] >= 1).sum()}")
    
    print(f"\nResults saved to: {args.output}")
    if args.visualize:
        print(f"Visualizations saved to: images/vision/")
    
    print(f"\n[OK] Analysis complete!")


if __name__ == "__main__":
    main()