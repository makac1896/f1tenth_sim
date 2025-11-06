#!/usr/bin/env python3
"""
Vision Hybrid Algorithm Test Script

Tests the abstracted vision hybrid algorithm (RGB, depth, and hybrid modes) 
on recorded data to validate performance and generate debugging visualizations.

Usage:
    # Test depth mode only
    python test_vision_hybrid.py --mode depth --frames 10
    
    # Test hybrid mode (depth + RGB fallback)
    python test_vision_hybrid.py --mode hybrid --frames 20 --start 5
    
    # Test all modes for comparison
    python test_vision_hybrid.py --mode all --frames 5
    
    # Test specific dataset
    python test_vision_hybrid.py --mode depth --vision_dir all_logs/vision/vision_2 --depth_dir all_logs/depth/depth_2
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
from sim.data.data_loader import F1TenthDataLoader
from raw_to_png_converter import raw_to_png


def save_depth_visualization(depth_image, filename, output_dir="images/hybrid"):
    """Create visualization of depth image for debugging"""
    os.makedirs(output_dir, exist_ok=True)
    
    if depth_image is None:
        return None
        
    # Normalize depth for visualization (0-5m range)
    depth_normalized = np.clip(depth_image, 0, 5.0) / 5.0
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Add depth scale bar
    h, w = depth_colored.shape[:2]
    scale_bar = np.zeros((50, w, 3), dtype=np.uint8)
    for i in range(w):
        depth_val = (i / w) * 5.0  # 0 to 5 meters
        color_val = int((i / w) * 255)
        scale_bar[:, i] = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
    
    # Add text labels
    cv2.putText(scale_bar, '0m', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(scale_bar, '5m', (w-40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Combine depth image with scale bar
    result = np.vstack([depth_colored, scale_bar])
    
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, result)
    return filepath


def save_hybrid_debug_stages(rgb_image, depth_image, result, frame_name, output_dir="images/hybrid"):
    """Save all processing stages for hybrid algorithm debugging"""
    os.makedirs(output_dir, exist_ok=True)
    
    debug = result['debug']
    height, width = rgb_image.shape[:2]
    
    # Get ROI coordinates
    roi_top_fraction = 0.5  # Same as algorithm
    top = int(height * roi_top_fraction)
    bottom = height
    
    # 1. Original RGB with ROI
    rgb_with_roi = rgb_image.copy()
    cv2.rectangle(rgb_with_roi, (0, top), (width, bottom), (0, 0, 255), 2)
    cv2.putText(rgb_with_roi, f"Mode: {debug['mode']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(rgb_with_roi, 'ROI', (10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(f"{output_dir}/{frame_name}_1_rgb_original.png", rgb_with_roi)
    
    # 2. RGB Free space detection (if available)
    if 'rgb_free_space' in debug:
        rgb_free_space = debug['rgb_free_space'].copy()
        cv2.rectangle(rgb_free_space, (0, top), (width, bottom), (255, 0, 0), 2)
        cv2.putText(rgb_free_space, 'RGB Free Space', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(f"{output_dir}/{frame_name}_2_rgb_free_space.png", rgb_free_space)
    
    # 3. Depth visualization
    if depth_image is not None:
        depth_viz_path = save_depth_visualization(depth_image, f"{frame_name}_3_depth_original.png", output_dir)
        
        # 4. Depth ROI and processing
        depth_with_roi = depth_image.copy()
        # Normalize for visualization
        depth_norm = np.clip(depth_with_roi, 0, 5.0) / 5.0
        depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.rectangle(depth_colored, (0, top), (width, bottom), (255, 255, 255), 2)
        cv2.putText(depth_colored, 'Depth ROI', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(f"{output_dir}/{frame_name}_4_depth_roi.png", depth_colored)
        
        # 5. Depth navigability
        if 'depth_navigable' in debug:
            nav_viz = depth_colored.copy()
            navigable = debug['depth_navigable']
            for x in range(len(navigable)):
                color = (0, 255, 0) if navigable[x] else (0, 0, 255)
                cv2.line(nav_viz, (x, top), (x, bottom), color, 1)
            cv2.putText(nav_viz, 'Green=Navigable, Red=Blocked', (10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(f"{output_dir}/{frame_name}_5_depth_navigable.png", nav_viz)
    
    # 6. Final result with gaps
    final_image = rgb_image.copy()
    
    # Draw ROI
    cv2.rectangle(final_image, (0, top), (width, bottom), (255, 0, 0), 2)
    cv2.putText(final_image, 'ROI', (10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw RGB gaps (yellow)
    if 'rgb_gaps' in debug and debug['rgb_gaps']:
        for i, gap in enumerate(debug['rgb_gaps']):
            if isinstance(gap, tuple) and len(gap) >= 4:
                start_x, end_x, center_x, width = gap
                cv2.line(final_image, (start_x, top), (start_x, bottom), (0, 255, 255), 2)
                cv2.line(final_image, (end_x, top), (end_x, bottom), (0, 255, 255), 2)
                cv2.line(final_image, (center_x, top), (center_x, bottom), (0, 255, 0), 2)
                cv2.putText(final_image, f'RGB{i+1}', (center_x-15, top+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Draw depth gaps (cyan)
    if 'depth_gaps' in debug and debug['depth_gaps']:
        for i, gap in enumerate(debug['depth_gaps']):
            if isinstance(gap, tuple) and len(gap) >= 4:
                start_x, end_x, center_x, width = gap
                cv2.line(final_image, (start_x, top-5), (start_x, bottom-5), (255, 255, 0), 2)
                cv2.line(final_image, (end_x, top-5), (end_x, bottom-5), (255, 255, 0), 2)
                cv2.line(final_image, (center_x, top-5), (center_x, bottom-5), (255, 0, 255), 2)
                cv2.putText(final_image, f'D{i+1}', (center_x-10, top+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Add info text
    y_offset = 50
    cv2.putText(final_image, f"Mode: {debug.get('actual_mode_used', 'UNKNOWN')}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    rgb_gap_count = len(debug.get('rgb_gaps', []))
    cv2.putText(final_image, f"RGB Gaps: {rgb_gap_count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    depth_gap_count = len(debug.get('depth_gaps', []))
    cv2.putText(final_image, f"Depth Gaps: {depth_gap_count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if result['steering_angle'] is not None:
        y_offset += 25
        cv2.putText(final_image, f"Steering: {degrees(result['steering_angle']):.1f}°", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite(f"{output_dir}/{frame_name}_6_final_result.png", final_image)


def test_single_mode(data_loader, mode, start_frame=0, num_frames=5):
    """Test the hybrid algorithm in a specific mode"""
    
    print(f"\n{'='*60}")
    print(f"Testing Vision Hybrid Algorithm - Mode: {mode.upper()}")
    print(f"{'='*60}")
    
    # Initialize algorithm with the specified mode
    algorithm = VisionHybridFollower(driving_mode=mode)
    
    results = []
    successful_frames = 0
    
    for i in range(start_frame, min(start_frame + num_frames, len(data_loader))):
        print(f"\n--- Frame {i+1}: Processing ---")
        
        # Get synchronized data
        pair = data_loader.get_synchronized_pair(i)
        if not pair:
            print(f"[ERROR] No data for frame {i}")
            continue
        
        # Extract RGB image
        vision_data = pair['vision_data']
        if not vision_data or vision_data['image'] is None:
            print(f"[ERROR] No RGB image for frame {i}")
            continue
            
        rgb_image = vision_data['image']
        
        # Extract depth image (if available)
        depth_image = None
        if 'depth_data' in pair and pair['depth_data']:
            depth_image = pair['depth_data']['image']
        
        # Process with algorithm
        frame_name = f"frame_{i+1:03d}_{mode}"
        
        try:
            result = algorithm.process_image(rgb_image, depth_image)
            
            # Save debug visualizations
            save_hybrid_debug_stages(rgb_image, depth_image, result, frame_name)
            
            # Print results
            steering_angle = result['steering_angle']
            debug = result['debug']
            
            if steering_angle is not None:
                print(f"[OK] Steering: {degrees(steering_angle):.2f}°")
                successful_frames += 1
            else:
                print(f"[ERROR] No steering solution")
            
            print(f"     Mode used: {debug.get('actual_mode_used', 'UNKNOWN')}")
            rgb_gap_count = len(debug.get('rgb_gaps', []))
            print(f"     RGB gaps: {rgb_gap_count}")
            depth_gap_count = len(debug.get('depth_gaps', []))
            print(f"     Depth gaps: {depth_gap_count}")
            
            if depth_image is not None:
                print(f"     Depth available: Yes ({depth_image.shape})")
            else:
                print(f"     Depth available: No")
            
            results.append({
                'frame': i,
                'steering_angle': steering_angle,
                'mode_used': debug.get('actual_mode_used', 'UNKNOWN'),
                'rgb_gaps': len(debug.get('rgb_gaps', [])),
                'depth_gaps': len(debug.get('depth_gaps', [])),
                'success': steering_angle is not None
            })
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Mode {mode.upper()} Summary:")
    print(f"Successful frames: {successful_frames}/{num_frames}")
    print(f"Success rate: {(successful_frames/num_frames)*100:.1f}%")
    
    if results:
        successful_results = [r for r in results if r['success']]
        if successful_results:
            angles = [abs(degrees(r['steering_angle'])) for r in successful_results]
            print(f"Average steering angle: {np.mean(angles):.2f}°")
            print(f"Max steering angle: {max(angles):.2f}°")
            
            # Mode usage statistics
            mode_usage = {}
            for r in successful_results:
                mode_used = r['mode_used']
                mode_usage[mode_used] = mode_usage.get(mode_used, 0) + 1
            
            print(f"Mode usage: {mode_usage}")
    
    print(f"Visualizations saved to: images/hybrid/")
    return results


def compare_modes(data_loader, start_frame=0, num_frames=5):
    """Compare performance across all modes"""
    
    print(f"\n{'='*80}")
    print(f"COMPARATIVE ANALYSIS - All Modes")
    print(f"{'='*80}")
    
    modes = ['rgb', 'depth', 'hybrid']
    all_results = {}
    
    for mode in modes:
        print(f"\n>>> Testing {mode.upper()} mode...")
        results = test_single_mode(data_loader, mode, start_frame, num_frames)
        all_results[mode] = results
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print(f"COMPARISON REPORT")
    print(f"{'='*80}")
    
    for mode in modes:
        results = all_results[mode]
        successful = len([r for r in results if r['success']])
        success_rate = (successful / len(results) * 100) if results else 0
        
        print(f"\n{mode.upper()} Mode:")
        print(f"  Success rate: {success_rate:.1f}% ({successful}/{len(results)})")
        
        if successful > 0:
            successful_results = [r for r in results if r['success']]
            angles = [abs(degrees(r['steering_angle'])) for r in successful_results]
            print(f"  Avg steering: {np.mean(angles):.2f}°")
            print(f"  Max steering: {max(angles):.2f}°")


def main():
    parser = argparse.ArgumentParser(description='Test Vision Hybrid Algorithm')
    
    # Mode selection
    parser.add_argument('--mode', choices=['rgb', 'depth', 'hybrid', 'all'], default='depth',
                       help='Algorithm mode to test (default: depth)')
    
    # Data source
    parser.add_argument('--vision_dir', default='all_logs/vision/vision_1',
                       help='Vision data directory')
    parser.add_argument('--depth_dir', default='all_logs/depth/depth_1', 
                       help='Depth data directory')
    parser.add_argument('--lidar_dir', default='all_logs/lidar/lidar_1',
                       help='Lidar data directory (for synchronization)')
    
    # Frame selection
    parser.add_argument('--frames', type=int, default=5,
                       help='Number of frames to process')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting frame index')
    
    # Algorithm parameters
    parser.add_argument('--lookahead', type=float, default=1.0,
                       help='Depth lookahead distance (meters)')
    parser.add_argument('--min_gap_px', type=int, default=30,
                       help='Minimum gap width in pixels')
    
    args = parser.parse_args()
    
    print("Vision Hybrid Algorithm Performance Test")
    print("="*50)
    print(f"Mode: {args.mode}")
    print(f"Vision Dir: {args.vision_dir}")
    print(f"Depth Dir: {args.depth_dir}")
    print(f"Frames: {args.start} to {args.start + args.frames - 1}")
    
    # Validate directories
    for dir_path in [args.vision_dir, args.depth_dir, args.lidar_dir]:
        if not os.path.exists(dir_path):
            print(f"ERROR: Directory does not exist: {dir_path}")
            return
    
    # Initialize data loader with all three data types
    try:
        data_loader = F1TenthDataLoader(
            lidar_dir=args.lidar_dir,
            vision_dir=args.vision_dir, 
            depth_dir=args.depth_dir,
            sync_tolerance_ms=200
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize data loader: {e}")
        return
    
    # Check data availability
    if len(data_loader) == 0:
        print("ERROR: No synchronized data found!")
        return
    
    print(f"Loaded {len(data_loader)} synchronized data frames")
    
    # Create output directory
    os.makedirs("images/hybrid", exist_ok=True)
    
    # Run tests
    if args.mode == 'all':
        compare_modes(data_loader, args.start, args.frames)
    else:
        test_single_mode(data_loader, args.mode, args.start, args.frames)
    
    print(f"\n[OK] Testing complete!")
    print(f"Check images/hybrid/ for detailed visualizations")


if __name__ == "__main__":
    main()