#!/usr/bin/env python3
"""
Simplified Depth Column Navigability Test

Shows only the column navigability overlay - green for navigable columns,
red for blocked columns within the ROI.
"""

import argparse
import cv2
import numpy as np
import os
import json
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


def save_column_navigability(depth_image, result, frame_name, output_dir="images/depth_columns"):
    """Save column navigability visualization AND original depth image for validation"""
    os.makedirs(output_dir, exist_ok=True)
    
    if depth_image is None:
        return
    
    debug = result['debug']
    height, width = depth_image.shape
    
    # Get ROI coordinates (same as algorithm)
    roi_top_fraction = 0.5  
    top = int(height * roi_top_fraction)
    bottom = height
    
    # 1. Save original depth image for reference
    # Normalize depth for visualization (0-5m -> 0-255)
    depth_normalized = np.clip(depth_image, 0.0, 5.0) / 5.0 * 255
    depth_viz = depth_normalized.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
    
    # Add ROI boundary to original depth
    cv2.rectangle(depth_colored, (0, top), (width, bottom), (255, 255, 255), 2)
    cv2.imwrite(f"{output_dir}/{frame_name}_original.png", depth_colored)
    
    # 2. Column navigability analysis
    if 'depth_processing_stages' in debug and 'navigable_mask' in debug['depth_processing_stages']:
        navigable_mask = debug['depth_processing_stages']['navigable_mask']
        
        # Create clean black background
        column_viz = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw ROI boundary (gray)
        cv2.rectangle(column_viz, (0, top), (width, bottom), (80, 80, 80), 1)
        
        # Draw column navigability within ROI
        for x in range(width):
            if x < len(navigable_mask):
                if navigable_mask[x]:
                    # Bright green for navigable columns
                    cv2.line(column_viz, (x, top), (x, bottom), (0, 255, 0), 1)
                else:
                    # Bright red for blocked columns
                    cv2.line(column_viz, (x, top), (x, bottom), (0, 0, 255), 1)
        
        cv2.imwrite(f"{output_dir}/{frame_name}_columns.png", column_viz)
        
        # Print summary stats
        navigable_count = np.sum(navigable_mask)
        blocked_count = len(navigable_mask) - navigable_count
        print(f"  Columns: {navigable_count} navigable, {blocked_count} blocked")
        print(f"  Saved: {frame_name}_original.png (depth visualization)")
        print(f"  Saved: {frame_name}_columns.png (navigability overlay)")
        return [f"{output_dir}/{frame_name}_original.png", f"{output_dir}/{frame_name}_columns.png"]
    
    return None


def create_concise_log_entry(depth_image, result, frame_name, depth_file_path, frame_index):
    """
    Create concise log entry with key metrics for practical analysis
    
    Args:
        depth_image: Raw depth image data
        result: Algorithm processing result
        frame_name: Frame identifier
        depth_file_path: Path to original depth file
        frame_index: Frame number in sequence
        
    Returns:
        dict: Concise log entry with actionable data
    """
    debug = result['debug']
    height, width = depth_image.shape
    roi_top = int(height * 0.5)
    
    # Extract key metrics
    timestamp = Path(depth_file_path).name.replace("depth_", "").replace(".raw", "")
    steering_degrees = degrees(result['steering_angle']) if result['steering_angle'] else None
    
    # Basic frame data
    log_entry = {
        "frame": frame_index,
        "timestamp": timestamp,
        "steering_angle": round(steering_degrees, 2) if steering_degrees else None,
        "success": result['steering_angle'] is not None
    }
    
    # Depth processing results
    if 'depth_processing_stages' in debug:
        stages = debug['depth_processing_stages']
        
        # ROI depth statistics (concise)
        roi_depth = depth_image[roi_top:, :]
        valid_roi = roi_depth[(roi_depth > 0.1) & (roi_depth < 5.0)]
        
        if len(valid_roi) > 0:
            log_entry["depth_stats"] = {
                "min": round(float(np.min(valid_roi)), 2),
                "max": round(float(np.max(valid_roi)), 2), 
                "median": round(float(np.median(valid_roi)), 2),
                "mean": round(float(np.mean(valid_roi)), 2)
            }
        
        # Column navigability (summary only)
        if 'navigable_mask' in stages:
            navigable_mask = stages['navigable_mask']
            navigable_count = int(np.sum(navigable_mask))
            blocked_count = len(navigable_mask) - navigable_count
            
            log_entry["columns"] = {
                "total": len(navigable_mask),
                "navigable": navigable_count,
                "blocked": blocked_count,
                "navigable_percent": round(100 * navigable_count / len(navigable_mask), 1)
            }
            
            # Find navigable regions (continuous segments)
            navigable_regions = []
            in_region = False
            region_start = None
            
            for i, is_nav in enumerate(navigable_mask):
                if is_nav and not in_region:
                    region_start = i
                    in_region = True
                elif not is_nav and in_region:
                    navigable_regions.append({
                        "start": region_start,
                        "end": i - 1,
                        "width": i - region_start
                    })
                    in_region = False
            
            # Handle case where region extends to end
            if in_region:
                navigable_regions.append({
                    "start": region_start,
                    "end": len(navigable_mask) - 1,
                    "width": len(navigable_mask) - region_start
                })
            
            log_entry["navigable_regions"] = navigable_regions
    
    # Gap analysis (key gaps only)
    if 'selected_gaps' in debug:
        gaps = debug['selected_gaps']
        gap_summary = []
        
        for i, gap in enumerate(gaps):
            if len(gap) >= 4:  # (start, end, center, width)
                gap_summary.append({
                    "start": int(gap[0]),
                    "end": int(gap[1]),
                    "center": int(gap[2]),
                    "width": int(gap[3]),
                    "width_percent": round(100 * gap[3] / width, 1)
                })
        
        log_entry["gaps"] = gap_summary
        log_entry["gap_count"] = len(gap_summary)
    
    # QA Fix metrics for algorithm improvement tracking
    if 'depth_processing_stages' in debug:
        stages = debug['depth_processing_stages']
        qa_metrics = {}
        
        # Adaptive lookahead effectiveness
        if 'adaptive_lookahead' in stages:
            qa_metrics["adaptive_lookahead"] = round(float(stages['adaptive_lookahead']), 2)
            qa_metrics["lookahead_adaptation"] = round(float(stages['adaptive_lookahead']) - 1.0, 2)
        
        # Temporal smoothing impact
        if 'median_depth_raw' in stages and 'median_depth_smooth' in stages:
            raw_depths = stages['median_depth_raw']
            smooth_depths = stages['median_depth_smooth']
            valid_raw = raw_depths[raw_depths > 0.1]
            valid_smooth = smooth_depths[smooth_depths > 0.1]
            
            if len(valid_raw) > 0 and len(valid_smooth) > 0:
                qa_metrics["temporal_smoothing_effect"] = round(
                    float(np.std(valid_smooth) / np.std(valid_raw)), 3) if np.std(valid_raw) > 0 else 1.0
        
        # Gap stability measurement  
        if 'gaps_raw' in stages and 'gaps_smooth' in stages:
            raw_gaps = stages['gaps_raw']
            smooth_gaps = stages['gaps_smooth']
            if len(raw_gaps) > 0 and len(smooth_gaps) > 0:
                raw_center = raw_gaps[0][2] if len(raw_gaps[0]) > 2 else 320
                smooth_center = smooth_gaps[0][2] if len(smooth_gaps[0]) > 2 else 320
                qa_metrics["gap_center_shift"] = abs(int(raw_center) - int(smooth_center))
        
        log_entry["qa_fixes"] = qa_metrics
    
    return log_entry


def save_concise_logs(log_entries, output_dir="logs/depth_analysis"):
    """Save concise, actionable logs for practical analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save single consolidated log with all frames (main file for analysis)
    consolidated_log = {
        "test_info": {
            "total_frames": len(log_entries),
            "successful_frames": len([e for e in log_entries if e.get("success", False)]),
            "success_rate": len([e for e in log_entries if e.get("success", False)]) / len(log_entries),
            "test_timestamp": datetime.now().isoformat(),
            "data_source": "depth_3"
        },
        "frames": log_entries
    }
    
    with open(f"{output_dir}/depth_analysis_results.json", 'w') as f:
        json.dump(consolidated_log, f, indent=2)
    
    # 2. Save CSV format for easy spreadsheet analysis
    csv_data = []
    for entry in log_entries:
        csv_row = {
            "frame": entry.get("frame", ""),
            "timestamp": entry.get("timestamp", ""),
            "steering_angle": entry.get("steering_angle", ""),
            "success": entry.get("success", False),
            "navigable_columns": entry.get("columns", {}).get("navigable", ""),
            "blocked_columns": entry.get("columns", {}).get("blocked", ""),
            "navigable_percent": entry.get("columns", {}).get("navigable_percent", ""),
            "gap_count": entry.get("gap_count", 0),
            "median_depth": entry.get("depth_stats", {}).get("median", ""),
            "min_depth": entry.get("depth_stats", {}).get("min", ""),
            "max_depth": entry.get("depth_stats", {}).get("max", "")
        }
        csv_data.append(csv_row)
    
    # Write CSV header and data
    if csv_data:
        with open(f"{output_dir}/depth_analysis_summary.csv", 'w') as f:
            headers = csv_data[0].keys()
            f.write(",".join(headers) + "\n")
            for row in csv_data:
                f.write(",".join(str(row[h]) for h in headers) + "\n")
    
    # 3. Save quick summary statistics
    successful_entries = [e for e in log_entries if e.get("success", False)]
    
    summary_stats = {
        "overview": {
            "total_frames": len(log_entries),
            "successful_frames": len(successful_entries),
            "success_rate_percent": round(100 * len(successful_entries) / len(log_entries), 1)
        },
        "steering_analysis": {
            "average_steering": round(np.mean([e["steering_angle"] for e in successful_entries]), 2) if successful_entries else None,
            "steering_range": [
                round(np.min([e["steering_angle"] for e in successful_entries]), 2),
                round(np.max([e["steering_angle"] for e in successful_entries]), 2)
            ] if successful_entries else [None, None],
            "steering_std": round(np.std([e["steering_angle"] for e in successful_entries]), 2) if successful_entries else None
        },
        "navigation_analysis": {
            "avg_navigable_columns": round(np.mean([e.get("columns", {}).get("navigable", 0) for e in log_entries]), 1),
            "avg_navigable_percent": round(np.mean([e.get("columns", {}).get("navigable_percent", 0) for e in log_entries]), 1),
            "avg_gaps_found": round(np.mean([e.get("gap_count", 0) for e in log_entries]), 1)
        }
    }
    
    with open(f"{output_dir}/quick_summary.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    return output_dir


def test_depth_columns(depth_dir, start_frame=0, num_frames=5):
    """Test depth algorithm and show column navigability with comprehensive logging"""
    
    print(f"Depth Column Navigability Test")
    print(f"==============================")
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
    
    print(f"Processing frames {start_frame} to {end_frame-1}")
    
    successful_frames = 0
    log_entries = []  # Comprehensive logging
    
    for i in range(start_frame, end_frame):
        timestamp = sorted_timestamps[i]
        depth_file = depth_files[timestamp]
        frame_name = f"frame_{i+1:03d}"
        
        print(f"\n--- Frame {i+1}: {Path(depth_file).name} ---")
        
        # Load depth image
        depth_image = load_depth_image(depth_file)
        if depth_image is None:
            print(f"[ERROR] Failed to load depth image")
            continue
        
        # Create dummy RGB image (algorithm expects both)
        dummy_rgb = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
        
        try:
            # Process with depth algorithm
            result = algorithm.process_image(dummy_rgb, depth_image)
            
            # Save column navigability visualization
            saved_files = save_column_navigability(depth_image, result, frame_name)
            
            # Create concise log entry for this frame
            log_entry = create_concise_log_entry(
                depth_image, result, frame_name, depth_file, i+1
            )
            log_entries.append(log_entry)
            
            # Print results with prominent steering angle display
            steering_angle = result['steering_angle']
            
            if steering_angle is not None:
                print(f"[OK] 🎯 STEERING: {degrees(steering_angle):.2f}° (radians: {steering_angle:.4f})")
                successful_frames += 1
            else:
                print(f"[ERROR] ❌ NO STEERING SOLUTION")
            
            # Print concise analysis with steering angle emphasized
            if 'columns' in log_entry:
                col_info = log_entry['columns']
                print(f"  Navigable: {col_info['navigable']}/{col_info['total']} columns ({col_info['navigable_percent']}%)")
            
            if 'depth_stats' in log_entry:
                depth = log_entry['depth_stats']
                print(f"  Depth: median={depth['median']}m, range={depth['min']}-{depth['max']}m")
            
            if 'gaps' in log_entry:
                gaps = log_entry['gaps']
                print(f"  Gaps: {len(gaps)} found")
                for i, gap in enumerate(gaps):
                    print(f"    Gap {i}: pixels {gap['start']}-{gap['end']} (width: {gap['width']}px)")
            
            if saved_files:
                print(f"  Files: {len(saved_files)} visualizations saved")
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            # Still create log entry for failed frames
            log_entry = {
                "frame": i+1,
                "timestamp": Path(depth_file).name.replace("depth_", "").replace(".raw", ""),
                "steering_angle": None,
                "success": False,
                "error": str(e)
            }
            log_entries.append(log_entry)
            continue
    
    # Save concise logs
    log_dir = save_concise_logs(log_entries)
    
    # Summary with steering angle table
    print(f"\n{'='*60}")
    print(f"ALGORITHM VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Success Rate: {successful_frames}/{num_frames} frames ({100*successful_frames/num_frames:.1f}%)")
    
    # Steering angle summary table
    print(f"\n🎯 STEERING ANGLES:")
    print(f"Frame | Timestamp      | Steering (°) | Navigable | Gaps")
    print(f"------|----------------|--------------|-----------|------")
    
    for entry in log_entries:
        if entry.get('success', False):
            frame_num = entry['frame']
            timestamp = entry['timestamp'][-8:-1]  # Last 7 chars before Z
            steering = entry['steering_angle']
            navigable_pct = entry.get('columns', {}).get('navigable_percent', 0)
            gap_count = entry.get('gap_count', 0)
            print(f" {frame_num:4d} | {timestamp:14s} | {steering:9.2f}°   | {navigable_pct:6.1f}%   | {gap_count}")
    
    # Statistics
    successful_entries = [e for e in log_entries if e.get('success', False)]
    if successful_entries:
        steering_angles = [e['steering_angle'] for e in successful_entries]
        avg_steering = np.mean(steering_angles)
        std_steering = np.std(steering_angles)
        min_steering = np.min(steering_angles)
        max_steering = np.max(steering_angles)
        
        print(f"\n📊 STEERING STATISTICS:")
        print(f"  Average: {avg_steering:.2f}°")
        print(f"  Range:   {min_steering:.2f}° to {max_steering:.2f}°")
        print(f"  Std Dev: {std_steering:.3f}°")
        print(f"  Variation: {max_steering - min_steering:.2f}°")
    
    print(f"\n📁 FILES SAVED:")
    print(f"  Visualizations: images/depth_columns/ ({2*len(log_entries)} files)")
    print(f"  Analysis logs:  {log_dir}/")
    print(f"    - depth_analysis_results.json ({len(log_entries)} frames)")
    print(f"    - depth_analysis_summary.csv")  
    print(f"    - quick_summary.json")
    print(f"\n💡 Use steering angles above to verify algorithm performance!")


def main():
    parser = argparse.ArgumentParser(description='Test Depth Column Navigability')
    
    parser.add_argument('--depth_dir', default='all_logs/depth/depth_3',
                       help='Depth data directory')
    parser.add_argument('--frames', type=int, default=5,
                       help='Number of frames to process')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting frame index')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.depth_dir):
        print(f"ERROR: Depth directory does not exist: {args.depth_dir}")
        return
    
    # Create output directories
    os.makedirs("images/depth_columns", exist_ok=True)
    os.makedirs("logs/depth_analysis", exist_ok=True)
    
    # Run test
    test_depth_columns(args.depth_dir, args.start, args.frames)
    
    print(f"\n[OK] Complete! Check images/depth_columns/ for column navigability visualizations")


if __name__ == "__main__":
    main()