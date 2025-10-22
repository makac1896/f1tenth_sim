"""
F1Tenth Gap Follow Analysis Script

Processes specified number of frames and compares calculated gap follow results
with actual car logged data. Provides detailed analysis and statistics.

Usage:
    python gap_analysis.py --frames 100 --output results.csv
    python gap_analysis.py --start 0 --end 500 --tolerance 5.0
"""
import sys
import os
import argparse
import csv
from datetime import datetime
from math import degrees

# Add sim package to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sim'))

from sim.data.data_loader import F1TenthDataLoader
from sim.algorithms.lidar_gap_follow import LidarGapFollower
from sim.algorithms.lidar_safety import LidarSafetyChecker
from sim.utils.visualization import format_debug_output, timestamp_to_string
from sim.utils.image_processing import ImageProcessor


class GapAnalysisResults:
    """Container for analysis results and statistics"""
    
    def __init__(self):
        self.frame_results = []
        self.total_frames = 0
        self.frames_with_car_data = 0
        self.successful_comparisons = 0
        
    def add_frame_result(self, result):
        """Add a frame analysis result"""
        self.frame_results.append(result)
        self.total_frames += 1
        
        if result['car_data_available']:
            self.frames_with_car_data += 1
            
        if result['comparison_valid']:
            self.successful_comparisons += 1
    
    def get_statistics(self):
        """Calculate summary statistics"""
        if not self.frame_results:
            return {}
        
        # Calculate angle differences for valid comparisons
        angle_diffs = []
        gap_count_diffs = []
        
        for result in self.frame_results:
            if result['comparison_valid']:
                angle_diffs.append(result['steering_angle_diff_deg'])
                gap_count_diffs.append(result['gap_count_diff'])
        
        stats = {
            'total_frames': self.total_frames,
            'frames_with_car_data': self.frames_with_car_data,
            'successful_comparisons': self.successful_comparisons,
            'car_data_rate': self.frames_with_car_data / self.total_frames if self.total_frames > 0 else 0,
            'comparison_rate': self.successful_comparisons / self.total_frames if self.total_frames > 0 else 0
        }
        
        if angle_diffs:
            stats.update({
                'avg_angle_diff_deg': sum(angle_diffs) / len(angle_diffs),
                'max_angle_diff_deg': max(angle_diffs),
                'min_angle_diff_deg': min(angle_diffs),
                'avg_gap_count_diff': sum(gap_count_diffs) / len(gap_count_diffs),
                'max_gap_count_diff': max(gap_count_diffs),
                'agreement_within_5deg': sum(1 for diff in angle_diffs if abs(diff) <= 5.0) / len(angle_diffs),
                'agreement_within_10deg': sum(1 for diff in angle_diffs if abs(diff) <= 10.0) / len(angle_diffs)
            })
        
        return stats


def analyze_frame_direct(pair_data, gap_follower, safety_checker, store_gaps=False):
    """
    Analyze a single frame directly from pair data (for lidar-only processing)
    
    Args:
        pair_data (dict): Synchronized pair data or mock pair
        gap_follower (LidarGapFollower): Gap following algorithm
        safety_checker (LidarSafetyChecker): Safety assessment algorithm
        store_gaps (bool): Whether to store gap data for visualization
        
    Returns:
        dict: Analysis results for this frame
    """
    lidar_data = pair_data['lidar_data']
    if not lidar_data:
        return None
    
    # Run gap following algorithm
    metadata = lidar_data['scan_metadata']
    gap_result = gap_follower.process_lidar_data(
        lidar_data['ranges'], 
        metadata['angle_min'], 
        metadata['angle_max']
    )
    
    # Run safety assessment
    metadata = lidar_data['scan_metadata']
    safety_result = safety_checker.assess_safety(
        lidar_data['ranges'], 
        current_speed=1.0,  # Default speed
        angle_min=metadata['angle_min'],
        angle_max=metadata['angle_max'],
        range_min=metadata['range_min'],
        range_max=metadata['range_max']
    )
    
    # Check if car data is available (from drive_log)
    car_data = lidar_data.get('drive_log', {})
    car_data_available = bool(car_data)
    
    # Build result dictionary
    result = {
        'frame_index': pair_data['index'],
        'timestamp': pair_data['lidar_timestamp'],
        'car_data_available': car_data_available,
        'comparison_valid': False,
        
        # Our algorithm results
        'our_steering_angle_deg': degrees(gap_result['steering_angle']) if gap_result['steering_angle'] is not None else None,
        'our_gap_count': len(gap_result['debug']['gaps']),
        'our_largest_gap_size': len(max(gap_result['debug']['gaps'], key=len)) if gap_result['debug']['gaps'] else 0,
        
        # Car logged data (filled in below if available)
        'car_steering_angle_deg': None,
        'car_gap_count': None,
        'car_gap_size': None,
        'car_midpoint_angle_deg': None,
        'car_distance_m': None,
        
        # Safety assessment
        'safety_status': safety_result['safety_status'],
        'time_to_collision': safety_result['time_to_collision'],
        
        # Comparison metrics (filled in below if comparison is valid)
        'steering_angle_diff_deg': None,
        'gap_count_diff': None,
        'gap_size_diff': None,
        
        # Store gap data for visualization (not saved to CSV)
        '_gaps': gap_result['debug']['gaps'] if store_gaps else None,
        '_selected_gap_idx': 0 if gap_result['debug']['gaps'] else None
    }
    
    # Fill in car data if available
    if car_data_available:
        result['car_steering_angle_deg'] = degrees(car_data.get('steering_angle', 0))
        result['car_gap_count'] = car_data.get('gap_count', 0)
        result['car_gap_size'] = car_data.get('best_gap_size', 0)
        result['car_midpoint_angle_deg'] = car_data.get('midpoint_angle_deg', 0)
        result['car_distance_m'] = car_data.get('distance_m', 0)
        
        # Calculate comparison metrics if both have valid data
        if result['our_steering_angle_deg'] is not None and result['car_steering_angle_deg'] is not None:
            result['comparison_valid'] = True
            result['steering_angle_diff_deg'] = result['our_steering_angle_deg'] - result['car_steering_angle_deg']
            result['gap_count_diff'] = result['our_gap_count'] - result['car_gap_count']
            result['gap_size_diff'] = result['our_largest_gap_size'] - result['car_gap_size']
    
    return result


def analyze_frame(data_loader, frame_index, gap_follower, safety_checker, store_gaps=False):
    """
    Analyze a single frame and compare with car data
    
    Args:
        data_loader: F1TenthDataLoader instance
        frame_index: Frame index to analyze
        gap_follower: LidarGapFollower instance
        safety_checker: LidarSafetyChecker instance
        store_gaps: Store gap indices for visualization
        
    Returns:
        dict: Frame analysis results
    """
    # Load synchronized data
    pair = data_loader.get_synchronized_pair(frame_index)
    if not pair:
        return None
    
    lidar_data = pair['lidar_data']
    ranges = lidar_data['ranges']
    metadata = lidar_data['scan_metadata']
    
    # Run our algorithms
    gap_result = gap_follower.process_lidar_data(
        ranges, metadata['angle_min'], metadata['angle_max']
    )
    
    safety_result = safety_checker.assess_safety(
        ranges, 1.0,  # Assume 1 m/s speed
        metadata['angle_min'], metadata['angle_max'],
        metadata['range_min'], metadata['range_max']
    )
    
    # Extract car logged data
    car_data = lidar_data.get('drive_log')
    car_data_available = car_data is not None
    
    # Prepare result structure
    result = {
        'frame_index': frame_index,
        'timestamp': pair['lidar_timestamp'],
        'car_data_available': car_data_available,
        'comparison_valid': False,
        
        # Our algorithm results
        'our_steering_angle_deg': degrees(gap_result['steering_angle']) if gap_result['steering_angle'] else None,
        'our_gap_count': gap_result['debug']['gaps_found'],
        'our_largest_gap_size': len(max(gap_result['debug']['gaps'], key=len)) if gap_result['debug']['gaps'] else 0,
        'safety_status': safety_result['safety_status'],
        'time_to_collision': safety_result['time_to_collision'],
        
        # Car data (if available)
        'car_steering_angle_deg': None,
        'car_gap_count': None,
        'car_gap_size': None,
        'car_midpoint_angle_deg': None,
        'car_distance_m': None,
        
        # Comparison metrics (if valid)
        'steering_angle_diff_deg': None,
        'gap_count_diff': None,
        'gap_size_diff': None,
        
        # Store gap data for visualization (not saved to CSV)
        '_gaps': gap_result['debug']['gaps'] if store_gaps else None,
        '_selected_gap_idx': 0 if gap_result['debug']['gaps'] else None
    }
    
    # Fill in car data if available
    if car_data_available:
        result['car_steering_angle_deg'] = degrees(car_data.get('steering_angle', 0))
        result['car_gap_count'] = car_data.get('gap_count', 0)
        result['car_gap_size'] = car_data.get('best_gap_size', 0)
        result['car_midpoint_angle_deg'] = car_data.get('midpoint_angle_deg', 0)
        result['car_distance_m'] = car_data.get('distance_m', 0)
        
        # Calculate comparison metrics if both have valid data
        if result['our_steering_angle_deg'] is not None and result['car_steering_angle_deg'] is not None:
            result['comparison_valid'] = True
            result['steering_angle_diff_deg'] = result['our_steering_angle_deg'] - result['car_steering_angle_deg']
            result['gap_count_diff'] = result['our_gap_count'] - result['car_gap_count']
            result['gap_size_diff'] = result['our_largest_gap_size'] - result['car_gap_size']
    
    return result


def save_results_to_csv(results, output_file):
    """Save analysis results to CSV file"""
    if not results.frame_results:
        print("No results to save")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    fieldnames = [
        'frame_index', 'timestamp', 'car_data_available', 'comparison_valid',
        'our_steering_angle_deg', 'our_gap_count', 'our_largest_gap_size',
        'car_steering_angle_deg', 'car_gap_count', 'car_gap_size',
        'car_midpoint_angle_deg', 'car_distance_m',
        'steering_angle_diff_deg', 'gap_count_diff', 'gap_size_diff',
        'safety_status', 'time_to_collision'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results.frame_results:
            # Convert timestamp to string for CSV and exclude internal fields
            row = result.copy()
            row['timestamp'] = timestamp_to_string(result['timestamp'])
            
            # Remove internal fields (starting with _) from CSV output
            row = {k: v for k, v in row.items() if not k.startswith('_')}
            
            writer.writerow(row)
    
    print(f"Results saved to {output_file}")


def print_summary_statistics(results):
    """Print summary statistics to console"""
    stats = results.get_statistics()
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Frames with car data: {stats['frames_with_car_data']} ({stats['car_data_rate']:.1%})")
    print(f"Valid comparisons: {stats['successful_comparisons']} ({stats['comparison_rate']:.1%})")
    
    if 'avg_angle_diff_deg' in stats:
        print(f"\nSTEERING ANGLE COMPARISON:")
        print(f"  Average difference: {stats['avg_angle_diff_deg']:.2f}°")
        print(f"  Max difference: {stats['max_angle_diff_deg']:.2f}°")
        print(f"  Min difference: {stats['min_angle_diff_deg']:.2f}°")
        print(f"  Agreement within ±5°: {stats['agreement_within_5deg']:.1%}")
        print(f"  Agreement within ±10°: {stats['agreement_within_10deg']:.1%}")
        
        print(f"\nGAP COUNT COMPARISON:")
        print(f"  Average gap count difference: {stats['avg_gap_count_diff']:.1f}")
        print(f"  Max gap count difference: {stats['max_gap_count_diff']}")


def generate_output_filename(start_frame, end_frame, dataset_name="lidar_1"):
    """Generate a descriptive output filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_range = f"frames_{start_frame:04d}-{end_frame-1:04d}"
    return f"analysis_output/gap_analysis_{dataset_name}_{frame_range}_{timestamp}.csv"


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Analyze F1Tenth gap follow algorithm performance')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to process')
    parser.add_argument('--start', type=int, default=0, help='Starting frame index')
    parser.add_argument('--end', type=int, help='Ending frame index (overrides --frames)')
    parser.add_argument('--start-time', type=str, help='Start timestamp (e.g., "2025-10-21T16:52:47.670353Z")')
    parser.add_argument('--end-time', type=str, help='End timestamp (e.g., "2025-10-21T16:52:50.000000Z")')
    parser.add_argument('--output', type=str, help='Output CSV file (auto-generated if not specified)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed frame-by-frame results')
    parser.add_argument('--visualize', action='store_true', help='Generate gap visualization images')
    parser.add_argument('--lidar-dir', type=str, default='all_logs/lidar/lidar_1', help='Lidar data directory (e.g., all_logs/lidar/lidar_2)')
    parser.add_argument('--vision-dir', type=str, default='all_logs/vision/vision_1', help='Vision data directory (e.g., all_logs/vision/vision_2)')
    
    args = parser.parse_args()
    
    print("F1Tenth Gap Follow Analysis")
    print("="*40)
    
    # Initialize data loader
    try:
        # Only load vision data if visualization is requested
        vision_dir = args.vision_dir if args.visualize else None
        data_loader = F1TenthDataLoader(args.lidar_dir, vision_dir)
        
        if args.visualize:
            print(f"Loaded {len(data_loader)} synchronized pairs")
        else:
            print(f"Loaded {len(data_loader.lidar_files)} lidar files")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if len(data_loader) == 0:
        print("No data available for analysis")
        return
    
    # Determine frame range based on timestamps if provided
    start_frame = args.start
    end_frame = None
    
    if args.start_time or args.end_time:
        print("Filtering by timestamp range...")
        
        # Parse timestamps if provided
        start_timestamp = None
        end_timestamp = None
        
        if args.start_time:
            try:
                start_timestamp = datetime.fromisoformat(args.start_time.replace('Z', '+00:00'))
                print(f"Start time: {start_timestamp}")
            except ValueError as e:
                print(f"Error parsing start time '{args.start_time}': {e}")
                return
        
        if args.end_time:
            try:
                end_timestamp = datetime.fromisoformat(args.end_time.replace('Z', '+00:00'))
                print(f"End time: {end_timestamp}")
            except ValueError as e:
                print(f"Error parsing end time '{args.end_time}': {e}")
                return
        
        # Find frame indices that match timestamp range
        matching_frames = []
        
        if args.visualize and len(data_loader) > 0:
            # Use synchronized pairs for visualization
            for i in range(len(data_loader)):
                pair = data_loader.get_synchronized_pair(i)
                if pair and pair['lidar_data']:
                    frame_timestamp = pair['lidar_timestamp']
                    
                    # Check if frame is within timestamp range
                    if start_timestamp and frame_timestamp < start_timestamp:
                        continue
                    if end_timestamp and frame_timestamp > end_timestamp:
                        continue
                        
                    matching_frames.append(i)
        else:
            # Use lidar files directly when no visualization needed
            lidar_timestamps = sorted(data_loader.lidar_files.keys())
            for i, timestamp in enumerate(lidar_timestamps):
                # Check if frame is within timestamp range
                if start_timestamp and timestamp < start_timestamp:
                    continue
                if end_timestamp and timestamp > end_timestamp:
                    continue
                    
                matching_frames.append(i)
        
        if not matching_frames:
            print("No frames found within the specified timestamp range")
            return
        
        start_frame = matching_frames[0]
        end_frame = matching_frames[-1] + 1
        print(f"Found {len(matching_frames)} frames within timestamp range")
        
    else:
        # Use frame index range
        if args.end is not None:
            end_frame = min(args.end, len(data_loader))
        else:
            end_frame = min(start_frame + args.frames, len(data_loader))
    
    total_frames = end_frame - start_frame
    print(f"Analyzing frames {start_frame} to {end_frame-1} ({total_frames} frames)")
    
    # Generate output filename if not specified
    if args.output is None:
        dataset_name = os.path.basename(args.lidar_dir)
        args.output = generate_output_filename(start_frame, end_frame, dataset_name)
    
    print(f"Output file: {args.output}")
    
    # Initialize algorithms
    gap_follower = LidarGapFollower()
    safety_checker = LidarSafetyChecker()
    
    # Initialize results container
    results = GapAnalysisResults()
    
    # Initialize image processor if visualization requested
    image_processor = None
    if args.visualize:
        image_processor = ImageProcessor()
        print("Image visualization enabled - processing images...")
    
    # Process frames
    print("\nProcessing frames...")
    
    if args.visualize and len(data_loader) > 0:
        # Process synchronized pairs for visualization
        for i in range(start_frame, end_frame):
            if i % 50 == 0:
                print(f"  Frame {i}/{end_frame-1}")
            
            result = analyze_frame(data_loader, i, gap_follower, safety_checker, store_gaps=args.visualize)
            if result:
                results.add_frame_result(result)
    else:
        # Process lidar files directly
        lidar_timestamps = sorted(data_loader.lidar_files.keys())
        for i in range(start_frame, min(end_frame, len(lidar_timestamps))):
            if i % 50 == 0:
                print(f"  Frame {i}/{min(end_frame, len(lidar_timestamps))-1}")
            
            timestamp = lidar_timestamps[i]
            lidar_data = data_loader.load_lidar_data(timestamp)
            if lidar_data:
                # Create a mock pair structure for analyze_frame
                mock_pair = {
                    'lidar_data': lidar_data,
                    'vision_data': None,
                    'lidar_timestamp': timestamp,
                    'vision_timestamp': None,
                    'time_difference_ms': 0,
                    'index': i
                }
                
                result = analyze_frame_direct(mock_pair, gap_follower, safety_checker, store_gaps=False)
                if result:
                    results.add_frame_result(result)
            
            # Print detailed results if verbose
            if args.verbose and result['comparison_valid']:
                print(f"\nFrame {i}: {timestamp_to_string(result['timestamp'])}")
                print(f"  Our steering: {result['our_steering_angle_deg']:.2f}°")
                print(f"  Car steering: {result['car_steering_angle_deg']:.2f}°")
                print(f"  Difference: {result['steering_angle_diff_deg']:.2f}°")
                print(f"  Gap count - Our: {result['our_gap_count']}, Car: {result['car_gap_count']}")
    
    # Save results and print summary
    save_results_to_csv(results, args.output)
    print_summary_statistics(results)
    
    # Generate visualizations if requested
    if args.visualize and image_processor:
        print("\nGenerating gap visualizations...")
        processed_images = image_processor.process_analysis_results(results, data_loader)
        print(f"Generated {processed_images} visualization images in images/lidar/")
    
    print(f"\nAnalysis complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()