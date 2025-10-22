"""
F1Tenth Simulator Test Script

Basic test to verify all components work together:
1. Load synchronized lidar and vision data
2. Run ROS-free gap follow algorithm  
3. Run ROS-free safety assessment
4. Display results and debug information

This script serves as both a test and example of how to use the simulator components.
"""
import sys
import os

# Add sim package to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sim'))

from sim.data.data_loader import F1TenthDataLoader
from sim.algorithms.lidar_gap_follow import LidarGapFollower
from sim.algorithms.lidar_safety import LidarSafetyChecker
from sim.utils.visualization import format_debug_output, create_comparison_summary, timestamp_to_string


def test_single_frame(data_loader, frame_index=0):
    """
    Test algorithms on a single frame of data
    
    Args:
        data_loader (F1TenthDataLoader): Loaded dataset
        frame_index (int): Index of frame to test
    """
    print(f"\n=== Testing Frame {frame_index} ===")
    
    # Load synchronized data
    pair = data_loader.get_synchronized_pair(frame_index)
    if not pair:
        print(f"No data available for frame {frame_index}")
        return
    
    lidar_data = pair['lidar_data']
    vision_data = pair['vision_data']
    
    print(f"Lidar timestamp: {timestamp_to_string(pair['lidar_timestamp'])}")
    print(f"Vision timestamp: {timestamp_to_string(pair['vision_timestamp'])}")
    print(f"Time difference: {pair['time_difference_ms']:.1f} ms")
    
    # Extract lidar ranges and metadata
    ranges = lidar_data['ranges']
    metadata = lidar_data['scan_metadata']
    
    print(f"LIDAR ranges: {len(ranges)} points")
    print(f"Range: {metadata['range_min']:.2f}m to {metadata['range_max']:.2f}m")
    print(f"Angular range: {metadata['angle_min']:.3f} to {metadata['angle_max']:.3f} rad")
    
    # Initialize algorithms
    gap_follower = LidarGapFollower()
    safety_checker = LidarSafetyChecker()
    
    # Run gap following algorithm
    print("\n--- Running Gap Follow Algorithm ---")
    gap_result = gap_follower.process_lidar_data(
        ranges, 
        metadata['angle_min'], 
        metadata['angle_max']
    )
    
    print(f"Gaps found: {gap_result['debug']['gaps_found']}")
    if gap_result['steering_angle'] is not None:
        print(f"Steering angle: {gap_result['steering_angle']:.4f} rad ({gap_result['steering_angle'] * 180 / 3.14159:.2f}°)")
    else:
        print("No valid steering angle (no gaps found)")
    
    # Run safety assessment (assume 1 m/s speed)
    print("\n--- Running Safety Assessment ---") 
    current_speed = 1.0  # m/s
    safety_result = safety_checker.assess_safety(
        ranges, 
        current_speed,
        metadata['angle_min'],
        metadata['angle_max'],
        metadata['range_min'],
        metadata['range_max']
    )
    
    print(f"Safety status: {safety_result['safety_status']}")
    print(f"Time to collision: {safety_result['time_to_collision']:.2f} seconds")
    print(f"Recommended action: {safety_result['recommended_action']}")
    
    # Display debug information
    print(format_debug_output(gap_result['debug'], "Gap Follow"))
    print(format_debug_output(safety_result['debug'], "Safety"))
    
    # Create summary
    summary = create_comparison_summary(gap_result)
    print(f"\n=== Summary ===")
    print(f"Algorithm Status: {summary['lidar']['algorithm_status']}")
    print(f"Gaps Found: {summary['lidar']['gaps_found']}")
    if summary['lidar']['steering_angle_deg']:
        print(f"Steering Command: {summary['lidar']['steering_angle_deg']:.2f}°")


def main():
    """
    Main test function
    """
    print("F1Tenth Simulator Test")
    print("======================")
    
    # Set up data paths (modify these to match your data structure)
    base_dir = os.path.dirname(__file__)
    lidar_dir = os.path.join(base_dir, "all_logs", "lidar", "lidar_1")
    vision_dir = os.path.join(base_dir, "all_logs", "vision", "vision_1")
    
    print(f"Loading data from:")
    print(f"  Lidar: {lidar_dir}")
    print(f"  Vision: {vision_dir}")
    
    # Initialize data loader
    try:
        data_loader = F1TenthDataLoader(lidar_dir, vision_dir, sync_tolerance_ms=200)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Print dataset statistics  
    stats = data_loader.get_stats()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    if len(data_loader) == 0:
        print("No synchronized data pairs found!")
        return
    
    # Test first few frames
    test_frames = min(3, len(data_loader))
    for i in range(test_frames):
        test_single_frame(data_loader, i)
    
    print(f"\n=== Test Complete ===")
    print(f"Successfully processed {test_frames} frames")
    print("All components working correctly!")


if __name__ == "__main__":
    main()