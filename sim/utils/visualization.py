"""
Utility functions for F1Tenth simulation and analysis

Simple text-based utilities for data processing, debugging, and console output.
No external dependencies - just pure Python.
"""
import numpy as np
from math import degrees, radians


def print_lidar_summary(ranges, angle_min=-2.356194496154785, angle_max=2.356194496154785, 
                       title="LIDAR Scan", show_gaps=None, steering_angle=None):
    """
    Print a text summary of LIDAR scan data
    
    Args:
        ranges (list): LIDAR range measurements
        angle_min (float): Minimum angle in radians
        angle_max (float): Maximum angle in radians  
        title (str): Summary title
        show_gaps (list, optional): List of gap indices to highlight
        steering_angle (float, optional): Steering angle to display
    """
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Range count: {len(ranges)}")
    print(f"Angular span: {degrees(angle_min):.1f}° to {degrees(angle_max):.1f}°")
    print(f"Distance range: {min(ranges):.2f}m to {max(ranges):.2f}m")
    
    if show_gaps:
        print(f"Gaps detected: {len(show_gaps)}")
        for i, gap in enumerate(show_gaps):
            gap_size = len(gap)
            start_angle = (gap[0] / len(ranges)) * (angle_max - angle_min) + angle_min
            end_angle = (gap[-1] / len(ranges)) * (angle_max - angle_min) + angle_min
            print(f"  Gap {i+1}: {gap_size} points, {degrees(start_angle):.1f}° to {degrees(end_angle):.1f}°")
    
    if steering_angle is not None:
        print(f"Steering command: {degrees(steering_angle):.2f}°")


def calculate_gap_metrics(gaps, ranges, angle_min, angle_max):
    """
    Calculate metrics for detected gaps
    
    Args:
        gaps (list): List of gap indices
        ranges (list): LIDAR range data
        angle_min (float): Minimum angle in radians
        angle_max (float): Maximum angle in radians
        
    Returns:
        list: List of gap metric dictionaries
    """
    angle_range = angle_max - angle_min
    gap_metrics = []
    
    for i, gap in enumerate(gaps):
        if not gap:
            continue
            
        start_idx, end_idx = gap[0], gap[-1]
        start_angle = (start_idx / len(ranges)) * angle_range + angle_min
        end_angle = (end_idx / len(ranges)) * angle_range + angle_min
        
        gap_metrics.append({
            'gap_id': i,
            'size_points': len(gap),
            'start_angle_deg': degrees(start_angle),
            'end_angle_deg': degrees(end_angle), 
            'width_deg': degrees(abs(end_angle - start_angle)),
            'center_angle_deg': degrees((start_angle + end_angle) / 2),
            'min_distance': min(ranges[idx] for idx in gap),
            'max_distance': max(ranges[idx] for idx in gap),
            'avg_distance': np.mean([ranges[idx] for idx in gap])
        })
    
    return gap_metrics


def format_debug_output(debug_info, algorithm_name="Algorithm"):
    """
    Format debug information for readable console output
    
    Args:
        debug_info (dict): Debug information dictionary
        algorithm_name (str): Name of the algorithm
        
    Returns:
        str: Formatted debug string
    """
    output = [f"\n=== {algorithm_name} Debug Info ==="]
    
    for key, value in debug_info.items():
        if isinstance(value, float):
            if 'angle' in key.lower():
                output.append(f"{key}: {degrees(value):.2f} deg ({value:.4f} rad)")
            else:
                output.append(f"{key}: {value:.4f}")
        elif isinstance(value, list) and len(value) > 10:
            output.append(f"{key}: List with {len(value)} items")
        else:
            output.append(f"{key}: {value}")
    
    return '\n'.join(output)


def timestamp_to_string(timestamp):
    """
    Convert timestamp to readable string
    
    Args:
        timestamp (datetime): Timestamp object
        
    Returns:
        str: Formatted timestamp string
    """
    return timestamp.strftime("%H:%M:%S.%f")[:-3]  # Remove microseconds, keep milliseconds


def validate_lidar_data(lidar_data):
    """
    Validate lidar data structure and content
    
    Args:
        lidar_data (dict): Lidar data dictionary
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(lidar_data, dict):
        return False, "Lidar data must be a dictionary"
    
    required_keys = ['ranges', 'scan_metadata']
    for key in required_keys:
        if key not in lidar_data:
            return False, f"Missing required key: {key}"
    
    ranges = lidar_data['ranges']
    if not isinstance(ranges, list) or len(ranges) == 0:
        return False, "Ranges must be a non-empty list"
    
    metadata = lidar_data['scan_metadata']
    required_metadata = ['angle_min', 'angle_max', 'ranges_count']
    for key in required_metadata:
        if key not in metadata:
            return False, f"Missing metadata key: {key}"
    
    if len(ranges) != metadata['ranges_count']:
        return False, f"Range count mismatch: expected {metadata['ranges_count']}, got {len(ranges)}"
    
    return True, "Valid"


def create_comparison_summary(lidar_result, vision_result=None):
    """
    Create a summary comparing lidar and vision algorithm results
    
    Args:
        lidar_result (dict): Lidar algorithm result
        vision_result (dict, optional): Vision algorithm result
        
    Returns:
        dict: Comparison summary
    """
    summary = {
        'lidar': {
            'steering_angle_deg': degrees(lidar_result['steering_angle']) if lidar_result['steering_angle'] else None,
            'gaps_found': len(lidar_result['debug']['gaps']) if 'debug' in lidar_result else 0,
            'algorithm_status': 'success' if lidar_result['steering_angle'] is not None else 'no_gap'
        }
    }
    
    if vision_result:
        summary['vision'] = {
            'steering_angle_deg': degrees(vision_result['steering_angle']) if vision_result.get('steering_angle') else None,
            'algorithm_status': 'success' if vision_result.get('steering_angle') is not None else 'failed'
        }
        
        # Calculate agreement if both have steering angles
        if summary['lidar']['steering_angle_deg'] and summary['vision']['steering_angle_deg']:
            angle_diff = abs(summary['lidar']['steering_angle_deg'] - summary['vision']['steering_angle_deg'])
            summary['comparison'] = {
                'angle_difference_deg': angle_diff,
                'agreement_score': max(0, 1 - angle_diff / 45)  # Scale 0-1 based on 45° max difference
            }
    
    return summary