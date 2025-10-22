#!/usr/bin/env python3
"""
Analyze timestamp alignment between lidar and vision data folders.
"""

import os
import re
from datetime import datetime
from collections import defaultdict


def extract_timestamp_from_filename(filename):
    """Extract ISO timestamp from filename."""
    # Pattern for both lidar and vision files
    pattern = r'(scan|image)_(\d{8}T\d{6}\.\d{6}Z)'
    match = re.search(pattern, filename)
    if match:
        return match.group(2)
    return None


def parse_timestamp(timestamp_str):
    """Parse ISO timestamp string to datetime object."""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except:
        return None


def analyze_folder_timestamps(folder_path):
    """Analyze timestamps in a folder and return time range and count."""
    if not os.path.exists(folder_path):
        return None, None, 0
    
    timestamps = []
    files = os.listdir(folder_path)
    
    for filename in files:
        if filename.endswith('.json') or filename.endswith('.raw'):
            timestamp_str = extract_timestamp_from_filename(filename)
            if timestamp_str:
                timestamp = parse_timestamp(timestamp_str)
                if timestamp:
                    timestamps.append(timestamp)
    
    if not timestamps:
        return None, None, 0
    
    timestamps.sort()
    return timestamps[0], timestamps[-1], len(timestamps)


def find_timestamp_overlap(ts1_start, ts1_end, ts2_start, ts2_end, tolerance_seconds=10):
    """Check if two timestamp ranges overlap within tolerance."""
    if ts1_start is None or ts2_start is None:
        return False, 0
    
    # Calculate overlap
    overlap_start = max(ts1_start, ts2_start)
    overlap_end = min(ts1_end, ts2_end)
    
    if overlap_start <= overlap_end:
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        return True, overlap_duration
    
    # Check if they're close enough within tolerance
    gap = min(abs((ts1_start - ts2_end).total_seconds()), 
              abs((ts2_start - ts1_end).total_seconds()))
    
    if gap <= tolerance_seconds:
        return True, -gap  # Negative indicates gap, not overlap
    
    return False, gap


def main():
    base_path = r"c:\Users\makac\git\391_sim\all_logs"
    lidar_path = os.path.join(base_path, "lidar")
    vision_path = os.path.join(base_path, "vision")
    
    print("=== LIDAR AND VISION TIMESTAMP ALIGNMENT ANALYSIS ===\n")
    
    # Analyze each lidar folder
    lidar_data = {}
    for i in range(1, 6):  # lidar_1 to lidar_5
        folder_name = f"lidar_{i}"
        folder_path = os.path.join(lidar_path, folder_name)
        start, end, count = analyze_folder_timestamps(folder_path)
        lidar_data[i] = {
            'start': start,
            'end': end,
            'count': count,
            'folder': folder_name
        }
    
    # Analyze each vision folder
    vision_data = {}
    for i in range(1, 5):  # vision_1 to vision_4
        folder_name = f"vision_{i}"
        folder_path = os.path.join(vision_path, folder_name)
        start, end, count = analyze_folder_timestamps(folder_path)
        vision_data[i] = {
            'start': start,
            'end': end,
            'count': count,
            'folder': folder_name
        }
    
    # Print individual folder analysis
    print("LIDAR FOLDERS:")
    for i, data in lidar_data.items():
        if data['start']:
            print(f"  {data['folder']}: {data['start'].strftime('%H:%M:%S')} - {data['end'].strftime('%H:%M:%S')} ({data['count']} files)")
        else:
            print(f"  {data['folder']}: EMPTY")
    
    print("\nVISION FOLDERS:")
    for i, data in vision_data.items():
        if data['start']:
            print(f"  {data['folder']}: {data['start'].strftime('%H:%M:%S')} - {data['end'].strftime('%H:%M:%S')} ({data['count']} files)")
        else:
            print(f"  {data['folder']}: EMPTY")
    
    # Find matching pairs
    print("\n" + "="*60)
    print("TIMESTAMP ALIGNMENT ANALYSIS:")
    print("="*60)
    
    matches = []
    for lidar_idx, lidar_info in lidar_data.items():
        if lidar_info['start'] is None:
            continue
            
        for vision_idx, vision_info in vision_data.items():
            if vision_info['start'] is None:
                continue
            
            overlaps, duration = find_timestamp_overlap(
                lidar_info['start'], lidar_info['end'],
                vision_info['start'], vision_info['end'],
                tolerance_seconds=30  # Allow 30 second tolerance
            )
            
            if overlaps:
                matches.append({
                    'lidar': lidar_idx,
                    'vision': vision_idx,
                    'duration': duration,
                    'lidar_info': lidar_info,
                    'vision_info': vision_info
                })
    
    # Sort matches by overlap duration (descending)
    matches.sort(key=lambda x: x['duration'], reverse=True)
    
    if matches:
        print("ALIGNED PAIRS (sorted by overlap duration):")
        for match in matches:
            lidar_folder = match['lidar_info']['folder']
            vision_folder = match['vision_info']['folder']
            duration = match['duration']
            
            if duration > 0:
                print(f"  ✓ {lidar_folder} ↔ {vision_folder}")
                print(f"    Overlap: {duration:.1f} seconds")
                print(f"    LIDAR:  {match['lidar_info']['start'].strftime('%H:%M:%S')} - {match['lidar_info']['end'].strftime('%H:%M:%S')} ({match['lidar_info']['count']} files)")
                print(f"    VISION: {match['vision_info']['start'].strftime('%H:%M:%S')} - {match['vision_info']['end'].strftime('%H:%M:%S')} ({match['vision_info']['count']} files)")
            else:
                print(f"  ~ {lidar_folder} ↔ {vision_folder}")
                print(f"    Gap: {abs(duration):.1f} seconds (within tolerance)")
                print(f"    LIDAR:  {match['lidar_info']['start'].strftime('%H:%M:%S')} - {match['lidar_info']['end'].strftime('%H:%M:%S')} ({match['lidar_info']['count']} files)")
                print(f"    VISION: {match['vision_info']['start'].strftime('%H:%M:%S')} - {match['vision_info']['end'].strftime('%H:%M:%S')} ({match['vision_info']['count']} files)")
            print()
    else:
        print("No aligned pairs found within tolerance.")
    
    # Find unmatched folders
    matched_lidar = {match['lidar'] for match in matches}
    matched_vision = {match['vision'] for match in matches}
    
    unmatched_lidar = []
    unmatched_vision = []
    
    for i, data in lidar_data.items():
        if data['start'] and i not in matched_lidar:
            unmatched_lidar.append(i)
    
    for i, data in vision_data.items():
        if data['start'] and i not in matched_vision:
            unmatched_vision.append(i)
    
    if unmatched_lidar or unmatched_vision:
        print("UNMATCHED FOLDERS:")
        for i in unmatched_lidar:
            data = lidar_data[i]
            print(f"  LIDAR {data['folder']}: {data['start'].strftime('%H:%M:%S')} - {data['end'].strftime('%H:%M:%S')} ({data['count']} files)")
        
        for i in unmatched_vision:
            data = vision_data[i]
            print(f"  VISION {data['folder']}: {data['start'].strftime('%H:%M:%S')} - {data['end'].strftime('%H:%M:%S')} ({data['count']} files)")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    total_lidar = sum(1 for data in lidar_data.values() if data['start'])
    total_vision = sum(1 for data in vision_data.values() if data['start'])
    
    print(f"Total LIDAR folders with data: {total_lidar}")
    print(f"Total VISION folders with data: {total_vision}")
    print(f"Aligned pairs found: {len(matches)}")
    print(f"Unmatched LIDAR folders: {len(unmatched_lidar)}")
    print(f"Unmatched VISION folders: {len(unmatched_vision)}")


if __name__ == "__main__":
    main()