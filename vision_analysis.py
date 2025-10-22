"""
Vision Edge Detection Analysis Script

Similar to gap_analysis.py but for vision data. Processes raw vision frames
and outputs edge-detected images to images/vision/

Usage:
    python vision_analysis.py --frames 10
    python vision_analysis.py --vision-dir all_logs/vision/vision_2 --frames 50
    python vision_analysis.py --start-time "2025-10-21T16:52:47.000000Z" --end-time "2025-10-21T16:52:50.000000Z"
"""
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

# Add sim package to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sim'))

from sim.algorithms.vision_gap_follow import SimpleVisionGapFollower
from raw_to_png_converter import raw_to_png
import cv2
import glob
import shutil


def find_vision_files(vision_dir):
    """Find all .raw vision files in directory"""
    vision_files = {}
    pattern = os.path.join(vision_dir, "image_*.raw")
    
    for file_path in Path(vision_dir).glob("image_*.raw"):
        filename = file_path.name
        # Extract timestamp from filename: image_20251021T165247.546201Z.raw
        timestamp_str = filename.replace("image_", "").replace(".raw", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            vision_files[timestamp] = str(file_path)
        except ValueError:
            continue
    
    return vision_files


def filter_by_timestamp(vision_files, start_time=None, end_time=None):
    """Filter vision files by timestamp range"""
    if not start_time and not end_time:
        return vision_files
    
    filtered = {}
    for timestamp, file_path in vision_files.items():
        if start_time and timestamp < start_time:
            continue
        if end_time and timestamp > end_time:
            continue
        filtered[timestamp] = file_path
    
    return filtered


def process_vision_frames(vision_files, vision_processor, start_idx=0, end_idx=None):
    """Process vision frames and generate edge images"""
    timestamps = sorted(vision_files.keys())
    
    if end_idx is None:
        end_idx = len(timestamps)
    
    processed_count = 0
    
    for i in range(start_idx, min(end_idx, len(timestamps))):
        timestamp = timestamps[i]
        file_path = vision_files[timestamp]
        
        print(f"Processing frame {i+1}/{len(timestamps)}: {Path(file_path).name}")
        
        try:
            # Convert raw to PNG
            png_path = raw_to_png(file_path, "temp_vision_processing")
            if not png_path:
                print(f"  Failed to convert {file_path}")
                continue
            
            # Load image
            image = cv2.imread(png_path)
            os.remove(png_path)  # cleanup temp file
            
            if image is None:
                print(f"  Failed to load converted image")
                continue
            
            # Generate output filename
            time_str = timestamp.strftime("%H%M%S%f")[:-3]  # HHMMSSMMM
            output_filename = f"edges_frame_{i:04d}_{time_str}.png"
            
            # Process with edge detection
            vision_processor.process_image(image, output_filename, is_test=False)
            processed_count += 1
            
        except Exception as e:
            print(f"  Error processing frame: {e}")
            continue
    
    return processed_count


def clean_old_images(output_dir):
    """Clean old edge images from previous runs"""
    if not os.path.exists(output_dir):
        return
    
    # Find all edge image files
    edge_files = glob.glob(os.path.join(output_dir, "edges_frame_*.png"))
    
    if edge_files:
        print(f"Cleaning {len(edge_files)} old edge images from {output_dir}")
        for file_path in edge_files:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"  Warning: Could not remove {file_path}: {e}")
    else:
        print(f"No old edge images found in {output_dir}")


def main():
    """Main vision analysis function"""
    parser = argparse.ArgumentParser(description='Process vision frames with edge detection')
    parser.add_argument('--frames', type=int, default=10, help='Number of frames to process')
    parser.add_argument('--start', type=int, default=0, help='Starting frame index')
    parser.add_argument('--end', type=int, help='Ending frame index (overrides --frames)')
    parser.add_argument('--start-time', type=str, help='Start timestamp (e.g., "2025-10-21T16:52:47.000000Z")')
    parser.add_argument('--end-time', type=str, help='End timestamp (e.g., "2025-10-21T16:52:50.000000Z")')
    parser.add_argument('--vision-dir', type=str, default='all_logs/vision/vision_1', 
                       help='Vision data directory (e.g., all_logs/vision/vision_2)')
    
    args = parser.parse_args()
    
    print("Vision Edge Detection Analysis")
    print("=" * 40)
    
    # Find vision files
    if not os.path.exists(args.vision_dir):
        print(f"Error: Vision directory {args.vision_dir} does not exist")
        return
    
    vision_files = find_vision_files(args.vision_dir)
    print(f"Found {len(vision_files)} vision files")
    
    if not vision_files:
        print("No vision files found")
        return
    
    # Filter by timestamp if provided
    if args.start_time or args.end_time:
        print("Filtering by timestamp range...")
        
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
        
        vision_files = filter_by_timestamp(vision_files, start_timestamp, end_timestamp)
        print(f"Filtered to {len(vision_files)} files within timestamp range")
        
        if not vision_files:
            print("No files found within specified timestamp range")
            return
    
    # Determine frame range
    start_frame = args.start
    if args.end is not None:
        end_frame = min(args.end, len(vision_files))
    else:
        end_frame = min(start_frame + args.frames, len(vision_files))
    
    total_frames = end_frame - start_frame
    print(f"Processing frames {start_frame} to {end_frame-1} ({total_frames} frames)")
    
    # Clean old images from previous runs
    output_dir = "images/vision"
    clean_old_images(output_dir)
    
    # Initialize vision processor
    vision_processor = SimpleVisionGapFollower()
    
    # Process frames
    print("\nProcessing vision frames...")
    processed_count = process_vision_frames(
        vision_files, 
        vision_processor, 
        start_frame, 
        end_frame
    )
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {processed_count}/{total_frames} frames")
    print(f"Edge images saved to: images/vision/")


if __name__ == "__main__":
    main()