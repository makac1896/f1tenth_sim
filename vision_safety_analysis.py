"""
Vision Safety Analysis Script

Combines vision gap detection with AEB safety checking.
Processes vision and depth data to determine safe navigation.
"""

import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
import json

# Add sim package to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sim'))

from sim.algorithms.vision_gap_follow import VisionGapFollower
from sim.algorithms.rgb_aeb_safety import VisionAEBSafety
from raw_to_png_converter import raw_to_png
import cv2
import numpy as np


def find_vision_files(vision_dir):
    """Find all vision files (RGB only for now)"""
    rgb_files = {}
    
    # Find RGB files
    for file_path in Path(vision_dir).glob("image_*.raw"):
        filename = file_path.name
        timestamp_str = filename.replace("image_", "").replace(".raw", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            rgb_files[timestamp] = str(file_path)
        except ValueError:
            continue
    
    return rgb_files


def load_depth_image(depth_file):
    """Load depth image from raw file (assuming float32 format)"""
    try:
        # Assuming depth is stored as 960x480 float32 values
        depth_data = np.fromfile(depth_file, dtype=np.float32)
        depth_image = depth_data.reshape((480, 960))
        return depth_image
    except Exception as e:
        print(f"Error loading depth file {depth_file}: {e}")
        return None



def process_frame_with_safety(rgb_file, vision_processor, aeb_safety, frame_idx):
    """Process a single frame with both gap detection and safety checking"""
    try:
        # Load RGB image
        png_path = raw_to_png(rgb_file, "temp_safety_processing")
        if not png_path:
            return None
        
        rgb_image = cv2.imread(png_path)
        os.remove(png_path)  # cleanup
        
        if rgb_image is None:
            return None
        
        # Process with gap detection 
        vision_result = vision_processor.process_image(rgb_image)
        
        # Extract timestamp for filenames
        timestamp = Path(rgb_file).stem.replace("image_", "")
        
        # Try to load depth data (will be None for current dataset)
        depth_file = rgb_file.replace('image_', 'depth_')
        depth_image = None
        if os.path.exists(depth_file):
            depth_image = load_depth_image(depth_file)
        
        # Check safety with both RGB and depth (algorithm handles the logic)
        is_safe, reason, safety_data, annotated_image = aeb_safety.check_safety(rgb_image, depth_image)
        action = aeb_safety.get_emergency_action(is_safe, safety_data)

        # Save annotated AEB image to aeb folder
        aeb_dir = "images/vision/aeb"
        os.makedirs(aeb_dir, exist_ok=True)
        aeb_filename = f"aeb_frame_{frame_idx:04d}_{timestamp[:15]}.png"
        aeb_path = os.path.join(aeb_dir, aeb_filename)
        cv2.imwrite(aeb_path, annotated_image)

        return {
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'is_safe': is_safe,
            'reason': reason,
            'action': action['action'],
            'safety_analysis': safety_data
        }
        
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")
        return None


def clean_old_aeb_images(aeb_dir):
    """Clean old AEB images from previous runs"""
    if not os.path.exists(aeb_dir):
        return
    
    import glob
    # Find all AEB image files
    aeb_files = glob.glob(os.path.join(aeb_dir, "aeb_frame_*.png"))
    
    if aeb_files:
        print(f"Cleaning {len(aeb_files)} old AEB images from {aeb_dir}")
        for file_path in aeb_files:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"  Warning: Could not remove {file_path}: {e}")
    else:
        print(f"No old AEB images found in {aeb_dir}")


def main():
    """Main vision safety analysis function"""
    parser = argparse.ArgumentParser(description='Vision analysis with AEB safety')
    parser.add_argument('--frames', type=int, default=10, help='Number of frames to process')
    parser.add_argument('--start', type=int, default=0, help='Starting frame index')
    parser.add_argument('--vision-dir', type=str, default='all_logs/vision/vision_1', 
                       help='Vision data directory')
    parser.add_argument('--safety-distance', type=float, default=1.0, 
                       help='Minimum safe distance in meters')
    
    args = parser.parse_args()
    
    print("Vision Safety Analysis")
    print("=" * 40)
    
    if not os.path.exists(args.vision_dir):
        print(f"Error: Vision directory {args.vision_dir} does not exist")
        return
    
    # Find files
    rgb_files = find_vision_files(args.vision_dir)
    print(f"Found {len(rgb_files)} RGB files")
    
    if not rgb_files:
        print("No RGB files found")
        return
    
    timestamps = sorted(rgb_files.keys())
    end_frame = min(args.start + args.frames, len(timestamps))
    
    print(f"Processing frames {args.start} to {end_frame-1}")
    
    # Clean old AEB images from previous runs
    aeb_dir = "images/vision/aeb"
    clean_old_aeb_images(aeb_dir)
    
    # Initialize processors
    vision_processor = VisionGapFollower()
    aeb_safety = VisionAEBSafety(min_free_space_percentage=0.3, safety_buffer=0.1)
    
    # Process frames
    results = []
    unsafe_count = 0
    
    for i in range(args.start, end_frame):
        timestamp = timestamps[i]
        rgb_file = rgb_files[timestamp]
        
        print(f"Processing frame {i+1}/{len(timestamps)}: {Path(rgb_file).name}")
        
        result = process_frame_with_safety(
            rgb_file, vision_processor, aeb_safety, i
        )
        
        if result:
            # Convert all values to JSON-serializable types
            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(v) for v in obj]
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                else:
                    return obj
            
            result = make_json_serializable(result)
            results.append(result)
            if not result['is_safe']:
                unsafe_count += 1
                print(f"  UNSAFE: {result['reason']}")
            else:
                print(f"  SAFE: {result['reason']}")
    
    # Save results
    results_file = "vision_safety_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Processed {len(results)} frames")
    if len(results) > 0:
        print(f"Unsafe frames: {unsafe_count}/{len(results)} ({unsafe_count/len(results)*100:.1f}%)")
        print(f"Results saved to: {results_file}")
    else:
        print("No frames processed successfully")
    print(f"Images saved to: images/vision/")


if __name__ == "__main__":
    main()