"""
Simple test for edge detection algorithm
"""

import sys
import os
import cv2
from pathlib import Path

# Add sim package to Python path  
sys.path.append(os.path.join(os.path.dirname(__file__), 'sim'))

from sim.algorithms.vision_gap_follow import SimpleVisionGapFollower
from raw_to_png_converter import raw_to_png


def test_edge_detection():
    """Test edge detection on vision images"""
    
    # Initialize
    vision_follower = SimpleVisionGapFollower()
    print("Testing Edge Detection")
    print("=" * 20)
    
    # Find vision data
    vision_dirs = ["all_logs/vision/vision_1", "all_logs/vision/vision_2"]
    
    for vision_dir in vision_dirs:
        if os.path.exists(vision_dir):
            raw_files = list(Path(vision_dir).glob("image_*.raw"))
            print(f"Found {len(raw_files)} images in {vision_dir}")
            
            # Process first few images
            for i, raw_file in enumerate(raw_files[:5]):
                print(f"Processing {raw_file.name}...")
                
                # Convert raw to PNG
                png_path = raw_to_png(str(raw_file), "temp")
                if png_path:
                    # Load image
                    image = cv2.imread(png_path)
                    os.remove(png_path)  # cleanup
                    
                    # Apply edge detection - save to test folder
                    output_name = f"edges_{i+1}_{raw_file.stem}.png"
                    vision_follower.process_image(image, output_name, is_test=True)
                else:
                    print(f"  Failed to convert {raw_file}")
            break
    
    print("Done! Check images/vision/test/ directory")


if __name__ == "__main__":
    test_edge_detection()