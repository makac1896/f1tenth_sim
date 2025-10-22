#!/usr/bin/env python3
"""
Simple Vision Gap Following Test

Test the vision-based gap following algorithm on a few frames
to verify it's working correctly before moving on to comparisons.
"""

import cv2
from sim.algorithms.vision_gap_follow import VisionGapFollower
from raw_to_png_converter import raw_to_png
from math import degrees

def test_vision_algorithm():
    """Test vision gap following on real data"""
    print("Vision Gap Following Algorithm Test")
    print("=" * 40)
    
    # Initialize algorithm
    vision_follower = VisionGapFollower()
    
    # Test files
    test_files = [
        "all_logs/vision/vision_1/image_20251021T165247.546201Z.raw",
        "all_logs/vision/vision_1/image_20251021T165247.679219Z.raw",
        "all_logs/vision/vision_1/image_20251021T165247.878724Z.raw"
    ]
    
    for i, image_file in enumerate(test_files):
        print(f"\n--- Test {i+1}: {image_file.split('/')[-1]} ---")
        
        # Convert and load image
        png_path = raw_to_png(image_file, "temp_vision_test")
        if not png_path:
            print(f"❌ Failed to convert {image_file}")
            continue
            
        image = cv2.imread(png_path)
        print(f"Image loaded: {image.shape}")
        
        # Process with gap following
        result = vision_follower.process_image(
            image, 
            output_filename=f"vision_test_{i+1}.png"
        )
        
        # Show results
        steering_angle = result['steering_angle']
        debug = result['debug']
        
        if steering_angle is not None:
            print(f"✅ Steering: {degrees(steering_angle):.1f}°")
            print(f"   Gaps found: {debug['gaps_found']}")
            
            # Show gap details
            for j, gap in enumerate(debug['gaps']):
                start_x, end_x, center_x, width = gap
                print(f"   Gap {j+1}: center={center_x}px, width={width}px")
        else:
            print(f"❌ No steering solution found")
            print(f"   Gaps detected: {debug['gaps_found']}")
    
    print(f"\n✅ Test complete! Visualizations saved to images/vision/")

if __name__ == "__main__":
    test_vision_algorithm()