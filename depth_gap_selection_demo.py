#!/usr/bin/env python3
"""
Depth-Enhanced Gap Selection Demo

This demonstrates how the VisionGapFollower algorithm will select gaps
when both RGB and depth data are available. It shows the difference between:
1. Width-based selection (current fallback)
2. Depth-enhanced selection (future capability)

This is especially important at corners where moving into deeper gaps 
(further from obstacles) is safer than just choosing the widest gap.
"""

import numpy as np
import cv2
from sim.algorithms.vision_gap_follow import VisionGapFollower
import matplotlib.pyplot as plt
from math import degrees

def create_mock_depth_scenario():
    """
    Create a mock scenario with multiple gaps and synthetic depth data
    to demonstrate depth-enhanced gap selection
    """
    # Create a simple scenario: hallway with two gaps
    # Left gap: wider but shallower (closer obstacles)
    # Right gap: narrower but deeper (further obstacles)
    
    # RGB image: 640x480 with black obstacles and white free space
    height, width = 480, 640
    rgb_image = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray background
    
    # Create free space areas (white)
    # Left gap: wide but with close obstacles
    rgb_image[200:400, 100:250] = [255, 255, 255]  # Left gap (150px wide)
    
    # Right gap: narrower but deeper
    rgb_image[200:400, 400:500] = [255, 255, 255]  # Right gap (100px wide)
    
    # Add some obstacles (black)
    rgb_image[200:400, 0:100] = [0, 0, 0]      # Left wall
    rgb_image[200:400, 250:400] = [0, 0, 0]    # Middle obstacle  
    rgb_image[200:400, 500:640] = [0, 0, 0]    # Right wall
    
    # Create corresponding depth image (millimeters)
    depth_image = np.full((height, width), 2000, dtype=np.uint16)  # Default 2m
    
    # Left gap: closer obstacles (1m depth)
    depth_image[200:400, 100:250] = 1000  # 1 meter
    
    # Right gap: further obstacles (3m depth) 
    depth_image[200:400, 400:500] = 3000  # 3 meters
    
    # Obstacles are very close
    depth_image[200:400, 0:100] = 200      # 20cm (wall)
    depth_image[200:400, 250:400] = 300    # 30cm (obstacle)
    depth_image[200:400, 500:640] = 200    # 20cm (wall)
    
    return rgb_image, depth_image

def demo_gap_selection():
    """Demonstrate gap selection with and without depth information"""
    
    print("Depth-Enhanced Gap Selection Demo")
    print("=" * 50)
    
    # Create mock scenario
    rgb_image, depth_image = create_mock_depth_scenario()
    
    # Initialize algorithm
    vision_follower = VisionGapFollower()
    
    print("\n1. RGB-Only Gap Selection (Current):")
    print("-" * 40)
    
    # Process without depth
    result_rgb_only = vision_follower.process_image(rgb_image)
    gaps_rgb = result_rgb_only['debug']['gaps']
    
    if gaps_rgb:
        print(f"   Found {len(gaps_rgb)} gaps:")
        for i, gap in enumerate(gaps_rgb):
            print(f"   Gap {i+1}: center={gap['center']:.0f}px, width={gap['width']}px")
        
        best_gap_rgb = max(gaps_rgb, key=lambda g: g['width'])
        print(f"   Selected gap: center={best_gap_rgb['center']:.0f}px (widest)")
        print(f"   Steering: {degrees(result_rgb_only['steering_angle']):.1f}°")
    
    print("\n2. Depth-Enhanced Gap Selection (Future):")
    print("-" * 40)
    
    # Process with depth
    result_with_depth = vision_follower.process_image(rgb_image, depth_image)
    gaps_depth = result_with_depth['debug']['gaps']
    
    if gaps_depth:
        print(f"   Found {len(gaps_depth)} gaps:")
        for i, gap in enumerate(gaps_depth):
            depth_str = f", depth={gap['median_depth']:.0f}mm" if gap['median_depth'] else ""
            print(f"   Gap {i+1}: center={gap['center']:.0f}px, width={gap['width']}px{depth_str}")
        
        # Show scoring logic
        print(f"   Depth available: {result_with_depth['debug']['depth_available']}")
        print(f"   Steering: {degrees(result_with_depth['steering_angle']):.1f}°")
    
    # Create visualization
    create_visualization(rgb_image, depth_image, gaps_rgb, gaps_depth, 
                        result_rgb_only['steering_angle'], result_with_depth['steering_angle'])
    
    print("\n3. Corner Scenario Analysis:")
    print("-" * 40)
    print("   At corners, depth-enhanced selection helps by:")
    print("   • Preferring gaps with obstacles further away")
    print("   • Moving into free space earlier for safety")
    print("   • Balancing gap width with obstacle distance")
    print("   • Reducing collision risk in tight spaces")

def create_visualization(rgb_image, depth_image, gaps_rgb, gaps_depth, 
                        steering_rgb, steering_depth):
    """Create side-by-side visualization of gap selection"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # RGB image
    ax1.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    ax1.set_title('RGB Image')
    ax1.set_xlabel('Width-based selection (current)')
    
    # Depth image
    depth_colored = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
    ax2.imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
    ax2.set_title('Depth Image')
    ax2.set_xlabel('Depth-enhanced selection (future)')
    
    # Gap analysis - RGB only
    if gaps_rgb:
        widths = [gap['width'] for gap in gaps_rgb]
        centers = [gap['center'] for gap in gaps_rgb]
        colors = ['red' if gap['width'] == max(widths) else 'blue' for gap in gaps_rgb]
        ax3.bar(range(len(gaps_rgb)), widths, color=colors)
        ax3.set_title(f'RGB-Only: Largest Gap Selected\nSteering: {degrees(steering_rgb):.1f}°')
        ax3.set_xlabel('Gap Index')
        ax3.set_ylabel('Width (pixels)')
    
    # Gap analysis - with depth
    if gaps_depth:
        # Create combined score visualization
        gap_indices = range(len(gaps_depth))
        widths = [gap['width'] for gap in gaps_depth]
        depths = [gap['median_depth'] if gap['median_depth'] else 0 for gap in gaps_depth]
        
        # Normalize for visualization
        max_width = max(widths) if widths else 1
        max_depth = max(depths) if depths else 1
        
        width_scores = [w/max_width * 50 for w in widths]
        depth_scores = [d/max_depth * 50 for d in depths]
        
        x = np.arange(len(gaps_depth))
        width_bars = ax4.bar(x - 0.2, width_scores, 0.4, label='Width Score', alpha=0.7)
        depth_bars = ax4.bar(x + 0.2, depth_scores, 0.4, label='Depth Score', alpha=0.7)
        
        ax4.set_title(f'Depth-Enhanced: Balanced Selection\nSteering: {degrees(steering_depth):.1f}°')
        ax4.set_xlabel('Gap Index')
        ax4.set_ylabel('Normalized Score')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('images/vision/depth_gap_selection_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] Visualization saved to images/vision/depth_gap_selection_demo.png")

if __name__ == '__main__':
    import os
    os.makedirs('images/vision', exist_ok=True)
    demo_gap_selection()