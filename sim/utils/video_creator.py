"""
Video Creation Utility

Compiles vision analysis images into video format for easy viewing and analysis.
Supports different frame rates and video formats.
"""

import cv2
import os
import glob
import re
from pathlib import Path
import argparse


class VisionVideoCreator:
    """Create videos from vision analysis images"""
    
    def __init__(self, output_dir="videos/vision"):
        """Initialize video creator"""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_sorted_images(self, image_dir, pattern="edges_frame_*.png"):
        """Get sorted list of image files"""
        image_pattern = os.path.join(image_dir, pattern)
        image_files = glob.glob(image_pattern)
        
        # Sort by frame number extracted from filename
        def extract_frame_number(filename):
            match = re.search(r'frame_(\d+)_', filename)
            return int(match.group(1)) if match else 0
        
        image_files.sort(key=extract_frame_number)
        return image_files
    
    def create_video(self, image_dir, output_filename, fps=10, codec='mp4v'):
        """
        Create video from images
        
        Args:
            image_dir (str): Directory containing images
            output_filename (str): Name for output video file
            fps (int): Frames per second for video
            codec (str): Video codec ('mp4v', 'XVID', etc.)
        """
        image_files = self.get_sorted_images(image_dir)
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return None
        
        print(f"Found {len(image_files)} images")
        print(f"Creating video with {fps} FPS...")
        
        # Read first image to get dimensions
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            print(f"Could not read first image: {image_files[0]}")
            return None
        
        height, width, channels = first_image.shape
        print(f"Video dimensions: {width}x{height}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_path = os.path.join(self.output_dir, output_filename)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print("Error: Could not open video writer")
            return None
        
        # Process images
        for i, image_file in enumerate(image_files):
            if i % 10 == 0:  # Progress update every 10 frames
                print(f"Processing frame {i+1}/{len(image_files)}")
            
            image = cv2.imread(image_file)
            if image is None:
                print(f"Warning: Could not read {image_file}")
                continue
            
            # Ensure image is correct size
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            
            video_writer.write(image)
        
        video_writer.release()
        print(f"Video saved: {output_path}")
        return output_path


def main():
    """Main video creation function"""
    parser = argparse.ArgumentParser(description='Create video from vision analysis images')
    parser.add_argument('--input-dir', type=str, default='images/vision', 
                       help='Directory containing images (default: images/vision)')
    parser.add_argument('--output', type=str, default='vision_analysis.mp4',
                       help='Output video filename (default: vision_analysis.mp4)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second (default: 10)')
    parser.add_argument('--codec', type=str, default='mp4v',
                       choices=['mp4v', 'XVID', 'MJPG'],
                       help='Video codec (default: mp4v)')
    
    args = parser.parse_args()
    
    print("Vision Video Creator")
    print("=" * 30)
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Create video creator
    video_creator = VisionVideoCreator()
    
    # Create video
    video_path = video_creator.create_video(
        image_dir=args.input_dir,
        output_filename=args.output,
        fps=args.fps,
        codec=args.codec
    )
    
    if video_path:
        print(f"\nVideo creation complete!")
        print(f"Output: {video_path}")
        print(f"Frames per second: {args.fps}")
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / args.fps
        cap.release()
        
        print(f"Total frames: {frame_count}")
        print(f"Duration: {duration:.2f} seconds")
    else:
        print("Video creation failed!")


if __name__ == "__main__":
    main()