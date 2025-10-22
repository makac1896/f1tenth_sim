"""
Quick Gap Analysis Runner

Convenience script to run common analysis scenarios with proper naming.
"""
import subprocess
import sys
from datetime import datetime


def run_analysis(frames, start=0, dataset="lidar_1", description=""):
    """Run gap analysis with descriptive output"""
    
    print(f"\n{'='*60}")
    print(f"Running Analysis: {description}")
    print(f"Dataset: {dataset}, Frames: {start} to {start+frames-1}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "gap_analysis.py", 
        "--frames", str(frames),
        "--start", str(start),
        "--lidar-dir", f"all_logs/lidar/{dataset}",
        "--vision-dir", f"all_logs/vision/vision_1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running analysis: {e}")
        return False


def main():
    """Run multiple analysis scenarios"""
    print("F1Tenth Gap Analysis - Quick Runner")
    print("==================================")
    
    scenarios = [
        (50, 0, "lidar_1", "Initial Track Section - First 50 frames"),
        (100, 100, "lidar_1", "Mid Track Section - Frames 100-199"), 
        (50, 500, "lidar_1", "Later Track Section - Frames 500-549"),
        (20, 1000, "lidar_1", "End Track Section - Frames 1000-1019")
    ]
    
    successful = 0
    for frames, start, dataset, description in scenarios:
        if run_analysis(frames, start, dataset, description):
            successful += 1
        else:
            print(f"Failed: {description}")
    
    print(f"\n{'='*60}")
    print(f"Analysis Complete: {successful}/{len(scenarios)} scenarios successful")
    print(f"Check analysis_output/ folder for results")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()