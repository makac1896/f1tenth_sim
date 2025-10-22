"""
Data Loader for Synchronized Lidar and Vision Data

Handles loading and synchronization of timestamped lidar JSON files
and vision image files for F1Tenth simulation and analysis.

Key Features:
- Timestamp-based synchronization between lidar and vision data
- Efficient data loading with configurable tolerance
- Support for different data formats (raw images, JSON metadata)
- Iterator interface for easy batch processing
"""
import json
import os
import glob
from datetime import datetime
import numpy as np
from PIL import Image


class F1TenthDataLoader:
    """
    Data loader for synchronized F1Tenth lidar and vision datasets
    """
    
    def __init__(self, lidar_dir, vision_dir, sync_tolerance_ms=100):
        """
        Initialize the data loader
        
        Args:
            lidar_dir (str): Directory containing lidar JSON files
            vision_dir (str): Directory containing vision image files
            sync_tolerance_ms (int): Maximum time difference for synchronization (milliseconds)
        """
        self.lidar_dir = lidar_dir
        self.vision_dir = vision_dir
        self.sync_tolerance_ms = sync_tolerance_ms
        
        # Load and index data files
        self.lidar_files = self._load_lidar_files()
        
        # Only load vision files if vision directory is provided
        if self.vision_dir:
            self.vision_files = self._load_vision_files()
            self.synchronized_pairs = self._find_synchronized_pairs()
            print(f"Loaded {len(self.lidar_files)} lidar files")
            print(f"Loaded {len(self.vision_files)} vision files") 
            print(f"Found {len(self.synchronized_pairs)} synchronized pairs")
        else:
            self.vision_files = {}
            self.synchronized_pairs = []
            print(f"Loaded {len(self.lidar_files)} lidar files")
    
    def _load_lidar_files(self):
        """
        Load and index lidar JSON files by timestamp
        
        Returns:
            dict: Mapping of timestamp to file path
        """
        lidar_files = {}
        pattern = os.path.join(self.lidar_dir, "scan_*.json")
        
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            # Extract timestamp from filename: scan_20251021T165247.653583Z.json
            timestamp_str = filename.replace("scan_", "").replace(".json", "")
            timestamp = self._parse_timestamp(timestamp_str)
            if timestamp:
                lidar_files[timestamp] = file_path
        
        return lidar_files
    
    def _load_vision_files(self):
        """
        Load and index vision image files by timestamp
        
        Returns:
            dict: Mapping of timestamp to file paths (image and metadata)
        """
        if not self.vision_dir:
            return {}
            
        vision_files = {}
        pattern = os.path.join(self.vision_dir, "image_*.raw")
        
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            # Extract timestamp from filename: image_20251021T165247.546201Z.raw
            timestamp_str = filename.replace("image_", "").replace(".raw", "")
            timestamp = self._parse_timestamp(timestamp_str)
            if timestamp:
                # Check for corresponding JSON metadata
                json_path = file_path + ".json"
                vision_files[timestamp] = {
                    'image_path': file_path,
                    'json_path': json_path if os.path.exists(json_path) else None
                }
        
        return vision_files
    
    def _parse_timestamp(self, timestamp_str):
        """
        Parse ISO timestamp string to datetime object
        
        Args:
            timestamp_str (str): ISO timestamp string
            
        Returns:
            datetime: Parsed timestamp or None if invalid
        """
        try:
            # Handle format: 20251021T165247.653583Z
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except ValueError:
            return None
    
    def _find_synchronized_pairs(self):
        """
        Find pairs of lidar and vision data within sync tolerance
        
        Returns:
            list: List of tuples (lidar_timestamp, vision_timestamp)
        """
        pairs = []
        tolerance_seconds = self.sync_tolerance_ms / 1000.0
        
        for lidar_ts in self.lidar_files.keys():
            best_vision_ts = None
            min_diff = float('inf')
            
            for vision_ts in self.vision_files.keys():
                diff = abs((lidar_ts - vision_ts).total_seconds())
                if diff <= tolerance_seconds and diff < min_diff:
                    min_diff = diff
                    best_vision_ts = vision_ts
            
            if best_vision_ts is not None:
                pairs.append((lidar_ts, best_vision_ts))
        
        # Sort by lidar timestamp
        pairs.sort(key=lambda x: x[0])
        return pairs
    
    def load_lidar_data(self, timestamp):
        """
        Load lidar data for a specific timestamp
        
        Args:
            timestamp (datetime): Timestamp of the lidar data
            
        Returns:
            dict: Lidar data with ranges, metadata, and drive_log
        """
        if timestamp not in self.lidar_files:
            return None
        
        file_path = self.lidar_files[timestamp]
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def load_vision_data(self, timestamp):
        """
        Load vision data for a specific timestamp
        
        Args:
            timestamp (datetime): Timestamp of the vision data
            
        Returns:
            dict: Vision data with image array and metadata
        """
        if timestamp not in self.vision_files:
            return None
        
        file_info = self.vision_files[timestamp]
        
        # Load raw image data
        image_array = self._load_raw_image(file_info['image_path'])
        
        # Load metadata if available
        metadata = None
        if file_info['json_path']:
            with open(file_info['json_path'], 'r') as f:
                metadata = json.load(f)
        
        return {
            'image': image_array,
            'metadata': metadata,
            'timestamp': timestamp
        }
    
    def _load_raw_image(self, image_path):
        """
        Load raw image file and convert to numpy array
        
        Args:
            image_path (str): Path to raw image file
            
        Returns:
            np.ndarray: Image as numpy array (H, W, C)
        """
        try:
            # Try to load as raw binary data (assuming RGB format 640x480)
            with open(image_path, 'rb') as f:
                data = f.read()
            
            # Assume RGB format, 640x480 resolution (adjust if needed)
            expected_size = 640 * 480 * 3
            if len(data) == expected_size:
                return np.frombuffer(data, dtype=np.uint8).reshape((480, 640, 3))
            else:
                print(f"Warning: Unexpected image size {len(data)}, expected {expected_size}")
                return None
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def get_synchronized_pair(self, index):
        """
        Get a synchronized lidar-vision pair by index
        
        Args:
            index (int): Index of the synchronized pair
            
        Returns:
            dict: Contains lidar_data, vision_data, and timing info
        """
        if index >= len(self.synchronized_pairs):
            return None
        
        lidar_ts, vision_ts = self.synchronized_pairs[index]
        
        lidar_data = self.load_lidar_data(lidar_ts)
        vision_data = self.load_vision_data(vision_ts)
        
        time_diff_ms = abs((lidar_ts - vision_ts).total_seconds()) * 1000
        
        return {
            'lidar_data': lidar_data,
            'vision_data': vision_data,
            'lidar_timestamp': lidar_ts,
            'vision_timestamp': vision_ts,
            'time_difference_ms': time_diff_ms,
            'index': index
        }
    
    def __iter__(self):
        """Iterator interface for easy batch processing"""
        self.current_index = 0
        return self
    
    def __next__(self):
        """Get next synchronized pair"""
        if self.current_index >= len(self.synchronized_pairs):
            raise StopIteration
        
        pair = self.get_synchronized_pair(self.current_index)
        self.current_index += 1
        return pair
    
    def __len__(self):
        """Get number of synchronized pairs or lidar files if no vision data"""
        if self.vision_dir and self.synchronized_pairs:
            return len(self.synchronized_pairs)
        else:
            return len(self.lidar_files)
    
    def get_stats(self):
        """
        Get statistics about the loaded dataset
        
        Returns:
            dict: Dataset statistics
        """
        if not self.synchronized_pairs:
            return {'error': 'No synchronized pairs found'}
        
        time_diffs = []
        for lidar_ts, vision_ts in self.synchronized_pairs:
            diff_ms = abs((lidar_ts - vision_ts).total_seconds()) * 1000
            time_diffs.append(diff_ms)
        
        return {
            'total_lidar_files': len(self.lidar_files),
            'total_vision_files': len(self.vision_files),
            'synchronized_pairs': len(self.synchronized_pairs),
            'sync_rate': len(self.synchronized_pairs) / max(len(self.lidar_files), len(self.vision_files)),
            'avg_time_diff_ms': np.mean(time_diffs),
            'max_time_diff_ms': np.max(time_diffs),
            'min_time_diff_ms': np.min(time_diffs)
        }