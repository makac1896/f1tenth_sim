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
    Data loader for synchronized F1Tenth lidar, vision, and depth datasets
    """
    
    def __init__(self, lidar_dir, vision_dir=None, depth_dir=None, sync_tolerance_ms=100):
        """
        Initialize the data loader
        
        Args:
            lidar_dir (str): Directory containing lidar JSON files
            vision_dir (str, optional): Directory containing vision image files
            depth_dir (str, optional): Directory containing depth image files
            sync_tolerance_ms (int): Maximum time difference for synchronization (milliseconds)
        """
        self.lidar_dir = lidar_dir
        self.vision_dir = vision_dir
        self.depth_dir = depth_dir
        self.sync_tolerance_ms = sync_tolerance_ms
        
        # Load and index data files
        self.lidar_files = self._load_lidar_files()
        
        # Load vision files if vision directory is provided
        if self.vision_dir:
            self.vision_files = self._load_vision_files()
        else:
            self.vision_files = {}
        
        # Load depth files if depth directory is provided
        if self.depth_dir:
            self.depth_files = self._load_depth_files()
        else:
            self.depth_files = {}
        
        # Find synchronized data across all available modalities
        if self.vision_dir or self.depth_dir:
            self.synchronized_pairs = self._find_synchronized_pairs()
            print(f"Loaded {len(self.lidar_files)} lidar files")
            if self.vision_dir:
                print(f"Loaded {len(self.vision_files)} vision files")
            if self.depth_dir:
                print(f"Loaded {len(self.depth_files)} depth files")
            print(f"Found {len(self.synchronized_pairs)} synchronized pairs")
        else:
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
    
    def _load_depth_files(self):
        """
        Load and index depth image files by timestamp
        
        Returns:
            dict: Mapping of timestamp to file paths (depth and metadata)
        """
        if not self.depth_dir:
            return {}
            
        depth_files = {}
        pattern = os.path.join(self.depth_dir, "depth_*.raw")
        
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            # Extract timestamp from filename: depth_20251023T174951.131439Z.raw
            timestamp_str = filename.replace("depth_", "").replace(".raw", "")
            timestamp = self._parse_timestamp(timestamp_str)
            if timestamp:
                # Check for corresponding JSON metadata
                json_path = file_path + ".json"
                depth_files[timestamp] = {
                    'depth_path': file_path,
                    'json_path': json_path if os.path.exists(json_path) else None
                }
        
        return depth_files
    
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
        Find synchronized triplets of lidar, vision, and depth data within tolerance
        
        Returns:
            list: List of tuples (lidar_timestamp, vision_timestamp, depth_timestamp)
                  Any timestamp can be None if that modality is not available
        """
        pairs = []
        tolerance_seconds = self.sync_tolerance_ms / 1000.0
        
        for lidar_ts in self.lidar_files.keys():
            best_vision_ts = None
            best_depth_ts = None
            
            # Find closest vision timestamp
            if self.vision_files:
                min_vision_diff = float('inf')
                for vision_ts in self.vision_files.keys():
                    diff = abs((lidar_ts - vision_ts).total_seconds())
                    if diff <= tolerance_seconds and diff < min_vision_diff:
                        min_vision_diff = diff
                        best_vision_ts = vision_ts
            
            # Find closest depth timestamp
            if self.depth_files:
                min_depth_diff = float('inf')
                for depth_ts in self.depth_files.keys():
                    diff = abs((lidar_ts - depth_ts).total_seconds())
                    if diff <= tolerance_seconds and diff < min_depth_diff:
                        min_depth_diff = diff
                        best_depth_ts = depth_ts
            
            # Only include if we have at least one companion modality
            if best_vision_ts is not None or best_depth_ts is not None:
                pairs.append((lidar_ts, best_vision_ts, best_depth_ts))
        
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
    
    def load_depth_data(self, timestamp):
        """
        Load depth data for a specific timestamp
        
        Args:
            timestamp (datetime): Timestamp of the depth data
            
        Returns:
            dict: Depth data with depth array and metadata
        """
        if timestamp not in self.depth_files:
            return None
        
        file_info = self.depth_files[timestamp]
        
        # Load raw depth data
        depth_array = self._load_raw_depth(file_info['depth_path'])
        
        # Load metadata if available
        metadata = None
        if file_info['json_path']:
            with open(file_info['json_path'], 'r') as f:
                metadata = json.load(f)
        
        return {
            'depth': depth_array,
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
    
    def _load_raw_depth(self, depth_path):
        """
        Load raw depth file and convert to numpy array
        
        Args:
            depth_path (str): Path to raw depth file
            
        Returns:
            np.ndarray: Depth as numpy array (H, W) in meters
        """
        try:
            # Load as raw binary data (assuming 16-bit depth format 640x480)
            with open(depth_path, 'rb') as f:
                data = f.read()
            
            # Assume 16-bit depth format, 640x480 resolution
            expected_size = 640 * 480 * 2  # 2 bytes per pixel for 16-bit
            if len(data) == expected_size:
                # Load as 16-bit unsigned integers and convert to meters
                depth_raw = np.frombuffer(data, dtype=np.uint16).reshape((480, 640))
                # Convert from millimeters to meters (assuming typical depth camera format)
                depth_meters = depth_raw.astype(np.float32) / 1000.0
                return depth_meters
            else:
                print(f"Warning: Unexpected depth size {len(data)}, expected {expected_size}")
                return None
        except Exception as e:
            print(f"Error loading depth {depth_path}: {e}")
            return None
    
    def get_synchronized_pair(self, index):
        """
        Get a synchronized lidar-vision-depth triplet by index
        
        Args:
            index (int): Index of the synchronized pair
            
        Returns:
            dict: Contains lidar_data, vision_data, depth_data, and timing info
        """
        if index >= len(self.synchronized_pairs):
            return None
        
        lidar_ts, vision_ts, depth_ts = self.synchronized_pairs[index]
        
        # Load available data
        lidar_data = self.load_lidar_data(lidar_ts)
        vision_data = self.load_vision_data(vision_ts) if vision_ts else None
        depth_data = self.load_depth_data(depth_ts) if depth_ts else None
        
        # Calculate time differences (use available timestamps)
        time_diffs = {}
        if vision_ts:
            time_diffs['lidar_vision_ms'] = abs((lidar_ts - vision_ts).total_seconds()) * 1000
        if depth_ts:
            time_diffs['lidar_depth_ms'] = abs((lidar_ts - depth_ts).total_seconds()) * 1000
        if vision_ts and depth_ts:
            time_diffs['vision_depth_ms'] = abs((vision_ts - depth_ts).total_seconds()) * 1000
        
        return {
            'lidar_data': lidar_data,
            'vision_data': vision_data,
            'depth_data': depth_data,
            'lidar_timestamp': lidar_ts,
            'vision_timestamp': vision_ts,
            'depth_timestamp': depth_ts,
            'time_differences': time_diffs,
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
        
        lidar_vision_diffs = []
        lidar_depth_diffs = []
        vision_depth_diffs = []
        
        for lidar_ts, vision_ts, depth_ts in self.synchronized_pairs:
            if vision_ts:
                diff_ms = abs((lidar_ts - vision_ts).total_seconds()) * 1000
                lidar_vision_diffs.append(diff_ms)
            if depth_ts:
                diff_ms = abs((lidar_ts - depth_ts).total_seconds()) * 1000
                lidar_depth_diffs.append(diff_ms)
            if vision_ts and depth_ts:
                diff_ms = abs((vision_ts - depth_ts).total_seconds()) * 1000
                vision_depth_diffs.append(diff_ms)
        
        stats = {
            'total_lidar_files': len(self.lidar_files),
            'total_vision_files': len(self.vision_files),
            'total_depth_files': len(self.depth_files),
            'synchronized_triplets': len(self.synchronized_pairs),
            'vision_available_count': len([p for p in self.synchronized_pairs if p[1] is not None]),
            'depth_available_count': len([p for p in self.synchronized_pairs if p[2] is not None]),
        }
        
        if lidar_vision_diffs:
            stats.update({
                'lidar_vision_avg_diff_ms': np.mean(lidar_vision_diffs),
                'lidar_vision_max_diff_ms': np.max(lidar_vision_diffs),
                'lidar_vision_min_diff_ms': np.min(lidar_vision_diffs)
            })
        
        if lidar_depth_diffs:
            stats.update({
                'lidar_depth_avg_diff_ms': np.mean(lidar_depth_diffs),
                'lidar_depth_max_diff_ms': np.max(lidar_depth_diffs),
                'lidar_depth_min_diff_ms': np.min(lidar_depth_diffs)
            })
        
        return stats