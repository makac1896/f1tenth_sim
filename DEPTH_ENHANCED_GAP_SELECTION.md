# Depth-Enhanced Gap Selection for Vision Gap Following

## Overview

The VisionGapFollower algorithm has been enhanced with depth-based gap selection capabilities. This is especially important for corner navigation where moving into deeper free space (further from obstacles) can be safer than just choosing the widest gap.

## Key Features Added

### 1. Enhanced Algorithm Interface

```python
# Before: RGB-only processing
result = vision_follower.process_image(rgb_image)

# After: Optional depth integration
result = vision_follower.process_image(rgb_image, depth_image=depth_data)
```

### 2. Depth Analysis Processing Stage

- **Input**: RGB image + optional depth image from Intel D435i
- **Processing**: Analyzes depth information within detected gaps
- **Output**: Enhanced gap information with depth metrics

### 3. Smart Gap Selection Logic

The algorithm now uses a **balanced scoring approach**:

#### Width-Based (Fallback - Current)

- Selects the widest navigable gap
- Works with RGB-only data
- Good for general navigation

#### Depth-Enhanced (Future Capability)

- **60% weight**: Gap width (navigability)
- **40% weight**: Gap depth (safety margin)
- **Penalty**: High depth variance (unstable readings)
- **Result**: Safer corner navigation

## Technical Implementation

### Gap Data Structure

```python
# Enhanced gap information
gap = {
    'start': 100,           # Start pixel
    'end': 200,             # End pixel
    'center': 150,          # Center pixel
    'width': 100,           # Width in pixels
    'median_depth': 2500,   # Median depth in mm
    'max_depth': 3000,      # Maximum depth in mm
    'min_depth': 2000,      # Minimum depth in mm
    'depth_variance': 1000  # Depth variation in mm
}
```

### Depth Sampling Strategy

- **Vertical sampling**: Middle third of ROI (avoids ceiling/floor)
- **Horizontal sampling**: Full gap width
- **Filtering**: Removes invalid depths (< 100mm or > 5000mm)
- **Aggregation**: Uses median depth to avoid outliers

### Scoring Algorithm

```python
def _select_best_gap_with_depth(self, gaps):
    for gap in gaps:
        width_score = gap['width'] / 50.0  # Normalize by min gap width
        depth_score = min(gap['median_depth'] / 1000.0, 5.0)  # Cap at 5m
        stability_penalty = gap['depth_variance'] / 1000.0

        # Balanced scoring: prefer both wide and deep gaps
        score = (0.6 * width_score + 0.4 * depth_score) - (0.1 * stability_penalty)

    return highest_scored_gap
```

## Benefits for Corner Navigation

### Problem Scenario

At tight corners, the widest gap might have:

- ✅ Good navigability (wide enough)
- ❌ Close obstacles (shallow depth)
- ❌ Higher collision risk

### Depth-Enhanced Solution

The algorithm now considers:

- ✅ **Gap width**: Ensures navigability
- ✅ **Obstacle distance**: Prefers deeper gaps
- ✅ **Safety margin**: More room for error
- ✅ **Early avoidance**: Moves away from obstacles sooner

## Current Status

### Working Today

- ✅ Enhanced algorithm architecture
- ✅ Depth analysis methods implemented
- ✅ Backwards compatibility with RGB-only data
- ✅ Comprehensive debugging visualization
- ✅ Processing stages visualization

### Future Integration (When Depth Data Available)

- 🔄 Intel D435i depth stream recording
- 🔄 Synchronized RGB+Depth data loading
- 🔄 Real-world corner navigation testing
- 🔄 Parameter tuning for optimal performance

## Usage Examples

### Current RGB-Only Mode

```python
vision_follower = VisionGapFollower()
result = vision_follower.process_image(rgb_image)
# Uses width-based gap selection (fallback)
```

### Future Depth-Enhanced Mode

```python
vision_follower = VisionGapFollower()
result = vision_follower.process_image(rgb_image, depth_image)
# Uses depth-enhanced gap selection when depth_image provided
```

### Debug Information

```python
gaps = result['debug']['gaps']
for gap in gaps:
    print(f"Gap: {gap['width']}px wide, {gap['median_depth']}mm deep")

print(f"Depth available: {result['debug']['depth_available']}")
```

## Testing & Visualization

### Processing Stages Test

```bash
# See all processing stages including column analysis
python test_vision_stages.py --vision_dir all_logs/vision/vision_1 --frames 3
```

### Gap Selection Demo

```bash
# Demonstrate depth vs width-based selection
python depth_gap_selection_demo.py
```

## Architecture Benefits

### Clean Separation

- **Algorithm**: Pure gap detection logic (no I/O)
- **Visualization**: Separate utilities for debugging
- **Testing**: Comprehensive stage-by-stage analysis

### Extensibility

- Easy to add new gap selection criteria
- Modular depth analysis can handle different sensors
- Flexible scoring system for different scenarios

### Robustness

- Graceful fallback when depth unavailable
- Input validation and error handling
- Stable depth filtering and aggregation

---

This depth-enhanced approach will significantly improve the F1Tenth car's ability to navigate tight corners safely while maintaining good performance on straightaways.
