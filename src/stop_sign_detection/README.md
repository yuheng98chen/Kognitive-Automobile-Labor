# Stop Sign Detection Package

An optimized ROS node (`stop_sign_locator.py`) that accurately locates STOP signs in 3D space from segmentation masks and depth images, with intelligent false-positive rejection to distinguish STOP signs from similar objects like NO-ENTRY signs.

## Key Features

* **Multi-sensor fusion**: Synchronizes segmentation masks, depth images, and RGB data
* **False-positive rejection**: Computer vision checks to distinguish STOP signs from NO-ENTRY signs
* **Robust depth estimation**: Multiple fallback strategies for reliable 3D positioning  
* **Flexible pose sources**: Supports both Stargazer and TF-based camera pose input
* **Comprehensive debugging**: Optional visualization and debug mask publishing
* **World frame transformation**: Converts detections from camera frame to world coordinates

## Topics

### Subscribed Topics

* **`~/mask_topic`** (`kal_yolo_detection/NamedMaskArray`) - Segmentation masks from YOLO detection
* **`~/depth_topic`** (`sensor_msgs/Image`) - Aligned depth image (16UC1 in mm or float32 in m)  
* **`~/rgb_topic`** (`sensor_msgs/Image`) - RGB image for visual validation (BGR8)
* **`~/camera_info_topic`** (`sensor_msgs/CameraInfo`) - Camera calibration parameters
* **`~/stargazer_pose_topic`** (`geometry_msgs/PoseStamped`) - Primary camera pose source
* **`~/tf_topic`** (`tf/tfMessage`) - Fallback pose source via TF transforms

### Published Topics

* **`~/output_topic`** (`geometry_msgs/PointStamped`) - 3D stop sign position in world frame
* **`~/debug_topic`** (`sensor_msgs/Image`) - Debug visualization (optional)
* **`~/debug_mask_topic`** (`sensor_msgs/Image`) - Processed mask visualization (optional)

## Internal Operation

### 1. Multi-Stage Validation Pipeline

```
Segmentation Mask → Size Filter → Visual Validation → 3D Positioning → World Transform
```

### 2. False-Positive Rejection

* **White content analysis**: Rejects signs with >25% white pixels (NO-ENTRY signs)

* **Shape analysis**: Validates octagonal geometry (6-10 vertices) characteristic of STOP signs
* **HSV-based filtering**: Uses color space analysis for white bar detection

### 3. Robust 3D Positioning

* **Centroid calculation**: Uses OpenCV moments for accurate center detection

* **Depth validation**: Filters invalid/noisy depth readings (>5cm minimum)
* **Median filtering**: Uses median depth from mask region for noise robustness
* **Distance filtering**: Optional maximum detection range

### 4. Coordinate Transformation Chain

```
Pixel (u,v) → Camera Optical Frame → Camera ROS Frame → World Frame
```

* Projects 2D pixels to 3D using camera intrinsics

* Applies coordinate frame conversions (optical ↔ ROS conventions)
* Transforms to world coordinates using pose data (Stargazer or TF)

## Key Parameters

### Detection & Filtering

* `target_key`: Mask identifier (default: "stop_sign")
* `min_mask_area`: Minimum pixel area threshold (default: 200)
* `max_invalid_depth_ratio`: Tolerance for noisy depth data (default: 0.5)
* `max_distance_m`: Maximum detection range in meters (default: 0.0 = disabled)

### Hardware & Calibration  

* `camera_offset_y`: Camera offset from Stargazer frame (default: -0.21m)

### Debug & Visualization

* `publish_debug`: Enable debug image publishing (default: false)
* `publish_debug_mask`: Enable processed mask publishing (default: true)

## Usage

### Launch the Node

```bash
roslaunch stop_sign_detection stop_sign_detection.launch
```

### Enable Debug Mode

```bash
roslaunch stop_sign_detection stop_sign_detection.launch debug:=true
```

### Configuration

Parameters are configured via `config/stop_sign_segmentation.yaml` and can be overridden in launch files or via rosparam.

## Verification

Test the node compilation:

```bash
python3 -m py_compile scripts/stop_sign_locator.py
```

Monitor output during operation:

```bash
rostopic echo /yolo_detection/stop_sign_position
```

The node publishes `geometry_msgs/PointStamped` messages containing 3D world coordinates whenever a valid stop sign is detected and confirmed.
