# Trajectory Planner Node

## 1. Overview

This ROS node is responsible for generating a short-term, local trajectory for the vehicle to follow. It continuously calculates a smooth path that guides the vehicle down the center of the current lane, based on the vehicle's real-time position and pre-defined lane boundary maps.

## 2. Coordinate Systems

The system relies on a clear understanding of two key coordinate frames:

- **`stargazer` (Map Frame)**: This is the global, fixed reference frame. All pre-cached lane boundary data is defined in this `stargazer` frame. The final trajectory published by this node is also expressed in this frame.

- **Vehicle Frame**: This is the local coordinate frame attached to the moving vehicle, as defined by the `/stargazer/camera_pose` message. Its orientation is crucial for local planning:
  - **+X axis**: Points to the **LEFT** of the vehicle.
  - **+Y axis**: Points **BACKWARDS** from the vehicle's direction of travel.
  - **+Z axis**: Points **UPWARDS**.

    > **Note:** The forward direction for the vehicle is along its **negative Y-axis**.

## 3. Inputs

The node requires the following inputs to function:

1. **Vehicle Pose**:
    - **Topic**: `/stargazer/camera_pose`
    - **Type**: `geometry_msgs/PoseStamped`
    - **Description**: Provides the real-time 2D position (x, y) and orientation (yaw) of the vehicle within the `stargazer` map frame.

2. **Lane Boundary Maps**:
    - **Files**: The node automatically loads `interp_inner_lane.npy` and `interp_outer_lane.npy` from the `pre_cached_maps` directory within the `kal_env_map` package.
    - **Description**: These files contain dense NumPy arrays of `[x, y]` coordinates that define the inner and outer lane boundaries in the `stargazer` frame.

## 4. Output

The node produces a single primary output:

- **Topic**: `/trajectory`
- **Type**: `nav_msgs/Path`
- **Description**: A sequence of poses that represents the target path for the vehicle. The `header.frame_id` of this path is the `stargazer` map frame, and the poses within it define a world-referenced trajectory for the vehicle to follow.

## 5. Core Logic - How It Works

Upon receiving a new pose message, the node executes the following steps:

1. **Identify Right Boundary**: The node first determines which of the two lane boundaries (inner or outer) is currently to the vehicle's right.
    - It finds the closest point on each of the two boundaries to the vehicle.
    - It transforms these two points into the vehicle's local frame. A point is considered to be on the "right" if its local X-coordinate is negative.
    - **Selection Logic**:
        - If only one boundary is to the right, it is chosen.
        - If both boundaries are to the right, the one that is "less to the right" (i.e., its local X-coordinate is closer to zero) is chosen.
        - As a fallback, if neither boundary is to the right, the one that is geometrically closest to the vehicle is chosen.

2. **Generate Reference Segment**: Once the right-hand boundary is identified, the node generates a reference path segment from it.
    - It first projects a "virtual" start point a distance of `start_distance_ahead_m` forward from the vehicle's current position.
    - It finds the point on the chosen boundary that is closest to this virtual start point.
    - Starting from this boundary point, it walks along the boundary's vertices, collecting them until the total length of the segment reaches `trajectory_length_m`. The use of dense, interpolated maps ensures a smooth segment can always be found.
    - If the distance to the next vertex is greater than the remaining required length, it interpolates a final point to ensure the segment is exactly the desired length.

3. **Create Final Trajectory**: The reference segment is then used to create the final, drivable trajectory.
    - Each point in the reference segment is transformed into the vehicle's local frame.
    - The point is then shifted to the **left** (positive X direction in the vehicle frame) by the `lane_center_offset_m` distance.
    - This new, shifted point is transformed back into the global `stargazer` map frame.

4. **Publish**: The collection of these final, shifted points in the map frame is assembled into a `nav_msgs/Path` message and published on the `/trajectory` topic.

## 6. Configurable Parameters

The node's behavior can be tuned using parameters in the `config/trajectory_params.yaml` file:

- `trajectory_length_m`: The desired length of the published trajectory in meters.
- `start_distance_ahead_m`: How far in front of the vehicle (in meters) to begin the trajectory planning process. This helps in creating a more stable path.
- `lane_center_offset_m`: The lateral distance (in meters) to shift from the identified right-hand boundary to calculate the vehicle's target path.
- `map_frame_id`: The name of the global map frame (should be "stargazer").
