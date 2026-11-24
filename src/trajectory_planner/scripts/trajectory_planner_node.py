#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav_msgs.msg import Path
import tf.transformations
# For finding closest points efficiently
from scipy.spatial.distance import cdist
from rospkg import RosPack
from typing import Optional, Tuple, List


class TrajectoryPlanner:
    """
    This ROS node generates a short-term trajectory for the vehicle to follow.

    It works by identifying the right-hand lane boundary relative to the vehicle,
    extracting a segment of that boundary ahead of the car, and then shifting that
    segment to the left to create a target path down the center of the lane.
    """

    def __init__(self):
        rospy.init_node('trajectory_planner_node', anonymous=True)
        rospy.loginfo("Trajectory Planner Node Initialized")

        # --- Load Parameters ---
        self.map_frame_id = rospy.get_param('~map_frame_id', 'stargazer')
        self.trajectory_length_m = rospy.get_param(
            '~trajectory_length_m', 1.20)
        self.lane_offset_m = rospy.get_param('~lane_center_offset_m', 0.20)
        self.start_distance_ahead_m = rospy.get_param(
            '~start_distance_ahead_m', 0.10)
        # === 新增的横向修正参数，默认为0 (无修正) ===
        self.lateral_correction_m = rospy.get_param('~lateral_correction_m', 0.0)

        rospy.loginfo(
            f"Parameters: Traj Length: {self.trajectory_length_m}m, "
            f"Offset: {self.lane_offset_m}m, "
            f"Start Ahead: {self.start_distance_ahead_m}m, "
            f"Lateral Correction: {self.lateral_correction_m}m"  # 添加日志打印
        )

        # --- Load Map Data ---
        try:
            rp = RosPack()
            pkg_path_kal_env_map = rp.get_path('kal_env_map')
            inner_lane_path = pkg_path_kal_env_map + \
                '/pre_cached_maps/interp_inner_lane_v2.npy'
            outer_lane_path = pkg_path_kal_env_map + \
                '/pre_cached_maps/interp_outer_lane_v2.npy'
            self.inner_lane_pts = np.load(inner_lane_path)
            self.outer_lane_pts = np.load(outer_lane_path)
            rospy.loginfo(
                f"Successfully loaded lanes: Inner ({self.inner_lane_pts.shape[0]} pts), "
                f"Outer ({self.outer_lane_pts.shape[0]} pts)"
            )
        except Exception as e:
            rospy.logfatal(f"Error loading map files: {e}", exc_info=True)
            rospy.signal_shutdown("Failed to load map files.")
            return

        self.traj_pub = rospy.Publisher('/trajectory', Path, queue_size=1)
        self.pose_sub = rospy.Subscriber(
            '/stargazer/camera_pose', PoseStamped, self.pose_callback, queue_size=1)

    def find_closest_point_idx(self, target_point_map: np.ndarray, lane_points_map: np.ndarray) -> Optional[int]:
        """
        Finds the index of the point in a lane array that is closest to a target point.
        """
        if lane_points_map.shape[0] == 0:
            return None
        distances = cdist(target_point_map.reshape(1, 2),
                          lane_points_map[:, :2])
        return np.argmin(distances)

    def transform_point_to_vehicle_frame(self, point_map: np.ndarray, vehicle_pos_map: np.ndarray, vehicle_yaw_map: float) -> np.ndarray:
        """
        Transforms a 2D point from the map frame to the vehicle's local frame.
        """
        dx = point_map[0] - vehicle_pos_map[0]
        dy = point_map[1] - vehicle_pos_map[1]
        cos_yaw = np.cos(vehicle_yaw_map)
        sin_yaw = np.sin(vehicle_yaw_map)
        # The vehicle frame has +X Left, +Y Backwards.
        x_veh = dx * cos_yaw + dy * sin_yaw
        y_veh = -dx * sin_yaw + dy * cos_yaw
        return np.array([x_veh, y_veh])

    def transform_point_to_map_frame(self, point_vehicle: np.ndarray, vehicle_pos_map: np.ndarray, vehicle_yaw_map: float) -> np.ndarray:
        """
        Transforms a 2D point from the vehicle's local frame back to the map frame.
        """
        x_veh, y_veh = point_vehicle[0], point_vehicle[1]
        cos_yaw = np.cos(vehicle_yaw_map)
        sin_yaw = np.sin(vehicle_yaw_map)
        # Inverse rotation from vehicle frame to map frame.
        x_map = vehicle_pos_map[0] + x_veh * cos_yaw - y_veh * sin_yaw
        y_map = vehicle_pos_map[1] + x_veh * sin_yaw + y_veh * cos_yaw
        return np.array([x_map, y_map])

    def pose_callback(self, msg: PoseStamped):
        """
        The main callback executed upon receiving a new vehicle pose.
        """
        camera_pose = msg.pose
        vehicle_pos_map = np.array(
            [camera_pose.position.x, camera_pose.position.y])
        vehicle_z_map = camera_pose.position.z

        q = [camera_pose.orientation.x, camera_pose.orientation.y,
             camera_pose.orientation.z, camera_pose.orientation.w]
        if not any(q):
            q = [0, 0, 0, 1]
            rospy.logwarn_throttle(
                1.0, "Received an all-zero quaternion. Defaulting to (0,0,0,1).")

        _, _, vehicle_yaw_map = tf.transformations.euler_from_quaternion(q)

        # --- 1. Identify Right Boundary ---
        idx_closest_inner = self.find_closest_point_idx(
            vehicle_pos_map, self.inner_lane_pts)
        idx_closest_outer = self.find_closest_point_idx(
            vehicle_pos_map, self.outer_lane_pts)

        if idx_closest_inner is None or idx_closest_outer is None:
            rospy.logwarn_throttle(
                1.0, "Could not find a closest point on one of the lanes. Skipping.")
            return

        closest_inner_pt_map = self.inner_lane_pts[idx_closest_inner, :2]
        closest_outer_pt_map = self.outer_lane_pts[idx_closest_outer, :2]
        closest_inner_pt_veh = self.transform_point_to_vehicle_frame(
            closest_inner_pt_map, vehicle_pos_map, vehicle_yaw_map)
        closest_outer_pt_veh = self.transform_point_to_vehicle_frame(
            closest_outer_pt_map, vehicle_pos_map, vehicle_yaw_map)

        is_inner_right = closest_inner_pt_veh[0] < 0
        is_outer_right = closest_outer_pt_veh[0] < 0

        right_boundary_pts = None
        if is_inner_right and is_outer_right:
            right_boundary_pts = self.inner_lane_pts if closest_inner_pt_veh[
                0] > closest_outer_pt_veh[0] else self.outer_lane_pts
        elif is_inner_right:
            right_boundary_pts = self.inner_lane_pts
        elif is_outer_right:
            right_boundary_pts = self.outer_lane_pts
        else:
            rospy.logwarn_throttle(
                1.0, "Neither boundary is to the vehicle's right. Falling back to closest.")
            dist_inner = np.linalg.norm(
                vehicle_pos_map - closest_inner_pt_map)
            dist_outer = np.linalg.norm(
                vehicle_pos_map - closest_outer_pt_map)
            right_boundary_pts = self.inner_lane_pts if dist_inner <= dist_outer else self.outer_lane_pts

        if right_boundary_pts is None:
            rospy.logwarn_throttle(
                1.0, "Could not determine a right boundary. No trajectory published.")
            return

        # --- 2. Extract Trajectory Segment ---
        start_point_veh_frame = np.array([0, -self.start_distance_ahead_m])
        start_point_map_frame = self.transform_point_to_map_frame(
            start_point_veh_frame, vehicle_pos_map, vehicle_yaw_map)
        idx_boundary_start = self.find_closest_point_idx(
            start_point_map_frame, right_boundary_pts)

        segment_map_frame: List[np.ndarray] = []
        current_length = 0.0
        last_pt_map = right_boundary_pts[idx_boundary_start, :2]
        segment_map_frame.append(last_pt_map)

        num_boundary_pts = right_boundary_pts.shape[0]
        for i in range(1, num_boundary_pts):
            current_idx = (idx_boundary_start + i) % num_boundary_pts
            pt_map = right_boundary_pts[current_idx, :2]
            dist_to_last = np.linalg.norm(pt_map - last_pt_map)

            if dist_to_last < 1e-6:
                continue

            if current_length + dist_to_last >= self.trajectory_length_m:
                remaining_len = self.trajectory_length_m - current_length
                direction_vec = (pt_map - last_pt_map) / dist_to_last
                final_pt = last_pt_map + direction_vec * remaining_len
                segment_map_frame.append(final_pt)
                break

            segment_map_frame.append(pt_map)
            current_length += dist_to_last
            last_pt_map = pt_map
        else:
            rospy.logwarn_throttle(
                5.0, "The entire boundary is shorter than the desired trajectory length.")

        # --- 3. Shift Segment to Create Final Trajectory ---
        final_trajectory_map_frame: List[np.ndarray] = []
        for pt_map_on_boundary in segment_map_frame:
            pt_boundary_veh = self.transform_point_to_vehicle_frame(
                pt_map_on_boundary, vehicle_pos_map, vehicle_yaw_map)
            
            # === 关键修改：应用车道偏移和横向修正 ===
            total_offset = self.lane_offset_m + self.lateral_correction_m
            pt_target_veh = np.array(
                [pt_boundary_veh[0] + total_offset, pt_boundary_veh[1]])
            
            pt_target_map = self.transform_point_to_map_frame(
                pt_target_veh, vehicle_pos_map, vehicle_yaw_map)
            final_trajectory_map_frame.append(pt_target_map)

        # --- 4. Publish Trajectory ---
        if len(final_trajectory_map_frame) < 2:
            rospy.logwarn_throttle(
                1.0, "Final trajectory has less than 2 points. Suppressing publish.")
            return

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.map_frame_id

        for pt_map in final_trajectory_map_frame:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position = Point(
                pt_map[0], pt_map[1], vehicle_z_map)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.traj_pub.publish(path_msg)


if __name__ == '__main__':
    try:
        planner = TrajectoryPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Trajectory planner node interrupted.")
    except Exception as e:
        rospy.logfatal(
            f"Unhandled exception in TrajectoryPlanner: {e}", exc_info=True)
    finally:
        rospy.loginfo("Trajectory planner node shutting down.")