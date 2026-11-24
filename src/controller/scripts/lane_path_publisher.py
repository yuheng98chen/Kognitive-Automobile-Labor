#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import os
import rospkg
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from typing import Optional

def generate_path_msg(centerline: np.ndarray, frame_id: str = "stargazer") -> Path:
    path = Path()
    path.header.frame_id = frame_id
    for x, y in centerline:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    return path

def arc_length(points: np.ndarray) -> np.ndarray:
    deltas = np.diff(points, axis=0)
    dist = np.linalg.norm(deltas, axis=1)
    return np.insert(np.cumsum(dist), 0, 0)

def resample_polyline(points: np.ndarray, target_num: int) -> np.ndarray:
    s = arc_length(points)
    s_uniform = np.linspace(0, s[-1], target_num)
    interp_x = interp1d(s, points[:, 0], kind='linear')
    interp_y = interp1d(s, points[:, 1], kind='linear')
    return np.vstack((interp_x(s_uniform), interp_y(s_uniform))).T

def generate_centerline_by_nn(left_lane: np.ndarray, right_lane: np.ndarray, num_points: int = 500) -> np.ndarray:
    left_resampled = resample_polyline(left_lane, num_points)
    tree = cKDTree(right_lane)
    _, idxs = tree.query(left_resampled)
    right_matched = right_lane[idxs]
    return 0.5 * (left_resampled + right_matched)

def load_lane_points() -> Optional[np.ndarray]:
    rospack = rospkg.RosPack()
    map_path = os.path.join(rospack.get_path("kal_env_map"), "pre_cached_maps")
    left_path = os.path.join(map_path, "interp_outer_lane_v1.npy")
    right_path = os.path.join(map_path, "interp_inner_lane_v3.npy")
    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        rospy.logerr("Lane .npy files not found.")
        return None
    left_lane = np.load(left_path)
    right_lane = np.load(right_path)
    return generate_centerline_by_nn(left_lane, right_lane, num_points=500)

def main():
    rospy.init_node("lane_path_publisher")
    pub = rospy.Publisher("/path", Path, queue_size=1, latch=True)
    rospy.sleep(1.0)

    centerline = load_lane_points()
    if centerline is None:
        rospy.signal_shutdown("Failed to load lane points.")
        return

    # —— close loop circle path ——  
    base_loop = np.vstack((centerline, centerline[0]))  
    num_repeats = rospy.get_param("~repeat_loops", 5)   
    centerline = np.vstack([base_loop for _ in range(num_repeats)])

    path_msg = generate_path_msg(centerline)
    pub.publish(path_msg)
    rospy.loginfo("Published /path with {} points ({} loops).".format(len(path_msg.poses), num_repeats))

    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass