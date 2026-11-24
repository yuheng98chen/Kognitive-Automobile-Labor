#!/usr/bin/env python3

import rospy
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion, PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from topic_tools.srv import MuxSelect
import tf
from typing import List, Optional
from enum import Enum

LOOKAHEAD_DISTANCE = 0.8
MAX_SPEED = 1.2
STEERING_ANGLE_LIMIT = 0.3
OFFSET_RIGHT = 0.08
STOP_DISTANCE = 1.0
STOP_DURATION = 3.0

class State(Enum):
    TRACKING = 1
    WAITING = 2

class PurePursuitWithStop:
    def __init__(self):
        rospy.init_node('pure_pursuit_with_stop')

        self.path: List[PoseStamped] = []
        self.current_pose: Optional[PoseStamped] = None
        self.stop_sign_pos: Optional[PointStamped] = None
        self.state = State.TRACKING
        self.stop_start_time: Optional[rospy.Time] = None
        self.handled_stop_signs = {}  # key: (x, y, z), value: 是否可触发

        rospy.Subscriber('/path', Path, self.path_callback)
        rospy.Subscriber('/stargazer/camera_pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/yolo_detection/stop_sign_position', PointStamped, self.stop_sign_callback)

        self.cmd_pub = rospy.Publisher(
            '/anicar/mux/ackermann_cmd_mux/input/path_follow',
            AckermannDriveStamped,
            queue_size=1
        )

        self.select_mux_input('/anicar/mux/ackermann_cmd_mux/input/path_follow')
        rospy.loginfo("Pure Pursuit with stop sign logic initialized.")
        rospy.spin()

    def path_callback(self, msg: Path):
        self.path = msg.poses

    def stop_sign_callback(self, msg: PointStamped):
        self.stop_sign_pos = msg

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg
        self.update_control()

    def update_control(self):
        if not self.current_pose:
            return

        # WAITING 状态：保持停车或恢复 tracking
        if self.state == State.WAITING:
            if (rospy.Time.now() - self.stop_start_time).to_sec() >= STOP_DURATION:
                rospy.loginfo("Stop duration complete, resuming tracking.")
                self.state = State.TRACKING
            else:
                self.publish_zero_velocity()
                return

        # 检测到停牌后，仅按距离触发停车
        if self.stop_sign_pos:
            sx = self.stop_sign_pos.point.x
            sy = self.stop_sign_pos.point.y
            sz = self.stop_sign_pos.point.z
            stop_key = (round(sx, 2), round(sy, 2), round(sz, 2))
            if stop_key not in self.handled_stop_signs:
                self.handled_stop_signs[stop_key] = True

            cx = self.current_pose.pose.position.x
            cy = self.current_pose.pose.position.y
            dx = sx - cx
            dy = sy - cy
            dist = math.hypot(dx, dy)

            # 当远离时重置触发
            if dist > STOP_DISTANCE + 1.2 and not self.handled_stop_signs[stop_key]:
                rospy.loginfo(f"Stop sign at {stop_key} cleared, can re-trigger.")
                self.handled_stop_signs[stop_key] = True

            # 仅按欧氏距离触发停车
            if dist <= STOP_DISTANCE and self.handled_stop_signs[stop_key]:
                rospy.loginfo(f"Stop sign at {stop_key} reached (d={dist:.2f}), stopping for {STOP_DURATION}s.")
                self.publish_zero_velocity()
                self.stop_start_time = rospy.Time.now()
                self.state = State.WAITING
                self.handled_stop_signs[stop_key] = False
                return

        # 未触发停车时继续纯追踪
        self.pure_pursuit_control()

    def publish_zero_velocity(self):
        cmd = AckermannDriveStamped()
        cmd.drive.steering_angle = 0.0
        cmd.drive.speed = 0.0
        self.cmd_pub.publish(cmd)

    def pure_pursuit_control(self):
        if not self.path:
            rospy.logwarn("No path to follow.")
            return

        adjusted_path = self.get_forward_or_reversed_path()
        target = self.find_lookahead_point(adjusted_path)
        if target is None:
            rospy.logwarn("No valid lookahead point found.")
            return

        idx = adjusted_path.index(target)
        if idx < len(adjusted_path) - 1:
            next_p = adjusted_path[idx + 1].pose.position
            curr_p = target.pose.position
            path_dx = next_p.x - curr_p.x
            path_dy = next_p.y - curr_p.y
        elif idx > 0:
            curr_p = target.pose.position
            prev_p = adjusted_path[idx - 1].pose.position
            path_dx = curr_p.x - prev_p.x
            path_dy = curr_p.y - prev_p.y
        else:
            path_dx, path_dy = 1.0, 0.0

        length = math.hypot(path_dx, path_dy)
        if length:
            path_dx /= length
            path_dy /= length
            normal_x = path_dy
            normal_y = -path_dx
            offset_x = target.pose.position.x + normal_x * OFFSET_RIGHT
            offset_y = target.pose.position.y + normal_y * OFFSET_RIGHT
        else:
            offset_x = target.pose.position.x
            offset_y = target.pose.position.y

        dx = offset_x - self.current_pose.pose.position.x
        dy = offset_y - self.current_pose.pose.position.y
        heading = self.get_adjusted_heading(self.current_pose.pose.orientation)
        angle_to_target = math.atan2(dy, dx)
        steering = self.normalize_angle(angle_to_target - heading)
        steering = max(min(steering, STEERING_ANGLE_LIMIT), -STEERING_ANGLE_LIMIT)

        cmd = AckermannDriveStamped()
        cmd.drive.steering_angle = steering
        cmd.drive.speed = MAX_SPEED
        self.cmd_pub.publish(cmd)

    def get_forward_or_reversed_path(self) -> List[PoseStamped]:
        if not self.current_pose or len(self.path) < 2:
            return self.path

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        min_idx, min_dist = 0, float('inf')
        for i, pose in enumerate(self.path):
            px = pose.pose.position.x
            py = pose.pose.position.y
            d = math.hypot(px - cx, py - cy)
            if d < min_dist:
                min_idx, min_dist = i, d

        if min_idx < len(self.path) - 1:
            p1, p2 = self.path[min_idx], self.path[min_idx+1]
        else:
            p1, p2 = self.path[min_idx-1], self.path[min_idx]

        dx = p2.pose.position.x - p1.pose.position.x
        dy = p2.pose.position.y - p1.pose.position.y
        path_dir = math.atan2(dy, dx)

        yaw = self.get_adjusted_heading(self.current_pose.pose.orientation)
        if abs(self.normalize_angle(path_dir - yaw)) > math.pi/2:
            return list(reversed(self.path))
        return self.path

    def find_lookahead_point(self, path: List[PoseStamped]) -> Optional[PoseStamped]:
        if not path or not self.current_pose:
            return None

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        min_idx, min_dist = 0, float('inf')
        for i, pose in enumerate(path):
            px = pose.pose.position.x
            py = pose.pose.position.y
            d = math.hypot(px - cx, py - cy)
            if d < min_dist:
                min_idx, min_dist = i, d

        heading = self.get_adjusted_heading(self.current_pose.pose.orientation)
        for pose in path[min_idx:]:
            px, py = pose.pose.position.x, pose.pose.position.y
            d = math.hypot(px - cx, py - cy)
            ang = math.atan2(py - cy, px - cx)
            if d >= LOOKAHEAD_DISTANCE and abs(self.normalize_angle(ang - heading)) < math.pi/2:
                return pose
        return path[-1]

    def get_adjusted_heading(self, orientation: Quaternion) -> float:
        q = (orientation.x, orientation.y, orientation.z, orientation.w)
        _, _, yaw = tf.transformations.euler_from_quaternion(q)
        yaw -= math.pi/2
        return self.normalize_angle(yaw)

    def normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2*math.pi
        while angle < -math.pi:
            angle += 2*math.pi
        return angle

    def select_mux_input(self, input_topic: str):
        rospy.wait_for_service('/anicar/mux/ackermann_cmd_mux/select')
        try:
            select_srv = rospy.ServiceProxy('/anicar/mux/ackermann_cmd_mux/select', MuxSelect)
            resp = select_srv(input_topic)
            if resp.success:
                rospy.loginfo(f"MUX switched to input: {input_topic}")
            else:
                rospy.logwarn(f"MUX failed to switch to: {input_topic}")
        except rospy.ServiceException as e:
            rospy.logerr(f"MUX select service call failed: {e}")

if __name__ == '__main__':
    try:
        PurePursuitWithStop()
    except rospy.ROSInterruptException:
        pass