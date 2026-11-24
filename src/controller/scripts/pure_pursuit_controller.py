#!/usr/bin/env python3

import rospy
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from topic_tools.srv import MuxSelect
import tf
from typing import List, Optional

LOOKAHEAD_DISTANCE = 0.7  # meters
MAX_SPEED = 0.6  # m/s
STEERING_ANGLE_LIMIT = 0.4  # radians
OFFSET_RIGHT = 0.12  # 右偏0.1米


class PurePursuitController:
    def __init__(self):
        rospy.init_node('pure_pursuit_controller')

        self.path: List[PoseStamped] = []
        self.current_pose: Optional[PoseStamped] = None

        rospy.Subscriber('/path', Path, self.path_callback)
        rospy.Subscriber('/stargazer/camera_pose',
                         PoseStamped, self.pose_callback)

        self.cmd_pub = rospy.Publisher(
            '/anicar/mux/ackermann_cmd_mux/input/path_follow',
            AckermannDriveStamped,
            queue_size=1
        )

        self.select_mux_input(
            '/anicar/mux/ackermann_cmd_mux/input/path_follow')
        rospy.loginfo("Pure Pursuit Controller initialized.")
        rospy.spin()

    def path_callback(self, msg: Path):
        """
        Callback for the path topic.
        Args:
            msg (Path): The path message.
        """
        self.path = msg.poses
        rospy.loginfo("Path received: {} points".format(len(self.path)))

    def pose_callback(self, msg: PoseStamped):
        """
        Callback for the pose topic.
        Args:
            msg (PoseStamped): The pose message.
        """
        self.current_pose = msg
        if not self.path:
            return

        # Decide whether to use the forward or reversed path based on the current heading
        # 根据当前朝向决定路径正反
        adjusted_path = self.get_forward_or_reversed_path()

        # Find the lookahead point
        # 找前视点
        target = self.find_lookahead_point(adjusted_path)
        if target is None:
            rospy.logwarn("No valid lookahead point found.")
            return

        # Calculate the tangent direction of the path (estimated using path points before and after the target point)
        # 计算路径切线方向（用目标点前后路径点估计）
        target_idx = None
        for i, p in enumerate(adjusted_path):
            if p == target:
                target_idx = i
                break

        if target_idx is None:
            # If not found, use the target directly without offset
            # 找不到，直接用target不偏移
            offset_x = target.pose.position.x
            offset_y = target.pose.position.y
        else:
            # Calculate the path tangent direction
            # 计算路径切线方向
            if target_idx < len(adjusted_path) - 1:
                next_p = adjusted_path[target_idx + 1].pose.position
                curr_p = target.pose.position
                path_dx = next_p.x - curr_p.x
                path_dy = next_p.y - curr_p.y
            elif target_idx > 0:
                curr_p = target.pose.position
                prev_p = adjusted_path[target_idx - 1].pose.position
                path_dx = curr_p.x - prev_p.x
                path_dy = curr_p.y - prev_p.y
            else:
                # Default direction
                # 默认方向
                path_dx = 1.0
                path_dy = 0.0

            # Calculate the unit tangent vector
            # 计算单位切线向量
            length = math.hypot(path_dx, path_dy)
            if length == 0:
                rospy.logwarn("Zero length path segment, skipping offset.")
                offset_x = target.pose.position.x
                offset_y = target.pose.position.y
            else:
                path_dx /= length
                path_dy /= length

                # Right normal vector (tangent vector rotated 90 degrees clockwise)
                # 右侧法线向量（切线向量顺时针旋转90度）
                normal_x = path_dy
                normal_y = -path_dx

                # Offset to the right by 0.1 meters
                # 右偏0.1米
                offset_x = target.pose.position.x + normal_x * OFFSET_RIGHT
                offset_y = target.pose.position.y + normal_y * OFFSET_RIGHT

        # Calculate control command
        # 计算控制指令
        dx = offset_x - self.current_pose.pose.position.x
        dy = offset_y - self.current_pose.pose.position.y

        heading = self.get_adjusted_heading(self.current_pose.pose.orientation)
        angle_to_target = math.atan2(dy, dx)
        steering_angle = self.normalize_angle(angle_to_target - heading)
        steering_angle = max(
            min(steering_angle, STEERING_ANGLE_LIMIT), -STEERING_ANGLE_LIMIT)

        cmd = AckermannDriveStamped()
        cmd.drive.steering_angle = steering_angle
        cmd.drive.speed = MAX_SPEED
        self.cmd_pub.publish(cmd)

        rospy.loginfo("Offset Target: ({:.2f}, {:.2f}), Heading: {:.2f}, AngleToTarget: {:.2f}, Steering: {:.2f}".format(
            offset_x,
            offset_y,
            heading,
            angle_to_target,
            steering_angle
        ))

    def get_forward_or_reversed_path(self) -> List[PoseStamped]:
        """
        Determines whether to use the forward or reversed path based on the current heading.
        Returns:
            List[PoseStamped]: The adjusted path.
        """
        if not self.current_pose or not self.path or len(self.path) < 2:
            return self.path

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y

        min_idx = 0
        min_dist = float('inf')
        for i, pose in enumerate(self.path):
            px = pose.pose.position.x
            py = pose.pose.position.y
            dist = math.hypot(px - cx, py - cy)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        if min_idx < len(self.path) - 1:
            p1 = self.path[min_idx]
            p2 = self.path[min_idx + 1]
        elif min_idx > 0:
            p1 = self.path[min_idx - 1]
            p2 = self.path[min_idx]
        else:
            return self.path

        dx = p2.pose.position.x - p1.pose.position.x
        dy = p2.pose.position.y - p1.pose.position.y
        path_dir = math.atan2(dy, dx)

        car_yaw = self.get_adjusted_heading(self.current_pose.pose.orientation)
        angle_diff = self.normalize_angle(path_dir - car_yaw)

        if abs(angle_diff) > math.pi / 2:
            rospy.loginfo("Reversing path direction for better alignment.")
            return list(reversed(self.path))
        else:
            return self.path

    def find_lookahead_point(self, path: List[PoseStamped]) -> Optional[PoseStamped]:
        """
        Finds the lookahead point on the path.
        Args:
            path (List[PoseStamped]): The path to search.
        Returns:
            Optional[PoseStamped]: The lookahead point, or None if not found.
        """
        if not path or not self.current_pose:
            return None

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y

        min_idx = 0
        min_dist = float('inf')
        for i, pose in enumerate(path):
            px = pose.pose.position.x
            py = pose.pose.position.y
            dist = math.hypot(px - cx, py - cy)
            if dist < min_dist:
                min_idx = i
                min_dist = dist

        heading = self.get_adjusted_heading(self.current_pose.pose.orientation)

        for i in range(min_idx, len(path)):
            px = path[i].pose.position.x
            py = path[i].pose.position.y
            dist = math.hypot(px - cx, py - cy)
            angle = math.atan2(py - cy, px - cx)
            angle_diff = self.normalize_angle(angle - heading)
            if dist >= LOOKAHEAD_DISTANCE and abs(angle_diff) < math.pi / 2:
                return path[i]

        return path[-1] if path else None

    def get_adjusted_heading(self, orientation: Quaternion) -> float:
        """
        Gets the adjusted heading from the orientation.
        Args:
            orientation (Quaternion): The orientation quaternion.
        Returns:
            float: The adjusted heading in radians.
        """
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        yaw -= math.pi / 2
        return self.normalize_angle(yaw)

    def normalize_angle(self, angle: float) -> float:
        """
        Normalizes an angle to the range [-pi, pi].
        Args:
            angle (float): The angle to normalize.
        Returns:
            float: The normalized angle.
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def select_mux_input(self, input_topic: str):
        """
        Selects the input topic for the ackermann_cmd_mux.
        Args:
            input_topic (str): The input topic to select.
        """
        rospy.wait_for_service('/anicar/mux/ackermann_cmd_mux/select')
        try:
            select_srv = rospy.ServiceProxy(
                '/anicar/mux/ackermann_cmd_mux/select', MuxSelect)
            resp = select_srv(input_topic)
            if resp.success:
                rospy.loginfo("MUX switched to input: {}".format(input_topic))
            else:
                rospy.logwarn(
                    "MUX failed to switch to: {}".format(input_topic))
        except rospy.ServiceException as e:
            rospy.logerr("MUX select service call failed: {}".format(e))


if __name__ == '__main__':
    try:
        PurePursuitController()
    except rospy.ROSInterruptException:
        pass
