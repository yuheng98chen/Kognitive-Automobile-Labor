#!/usr/bin/env python3
"""
Stop Sign Manager Node
=======================
Subscribes to 3D stop sign detections, manages them, and adds confirmed signs
to the semantic environment map.

Features:
- Subscribes to PointStamped messages for detected stop signs.
- Maintains a list of known stop signs.
- When a new detection arrives, it's associated with a nearby known sign if
  one exists within a given tolerance.
- Each known sign uses a FIFO buffer to store recent detections.
- When the buffer is full and the positions are stable (low standard deviation),
  the sign's position is updated to the mean of the buffer.
- The confirmed, stable position is sent to the `kal_env_map` node via the
  `add_static_object` service.
- Publishes RViz markers for visualization of both pending and confirmed signs.
"""
# autopep8:off
from __future__ import annotations

import uuid
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Point, PointStamped
from kal_env_map.srv import AddStaticObject, AddStaticObjectRequest
from visualization_msgs.msg import Marker, MarkerArray
#autopep8: on


class StopSign:
    """Represents a single detected stop sign and its state."""

    def __init__(
        self,
        first_detection: Point,
        buffer_size: int,
        update_thresh: float
    ) -> None:
        self.id: str = str(uuid.uuid4())
        self.buffer: Deque[Tuple[float, float]] = deque(maxlen=buffer_size)
        self.confirmed_position: Optional[np.ndarray] = None
        self.current_position: np.ndarray = np.array([0., 0.])
        self.is_confirmed: bool = False
        self.is_added_to_map: bool = False
        self._buffer_size: int = buffer_size
        self._update_thresh: float = update_thresh

        self.add_detection(first_detection)

    def add_detection(self, point: Point) -> bool:
        """
        Add a new detection to the buffer and check for position stability.

        Args:
            point: Detected stop sign position

        Returns:
            True if the position was updated due to stability, False otherwise
        """
        self.buffer.append((point.x, point.y))
        self._update_current_position()

        if len(self.buffer) == self._buffer_size:
            return self._check_and_update_position()
        return False

    def _update_current_position(self) -> None:
        """Update the current position to the mean of the buffer."""
        self.current_position = np.mean(np.array(self.buffer), axis=0)

    def _check_and_update_position(self) -> bool:
        """
        Check if buffered positions are stable and update confirmed position.

        Returns:
            True if position was updated, False otherwise
        """
        points = np.array(self.buffer)
        std_dev = np.std(points, axis=0)

        if np.linalg.norm(std_dev) < self._update_thresh:
            # Stable, update position to the mean
            self.confirmed_position = np.mean(points, axis=0)
            self.is_confirmed = True
            rospy.loginfo(
                f"Stop sign {self.id[:8]} confirmed/updated at {self.confirmed_position}")
            return True
        return False

    @property
    def latest_position(self) -> np.ndarray:
        """
        Get the current best-guess position of the stop sign (mean of buffer).
        """
        return self.current_position


class StopSignManagerNode:
    """Manages stop sign detections and adds them to the map."""

    def __init__(self) -> None:
        rospy.init_node("stop_sign_manager_node")

        # --- Parameters ---
        self.stop_sign_topic: str = rospy.get_param(
            "~stop_sign_topic", "/yolo_detection/stop_sign_position")
        self.tolerance_radius: float = float(
            rospy.get_param("~tolerance_radius", 0.8))  # meters
        self.buffer_size: int = int(rospy.get_param("~buffer_size", 10))
        self.update_thresh: float = float(
            rospy.get_param("~update_thresh_stddev", 0.5))  # meters
        self.map_frame: str = rospy.get_param("~map_frame", "stargazer")
        self.add_object_service: str = rospy.get_param(
            "~add_object_service", "add_static_object")
        self.enable_debug_logging: bool = rospy.get_param(
            "~enable_debug_logging", True)

        # --- State ---
        self.known_signs: List[StopSign] = []

        # --- ROS I/O ---
        self._setup_ros_interfaces()

        rospy.loginfo("Stop Sign Manager node ready.")

    def _setup_ros_interfaces(self) -> None:
        """Initialize ROS subscribers, publishers, and service clients."""
        rospy.Subscriber(
            self.stop_sign_topic, PointStamped, self._detection_callback)

        self.marker_pub = rospy.Publisher(
            "~debug/stop_sign_markers", MarkerArray, queue_size=1)

        rospy.loginfo("Waiting for 'add_static_object' service...")
        rospy.wait_for_service(self.add_object_service)
        self.add_static_object_client = rospy.ServiceProxy(
            self.add_object_service, AddStaticObject)

        if self.enable_debug_logging:
            rospy.Timer(rospy.Duration(5), self._debug_log_signs)

    def _debug_log_signs(self, event) -> None:
        """Periodically log the state of all known stop signs."""
        if not self.known_signs:
            rospy.loginfo("[Debug] No known stop signs being tracked.")
            return

        header = f"\n--- Stop Sign Status ({rospy.Time.now().to_sec():.2f}) ---"
        log_msg_parts = [header]

        for i, sign in enumerate(self.known_signs):
            log_msg_parts.append(f"  [Sign {i+1}/{len(self.known_signs)}]")
            log_msg_parts.append(f"    - ID: {sign.id[:8]}")
            log_msg_parts.append(f"    - Confirmed: {sign.is_confirmed}")
            log_msg_parts.append(
                f"    - Buffer Size: {len(sign.buffer)} / {self.buffer_size}")
            log_msg_parts.append(
                f"    - Avg. Position: ({sign.current_position[0]:.2f}, {sign.current_position[1]:.2f})")

            if sign.confirmed_position is not None:
                log_msg_parts.append(
                    f"    - Confirmed Position: ({sign.confirmed_position[0]:.2f}, {sign.confirmed_position[1]:.2f})")
            else:
                log_msg_parts.append(f"    - Confirmed Position: None")

        log_msg_parts.append(
            "-------------------------------------------------")
        rospy.loginfo("\n".join(log_msg_parts))

    def _detection_callback(self, msg: PointStamped) -> None:
        """
        Handle incoming stop sign detection messages.

        Args:
            msg: Stop sign detection message containing position
        """
        if not self._validate_frame(msg.header.frame_id):
            return

        detection_pos = np.array([msg.point.x, msg.point.y])
        closest_sign, min_dist = self._find_closest_sign(detection_pos)

        if closest_sign is not None and min_dist < self.tolerance_radius:
            self._associate_with_existing_sign(
                closest_sign, msg.point, min_dist)
        else:
            if closest_sign:
                # Log details about why a new sign is being created
                rospy.logwarn(
                    f"Creating new sign at ({detection_pos[0]:.2f}, {detection_pos[1]:.2f}). "
                    f"Closest existing sign is at "
                    f"({closest_sign.latest_position[0]:.2f}, {closest_sign.latest_position[1]:.2f}) "
                    f"with distance {min_dist:.2f}m > tolerance {self.tolerance_radius:.2f}m."
                )
            self._create_new_sign(msg.point)

        self._publish_markers()

    def _validate_frame(self, frame_id: str) -> bool:
        """
        Validate that the detection is in the expected frame.

        Args:
            frame_id: Frame ID from the detection message

        Returns:
            True if frame is valid, False otherwise
        """
        if frame_id != self.map_frame:
            rospy.logwarn_throttle(
                10, f"Received stop sign in frame '{frame_id}' but expected "
                f"'{self.map_frame}'. TF transform needed.")
            return False
        return True

    def _find_closest_sign(
        self, detection_pos: np.ndarray
    ) -> Tuple[Optional[StopSign], float]:
        """
        Find the closest known stop sign to the detection.

        Args:
            detection_pos: Position of the new detection

        Returns:
            Tuple of (closest_sign, distance) or (None, inf) if no signs exist.
        """
        if not self.known_signs:
            return None, float('inf')

        distances = [
            np.linalg.norm(
                detection_pos - (
                    sign.confirmed_position if sign.is_confirmed
                    else sign.latest_position
                )
            )
            for sign in self.known_signs
        ]
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        return self.known_signs[min_dist_idx], min_dist

    def _associate_with_existing_sign(
        self, sign: StopSign, point: Point, distance: float
    ) -> None:
        """
        Associate a new detection with an existing stop sign.

        Args:
            sign: Existing stop sign to associate with
            point: New detection point
            distance: Distance between detection and existing sign
        """
        rospy.logdebug(
            f"Associating with existing sign {sign.id[:8]} (dist: {distance:.2f}m)")

        position_updated = sign.add_detection(point)
        if position_updated:
            self._add_to_map(sign)

    def _create_new_sign(self, point: Point) -> None:
        """
        Create a new stop sign tracker for an unassociated detection.

        Args:
            point: Detection point for the new stop sign
        """
        rospy.loginfo(
            "New potential stop sign detected. Creating new tracker.")
        new_sign = StopSign(point, self.buffer_size, self.update_thresh)
        self.known_signs.append(new_sign)

    def _add_to_map(self, sign: StopSign) -> None:
        """Add a confirmed stop sign to the environment map via service call."""
        if sign.is_added_to_map or not sign.confirmed_position is not None:
            return

        try:
            req = AddStaticObjectRequest()
            req.layer = "stop_sign"
            req.x, req.y = sign.confirmed_position
            log_msg = f"stop sign {sign.id[:8]}"

            # Use a short timeout to prevent blocking if service is slow
            rospy.wait_for_service(self.add_object_service, timeout=0.5)
            res = self.add_static_object_client(req)

            if res.accepted:
                rospy.loginfo(
                    f"Successfully added/updated {log_msg} in env map.")
            else:
                rospy.loginfo(
                    f"Env map rejected {log_msg} (likely a duplicate).")

        except rospy.ServiceException as e:
            rospy.logwarn(f"Service call to add object failed: {e}")
        except rospy.ROSException as e:
            rospy.logwarn(f"Service call timed out: {e}")

        sign.is_added_to_map = True

    def _publish_markers(self) -> None:
        """Publish RViz markers for all known stop signs."""
        marker_array = MarkerArray()
        for i, sign in enumerate(self.known_signs):
            marker = self._create_marker(sign, i)
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def _create_marker(self, sign: StopSign, marker_id: int) -> Marker:
        """
        Create an RViz marker for a single stop sign.

        Args:
            sign: The stop sign to create a marker for
            marker_id: The unique ID for this marker

        Returns:
            An RViz marker message
        """
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "stop_signs"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        pos = sign.latest_position
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = 1.0  # Fixed height for visualization
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        if sign.is_confirmed:
            marker.color.r = 0.0
            marker.color.g = 1.0  # Green for confirmed
            marker.color.b = 0.0
        else:
            marker.color.r = 1.0  # Red for pending
            marker.color.g = 0.0
            marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(5)  # Marker stays for 5 seconds
        return marker


def main() -> None:
    """Initialize and run the Stop Sign Manager node."""
    try:
        StopSignManagerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
