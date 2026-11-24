#!/usr/bin/env python3
# autopep8:off
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rospy
import tf.transformations as tft
import message_filters
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from kal_yolo_detection.msg import NamedMaskArray
# %autopep8:on

# ══════════════════════════════════════════════════════════════════════════════
# Configuration container
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class NodeConfig:
    # ---- Topics --------------------------------------------------------------
    mask_topic:          str = "/yolo_detection/segmentation_mask"
    depth_topic:         str = "/camera_front/depth/processed"
    rgb_topic:           str = "/camera_front/color/processed"
    camera_info_topic:   str = "/camera_front/processed/camera_info"
    stargazer_topic:     str = "/stargazer/camera_pose"
    tf_topic:            str = "/tf"
    output_topic:        str = "/yolo_detection/stop_sign_position"

    # ---- Debug Publisher Topics ----------------------------------------------
    debug_topic:         str = "/yolo_detection/stop_sign_debug_image"
    debug_mask_topic:    str = "/yolo_detection/stop_sign_debug_mask"
    depth_debug_topic:   str = "/yolo_detection/stop_sign_depth_debug"

    # ---- Detection thresholds ------------------------------------------------
    target_key:                str = "stop_sign"
    min_mask_area:             int = 1_200          # px ==> is about 2 metres
    max_mask_area:             int = 3_000          # px ==> After this it's too close, and irrelevant
    max_invalid_depth_ratio:   float = 0.4          # inside mask
    max_distance_m:            float = 1.90         # reject > 2.1 m in camera frame
    max_initial_distance_m:    float = 2.30         # sanity check in world frame

    # ---- Appearance rejector -------------------------------------------------
    white_ratio_thresh:        float = 0.25         # > 25 % white  → reject
    white_sat_thresh:          int = 40
    white_val_thresh:          int = 200
    vertex_lo:                 int = 6              # Min vertices required
    vertex_hi:                 int = 10             # Max vertices to tolerate

    # ---- Misc ----------------------------------------------------------------
    camera_offset_y:           float = -0.21        # camera frame → robot base (is 21 cm to the front of the top-facing camera)
    min_depth_m:              float = 0.05
    csv_log_path:             str = "/tmp/stop_sign_detections.csv"
    log_to_csv:               bool = False           # Enable to create a logging file with data about depth detections.
    enable_debug_publishers:  bool = True           # turn off to slim production
    debug_slop:               float = 0.07          # sync slop [s]

    world_frame:              str = "stargazer"
    camera_frame_for_tf:      str = "camera_top"


# ══════════════════════════════════════════════════════════════════════════════
# Debug publisher container
# ══════════════════════════════════════════════════════════════════════════════
class DebugPublishers:

    def __init__(self, cfg: NodeConfig) -> None:
        if cfg.enable_debug_publishers:
            self.rgb = rospy.Publisher(
                cfg.debug_topic,       Image, queue_size=1)
            self.mask = rospy.Publisher(
                cfg.debug_mask_topic,  Image, queue_size=1)
            self.depth = rospy.Publisher(
                cfg.depth_debug_topic, Image, queue_size=1)
        else:
            self.rgb = self.mask = self.depth = None  # type: ignore

    def publish_rgb(self, msg: Image) -> None:
        if self.rgb:
            self.rgb.publish(msg)

    def publish_mask(self, msg: Image) -> None:
        if self.mask:
            self.mask.publish(msg)

    def publish_depth(self, msg: Image) -> None:
        if self.depth:
            self.depth.publish(msg)


# ══════════════════════════════════════════════════════════════════════════════
# Main node
# ══════════════════════════════════════════════════════════════════════════════
class StopSignSegmentationNode:
    def __init__(self) -> None:
        rospy.init_node("stop_sign_segmentation_node", anonymous=True)

        # ── 1.  Read parameters into config dataclass
        self.cfg = self._load_config()

        # ── 2.  Misc state -----------------------------------------------------
        self.bridge = CvBridge()
        self._intrinsics_ready: bool = False
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.camera_frame_id: str = ""
        self.latest_pose: Optional[PoseStamped] = None

        # ── 3.  Debug publishers ----------------------------------------------
        self.debug_pub = DebugPublishers(self.cfg)

        # ── 4.  Output publisher ----------------------------------------------
        self.point_pub = rospy.Publisher(self.cfg.output_topic,
                                         PointStamped, queue_size=1)

        # ── 5.  Logging --------------------------------------------------------
        self.csv_writer: Optional[csv.writer] = None
        self._setup_csv_logger()

        # ── 6.  Subscribers & synchroniser ------------------------------------
        self._setup_subscribers()

        rospy.on_shutdown(self._cleanup)
        rospy.loginfo("Stop-sign segmentation node ready.")

    # ────────────────────────────────────────────────────────────────────
    # Parameter handling
    # ────────────────────────────────────────────────────────────────────
    def _load_config(self) -> NodeConfig:
        """Populate NodeConfig from ROS parameters or defaults.

        Returns:
            NodeConfig: The populated configuration object with any
                ROS parameter overrides applied.
        """
        cfg = NodeConfig()
        for field_ in cfg.__dataclass_fields__.values():
            param_name = f"~{field_.name}"
            if rospy.has_param(param_name):
                setattr(cfg, field_.name, rospy.get_param(param_name))
        return cfg

    # ────────────────────────────────────────────────────────────────────
    # CSV logger
    # ────────────────────────────────────────────────────────────────────
    def _setup_csv_logger(self) -> None:
        """Create the CSV logger if enabled by configuration.

        Returns:
            None
        """
        if not self.cfg.log_to_csv:
            return
        os.makedirs(os.path.dirname(self.cfg.csv_log_path), exist_ok=True)
        self.csv_file = open(self.cfg.csv_log_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "cam_x", "cam_y", "mask_px",  # camera pose x,y; found mask px size
            "obj_x", "obj_y", "obj_z",   # global pose x,y,z
            "min_d", "max_d", "median_d"  # lowest, highest, and median depth values
        ])

    # ────────────────────────────────────────────────────────────────────
    # Subscriber / Sync setup
    # ────────────────────────────────────────────────────────────────────
    def _setup_subscribers(self) -> None:
        """Initialise ROS subscribers and the message synchroniser.

        Sets up subscriptions to camera intrinsics, pose (Stargazer or TF
        fallback) and the time-synchronised mask, depth and RGB topics.

        Returns:
            None
        """
        # -- Camera intrinsics
        rospy.Subscriber(self.cfg.camera_info_topic, CameraInfo,
                         self._camera_info_cb, queue_size=1)

        # -- Pose: try Stargazer; fall back to /tf --> really undesired
        try:
            rospy.wait_for_message(self.cfg.stargazer_topic,
                                   PoseStamped, timeout=2.0)
            rospy.Subscriber(self.cfg.stargazer_topic, PoseStamped,
                             self._pose_cb, queue_size=1)
        except rospy.ROSException:
            rospy.logwarn("Stargazer pose unavailable, using /tf fallback.")
            from tf.msg import tfMessage
            rospy.Subscriber(self.cfg.tf_topic, tfMessage,
                             self._tf_cb, queue_size=10)

        # -- Time synchronised inputs ----------------------------------
        mask_sub = message_filters.Subscriber(
            self.cfg.mask_topic,  NamedMaskArray, queue_size=1)
        depth_sub = message_filters.Subscriber(
            self.cfg.depth_topic, Image,          queue_size=1)
        rgb_sub = message_filters.Subscriber(
            self.cfg.rgb_topic,   Image,          queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer(
            [mask_sub, depth_sub, rgb_sub],
            queue_size=5,
            slop=self.cfg.debug_slop
        )
        ts.registerCallback(self._sync_cb)

    # ────────────────────────────────────────────────────────────────────
    # Callbacks: Camera intrinsics & pose
    # ────────────────────────────────────────────────────────────────────
    def _camera_info_cb(self, msg: CameraInfo) -> None:
        """Cache camera intrinsics from the first ``CameraInfo`` message.

        Args:
            msg (CameraInfo): Camera intrinsics message.

        Returns:
            None
        """
        if self._intrinsics_ready:
            return
        self.fx, self.fy = msg.K[0], msg.K[4]
        self.cx, self.cy = msg.K[2], msg.K[5]
        self.camera_frame_id = msg.header.frame_id
        self._intrinsics_ready = True
        rospy.loginfo("Camera intrinsics received.")

    def _pose_cb(self, msg: PoseStamped) -> None:
        """Cache the latest Stargazer pose.

        Args:
            msg (PoseStamped): Latest pose update.
        """
        self.latest_pose = msg

    def _tf_cb(self, msg) -> None:
        """Extract and cache the camera pose from a TF message.

        Args:
            msg: A ``tfMessage`` containing the camera transform.
        """
        for tr in msg.transforms:
            if (tr.header.frame_id == self.cfg.world_frame
                    and tr.child_frame_id == self.cfg.camera_frame_for_tf):
                pose = PoseStamped()
                pose.header = tr.header
                pose.pose.position = tr.transform.translation
                pose.pose.orientation = tr.transform.rotation
                self.latest_pose = pose
                return

    # ────────────────────────────────────────────────────────────────────
    # Main synchronised callback
    # ────────────────────────────────────────────────────────────────────
    def _sync_cb(self, mask_msg: NamedMaskArray,
                 depth_msg: Image,
                 rgb_msg: Image) -> None:
        """Main pipeline entry point; processes a synchronised set of messages.

        Args:
            mask_msg (NamedMaskArray): Segmentation masks for the frame.
            depth_msg (Image): Depth image aligned to the RGB frame.
            rgb_msg (Image): RGB image.

        Returns:
            None
        """

        if not self._intrinsics_ready or self.latest_pose is None:
            rospy.logwarn_throttle(5.0, "Waiting for CameraInfo / Pose …")
            return

        depth_m = self._depth_to_array(depth_msg)
        if depth_m is None:
            return

        mask_cv = self._largest_target_mask(mask_msg.masks, mask_msg.keys)
        if mask_cv is None:
            return

        mask_bool = mask_cv.astype(bool)
        area = mask_bool.sum()
        if not (self.cfg.min_mask_area <= area <= self.cfg.max_mask_area): # reject detections either too close or far away
            return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        if not self._looks_like_stop(mask_bool, rgb): # Makes sure sign doesn't have too many white pixels, and has the correct vertices amount.
            return

        centroid = self._centroid(mask_bool)

        pos_cam, d_mean, d_min, d_max, d_median = self._3d_from_depth(
            mask_bool, depth_m, centroid)  # takes position as mean of mask-centre while ignoring outliers.
        if pos_cam is None:
            return

        pos_world = self._camera_to_world(pos_cam)
        if pos_world is None:
            return

        # sanity-check with current camera pose (prevents bogus TF)
        cam_xyz = np.array([self.latest_pose.pose.position.x,
                            self.latest_pose.pose.position.y,
                            self.latest_pose.pose.position.z])
        if np.linalg.norm(pos_world - cam_xyz) > self.cfg.max_initial_distance_m: # Reject noisy detections, which would be beyond the distance.
            return

        # ── Publish final PointStamped
        self._publish_point(pos_world, mask_msg.header)

        # ── CSV log ----------------------------------------------------
        if self.csv_writer:
            self.csv_writer.writerow([
                f"{cam_xyz[0]:.2f}", f"{cam_xyz[1]:.2f}", area,
                f"{pos_world[0]:.2f}", f"{pos_world[1]:.2f}", f"{pos_world[2]:.2f}",
                f"{d_min:.2f}", f"{d_max:.2f}", f"{d_median:.2f}"
            ])

        # ── Debug imagery ---------------------------------------------
        if self.cfg.enable_debug_publishers:
            self._publish_debug_images(rgb, mask_bool, pos_world, centroid,
                                       depth_m, mask_msg.header)

    # ────────────────────────────────────────────────────────────────────
    # Core helpers
    # ────────────────────────────────────────────────────────────────────
    def _depth_to_array(self, msg: Image) -> Optional[np.ndarray]:
        """Convert a ROS depth ``Image`` to a NumPy float32 array in metres.

        Args:
            msg (Image): Depth image message (``16UC1`` or ``32FC1``).

        Returns:
            Optional[np.ndarray]: Depth array in metres, or ``None`` on error.
        """
        try:
            if msg.encoding == "16UC1": # Should always be default case in our setup.
                arr = np.frombuffer(msg.data, np.uint16).reshape(
                    msg.height, msg.width).astype(np.float32) / 1_000.0
            else:  # 32FC1
                arr = np.frombuffer(msg.data, np.float32).reshape(
                    msg.height, msg.width)
            return arr
        except Exception as exc:
            rospy.logerr(f"Depth conversion failed: {exc}")
            return None

    # ––– mask selection ––––––––––––––––––––––––––––––––––––––––––––––––––
    def _largest_target_mask(self,
                             masks: Sequence[Image],
                             keys: Sequence[str]) -> Optional[np.ndarray]:
        """Return the largest mask whose key matches the configured target.

        Args:
            masks (Sequence[Image]): Mask messages.
            keys (Sequence[str]): Corresponding class keys.

        Returns:
            Optional[np.ndarray]: Binary mask image or ``None`` if none match.
        """
        max_px = 0
        largest: Optional[np.ndarray] = None
        for key, msg in zip(keys, masks):
            if key != self.cfg.target_key and not key.startswith(f"{self.cfg.target_key}_"):
                continue
            m = self.bridge.imgmsg_to_cv2(msg, "mono8")
            # slightly faster than cv2.countNonZero
            px = int(np.count_nonzero(m))
            if px > max_px:
                max_px, largest = px, m
        return largest

    # ––– visual validation –––––––––––––––––––––––––––––––––––––––––––––––
    def _looks_like_stop(self, mask: np.ndarray, rgb: np.ndarray) -> bool:
        """Determine whether the candidate region resembles a stop sign.

        Args:
            mask (np.ndarray): Binary segmentation mask for the candidate region.
            rgb (np.ndarray): Corresponding BGR image frame.

        Returns:
            bool: True if the heuristics judge the region to be a stop sign,
                False otherwise.
        """
        # ─ White-bar detector (no-entry) --------------------------------
        y, x = np.where(mask)
        if y.size == 0:  # should not happen
            return False
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()

        roi_rgb = rgb[y0:y1+1, x0:x1+1]
        roi_mask = mask[y0:y1+1, x0:x1+1]

        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)

        white = (s < self.cfg.white_sat_thresh) & (
            v > self.cfg.white_val_thresh) & roi_mask
        if white.sum() / roi_mask.sum() > self.cfg.white_ratio_thresh:
            return False

        # ─ Octagon shape check ------------------------------------------
        contours, _ = cv2.findContours(roi_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        return self.cfg.vertex_lo <= len(approx) <= self.cfg.vertex_hi

    # ––– centroid –––––––––––––––––––––––––––––––––––––––––––––––––––––––
    @staticmethod
    def _centroid(mask: np.ndarray) -> Tuple[int, int]:
        """Compute the centroid of a binary mask.

        Args:
            mask (np.ndarray): Binary mask where non-zero pixels form the region.

        Returns:
            Tuple[int, int]: Pixel coordinates ``(u, v)`` of the centroid.
        """
        m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
        if m["m00"] == 0:
            y, x = np.nonzero(mask)
            return int(x.mean()), int(y.mean())
        return int(round(m["m10"] / m["m00"])), int(round(m["m01"] / m["m00"]))

    # ––– depth → 3-D ––––––––––––––––––––––––––––––––––––––––––––––––––––
    def _3d_from_depth(
        self, mask: np.ndarray, depth: np.ndarray,
        c: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], float, float, float, float]:
        """Compute a robust 3-D position for the target from the depth image.

        Args:
            mask (np.ndarray): Binary mask for the target region.
            depth (np.ndarray): Depth image in metres.
            c (Tuple[int, int]): Pixel coordinates ``(u, v)`` of the region centroid.

        Returns:
            Tuple[Optional[np.ndarray], float, float, float, float]: A tuple
            containing ``(xyz_cam, mean, d_min, d_max, d_median)`` where:
                * ``xyz_cam`` is the 3-D point in the camera optical frame, or
                  ``None`` if the depth is invalid.
                * ``mean`` is the mean depth of the inlier pixels.
                * ``d_min`` and ``d_max`` are the minimum and maximum depths of
                  the inlier set.
                * ``d_median`` is the median depth of the valid pixels before
                  outlier removal.
        """
        u, v = c
        h, w = depth.shape
        xs, xe = max(0, u-10), min(w, u+10)
        ys, ye = max(0, v-10), min(h, v+10)

        roi_depth = depth[ys:ye, xs:xe]
        roi_mask = mask[ys:ye, xs:xe]
        valid = np.isfinite(roi_depth) & (
            roi_depth > self.cfg.min_depth_m) & roi_mask
        vals = roi_depth[valid]
        if vals.size < 10:
            return None, 0.0, 0.0, 0.0, 0.0

        d_median = float(np.median(vals))

        q1, q3 = np.percentile(vals, (25, 75))
        iqr = q3 - q1
        vals = vals[(vals >= q1 - 1.5*iqr) & (vals <= q3 + 1.5*iqr)]
        if vals.size == 0:
            return None, 0.0, 0.0, 0.0, 0.0

        d = float(vals.mean())
        if d > self.cfg.max_distance_m:
            return None, 0.0, 0.0, 0.0, 0.0

        x = (u - self.cx) * d / self.fx
        y = (v - self.cy) * d / self.fy
        cam_pt = np.array([x, y, d])                      # optical frame

        return cam_pt, d, float(vals.min()), float(vals.max()), d_median

    # ––– camera → world –––––––––––––––––––––––––––––––––––––––––––––––––
    def _camera_to_world(self, cam_pt: np.ndarray) -> Optional[np.ndarray]:
        """Transform a 3-D point from the camera optical frame to the world frame.

        Args:
            cam_pt (np.ndarray): Point coordinates in the camera optical frame
                ``(x, y, z)`` with +Z forward, +X right, +Y down.

        Returns:
            Optional[np.ndarray]: The point expressed in the world frame, or
            ``None`` if a valid camera pose has not yet been received.
        """
        if self.latest_pose is None:
            return None

        # optical → camera ROS
        x_o, y_o, z_o = cam_pt
        cam_ros = np.array([-x_o, -z_o, -y_o])

        # rotation + translation
        q = self.latest_pose.pose.orientation
        rot = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        offset = np.array([0.0, self.cfg.camera_offset_y, 0.0])

        t = self.latest_pose.pose.position
        world = rot @ cam_ros + rot @ offset + np.array([t.x, t.y, t.z])
        return world

    # ––– Publishers ––––––––––––––––––––––––––––––––––––––––––––––––––––
    def _publish_point(self, xyz: np.ndarray, header) -> None:
        """Publish a ``PointStamped`` message in the world frame.

        Args:
            xyz (np.ndarray): Point coordinates ``(x, y, z)`` in metres.
            header: Original header for timestamp/frame context.
        """
        msg = PointStamped()
        msg.header = header
        msg.header.frame_id = self.cfg.world_frame
        msg.point.x, msg.point.y, msg.point.z = xyz
        self.point_pub.publish(msg)

    def _publish_debug_images(
        self, rgb: np.ndarray, mask: np.ndarray,
        xyz: np.ndarray, centroid: Tuple[int, int],
        depth: np.ndarray, header
    ) -> None:
        """Create and publish RGB, mask and depth debug images.

        Args:
            rgb (np.ndarray): Original BGR frame.
            mask (np.ndarray): Binary target mask.
            xyz (np.ndarray): Target position in the world frame.
            centroid (Tuple[int, int]): Pixel centroid of the target.
            depth (np.ndarray): Depth image in metres.
            header: Header used to timestamp the debug images.
        """
        # ─ RGB overlay -------------------------------------------------
        overlay = rgb.copy()
        overlay[mask] = (0, 0, 255)
        blend = cv2.addWeighted(rgb, 0.5, overlay, 0.5, 0)
        cv2.circle(blend, centroid, 4, (0, 255, 0), -1)
        cv2.putText(blend, f"({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})",
                    (centroid[0]+6, centroid[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        self.debug_pub.publish_rgb(self.bridge.cv2_to_imgmsg(blend, "bgr8"))
        self.debug_pub.publish_mask(self.bridge.cv2_to_imgmsg(
            mask.astype(np.uint8)*255, "mono8"))

        # ─ Depth ROI legend (reuse original helper for brevity)
        # A compact inline variant to avoid code duplication:
        y, x = np.where(mask)
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        roi_depth = depth[y0:y1+1, x0:x1+1]
        roi_mask = mask[y0:y1+1, x0:x1+1]
        valid = np.isfinite(roi_depth) & (
            roi_depth > self.cfg.min_depth_m) & roi_mask
        if not np.any(valid):
            return
        d_min, d_max = roi_depth[valid].min(), roi_depth[valid].max()
        norm = np.zeros_like(roi_depth, np.uint8)
        if d_max > d_min:
            norm[valid] = (255 * (roi_depth[valid]-d_min) /
                           (d_max-d_min)).astype(np.uint8)
        depth_col = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        depth_col[~roi_mask] = 0
        depth_msg = self.bridge.cv2_to_imgmsg(depth_col, "bgr8")
        depth_msg.header = header
        self.debug_pub.publish_depth(depth_msg)

    # ────────────────────────────────────────────────────────────────────
    # Cleanup
    # ────────────────────────────────────────────────────────────────────
    def _cleanup(self) -> None:
        """Close the CSV log file on node shutdown."""
        if self.csv_writer:
            rospy.loginfo(f"Closing CSV log @ {self.cfg.csv_log_path}")
            self.csv_file.close()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """ROS node entry point."""
    _ = StopSignSegmentationNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
