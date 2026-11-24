#!/usr/bin/env python3
"""
Real-time semantic map in the 'stargazer' world frame, using a pre-built map.

This node initializes a semantic map by loading pre-defined lane boundaries
from .npy files. It then listens for static objects (like pylons) to be added
at runtime.

Layers
------
road_edge_left    : uint8 raster  (polyline from .npy)
road_edge_right   : uint8 raster  (polyline from .npy)
stop_sign         : uint8 raster  (circles)
pylon             : uint8 raster  (circles)
"""

from __future__ import annotations
import cv2
import numpy as np
import rospy
import rospkg
from nav_msgs.msg import OccupancyGrid
from kal_env_map.srv import AddStaticObject, AddStaticObjectResponse
from scipy.spatial import cKDTree
from typing import Dict, Tuple, List, Optional
import os

# ---------------------  parameters -----------------------------------------
RES = 0.05           # 5 cm per pixel
X_MAX = 30.0
Y_MAX = 20.0
FRAME_ID = "stargazer"
PUB_RATE = 1.0         # Hz
Y_MIN = -1.0
# ---------------------------------------------------------------------------

W, H = int(X_MAX / RES), int(round((Y_MAX - Y_MIN) / RES))

# Allocate rasters for our map layers.
layers: Dict[str, np.ndarray] = {
    "road_edge_left":   np.zeros((H, W), np.uint8),
    "road_edge_right":  np.zeros((H, W), np.uint8),
    "stop_sign":        np.zeros((H, W), np.uint8),
    "pylon":            np.zeros((H, W), np.uint8),
}

# Use KD-Trees for efficient nearest-neighbor search to dedup static objects.
trees: Dict[str, cKDTree] = {
    "stop_sign": cKDTree(np.empty((0, 2))),
    "pylon":     cKDTree(np.empty((0, 2))),
}
DEDUP_RADIUS_PYLON = 0.18      # 18 cm between pylon centers
DEDUP_RADIUS_STOP_SIGN = 0.15  # 15 cm for stop signs

# ------------- helper: world <-> grid indexing -----------------------------


def world_to_grid(x: float, y: float) -> Optional[Tuple[int, int]]:
    """Converts world coordinates (meters) to grid indices."""
    u = int(round(x / RES))
    v = int(round((y - Y_MIN) / RES))
    if 0 <= u < W and 0 <= v < H:
        return u, v
    return None


def points_to_pixels(points: np.ndarray) -> List[Tuple[int, int]]:
    """Converts a list of world points to a list of pixel coordinates."""
    pix = []
    for point in points:
        uv = world_to_grid(point[0], point[1])
        if uv:
            pix.append(uv)
    return pix
# ---------------------------------------------------------------------------


class EnvMapNode:
    def __init__(self):
        rospy.init_node("env_map_node")
        self.pub = rospy.Publisher("semantic_map", OccupancyGrid, queue_size=1)
        self._setup_msg()
        self._load_and_draw_lanes()

        # Service for adding static objects like pylons and stop signs
        rospy.Service("add_static_object",
                      AddStaticObject, self.handle_add_static)

        self.timer = rospy.Timer(rospy.Duration(1.0/PUB_RATE), self.publish)
        rospy.loginfo("Environment map node initialized.")

    def _load_and_draw_lanes(self):
        """Loads lane data from .npy files and draws them onto the map."""
        try:
            rospack = rospkg.RosPack()
            map_builder_path = rospack.get_path('kal_env_map')
            scripts_path = os.path.join(map_builder_path, 'pre_cached_maps')

            # Load and draw outer refined lane (road_edge_left)
            left_lane_path = os.path.join(
                scripts_path, 'interp_outer_lane_v1.npy')
            left_lane_pts = np.load(left_lane_path)
            self._draw_polyline(points_to_pixels(
                left_lane_pts), "road_edge_left")
            rospy.loginfo(
                f"Loaded {len(left_lane_pts)} points for left road edge.")

            # Load and draw inner lane (road_edge_right)
            right_lane_path = os.path.join(
                scripts_path, 'interp_inner_lane_v2.npy')
            right_lane_pts = np.load(right_lane_path)
            self._draw_polyline(points_to_pixels(
                right_lane_pts), "road_edge_right")
            rospy.loginfo(
                f"Loaded {len(right_lane_pts)} points for right road edge.")

        except rospkg.ResourceNotFound:
            rospy.logerr(
                "Could not find package 'kal_map_builder'. Make sure it's in your ROS workspace.")
        except FileNotFoundError as e:
            rospy.logerr(f"Lane file not found: {e}")
        except Exception as e:
            rospy.logerr(f"Error loading lane data: {e}")

    def _draw_polyline(self, pts: List[Tuple[int, int]], layer: str):
        if not pts:
            return
        pts_np = np.array(pts, dtype=np.int32)
        cv2.polylines(layers[layer], [pts_np],
                      isClosed=False, color=255, thickness=1)

    # -------------------- service ----------------------------------------
    def handle_add_static(self, req: AddStaticObject):
        layer = req.layer
        if layer not in ("stop_sign", "pylon"):
            rospy.logwarn(
                f"Invalid layer '{layer}' requested for static object.")
            return AddStaticObjectResponse(False)

        uv = world_to_grid(req.x, req.y)
        if not uv:
            rospy.logwarn(
                f"Coordinates ({req.x}, {req.y}) are outside map bounds.")
            return AddStaticObjectResponse(False)

        # Check for duplicates
        tree = trees[layer]
        dedup_radius = DEDUP_RADIUS_PYLON if layer == "pylon" else DEDUP_RADIUS_STOP_SIGN
        if tree.n > 0 and tree.query((req.x, req.y))[0] <= dedup_radius:
            rospy.loginfo(
                f"Duplicate {layer} ignored at ({req.x:.2f}, {req.y:.2f}).")
            return AddStaticObjectResponse(False)

        # Add object to map and KD-Tree
        trees[layer] = cKDTree(
            np.vstack([tree.data, (req.x, req.y)]) if tree.n else np.array([[req.x, req.y]]))

        radius_m = 0.10  # 10 cm radius for all objects for now
        radius_px = int(round(radius_m / RES))
        cv2.circle(layers[layer], uv, radius_px,
                   255, -1)  # -1 for filled circle
        rospy.loginfo(f"Added {layer} at ({req.x:.2f}, {req.y:.2f}).")

        return AddStaticObjectResponse(True)

    # -------------------- publishing -------------------------------------
    def _setup_msg(self):
        self.grid_msg = OccupancyGrid()
        self.grid_msg.header.frame_id = FRAME_ID
        self.grid_msg.info.resolution = RES
        self.grid_msg.info.width = W
        self.grid_msg.info.height = H
        self.grid_msg.info.origin.position.x = 0.0
        self.grid_msg.info.origin.position.y = Y_MIN
        self.grid_msg.info.origin.orientation.w = 1.0

    def publish(self, _):
        combined = np.zeros((H, W), np.int8)

        layer_values = {
            "road_edge_left":   70,
            "road_edge_right":  70,
            "stop_sign":        100,
            "pylon":            100,
        }
        for name, value in layer_values.items():
            combined[layers[name] > 0] = value

        self.grid_msg.data = combined.flatten().tolist()
        self.grid_msg.header.stamp = rospy.Time.now()
        self.pub.publish(self.grid_msg)


# --------------------------- main ------------------------------------------
if __name__ == "__main__":
    try:
        EnvMapNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
