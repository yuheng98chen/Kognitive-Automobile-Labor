#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import uuid
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Point, PointStamped
from kal_env_map.srv import AddStaticObject, AddStaticObjectRequest
from visualization_msgs.msg import Marker, MarkerArray
from pylon_detection.msg import PylonList

# -------------------------------------------------
PRELOAD_PYLONS_FOR_TESTING = False
# -------------------------------------------------


class Pylon:
    """单个路锥跟踪器（去掉 stddev 过滤版）"""

    def __init__(self, first: Point, buf: int) -> None:
        self.id = str(uuid.uuid4())
        self.buf: Deque[Tuple[float, float]] = deque(maxlen=buf)
        self.buf.append((first.x, first.y))

        self.cur = np.array([first.x, first.y])   # 当前平均
        self.confirmed: Optional[np.ndarray] = None
        self.is_confirmed = False
        self.added_to_map = False
        self._buf = buf

    # —— 新观测加入 —— #
    def add(self, p: Point) -> bool:
        self.buf.append((p.x, p.y))
        self.cur = np.mean(np.array(self.buf), axis=0)

        # 缓冲满 → 直接确认
        if len(self.buf) == self._buf and not self.is_confirmed:
            self.confirmed = np.mean(np.array(self.buf), axis=0)
            self.is_confirmed = True
            rospy.loginfo(f"Pylon {self.id[:8]} confirmed @ {self.confirmed}")
            return True
        return False


class PylonManagerNode:
    """管理 / 去抖 / 写入地图"""

    def __init__(self) -> None:
        rospy.init_node("pylon_manager_node")

        # ── 参数 ──────────────────────────────────────────────
        self.topic          = rospy.get_param("~pylon_topic",
                                              "/yolo_detection/pylon_position")
        self.tol_radius     = float(rospy.get_param("~tolerance_radius", 0.25))
        self.buf_size       = int  (rospy.get_param("~buffer_size", 2))  # ← 改为 2
        self.map_frame      = rospy.get_param("~map_frame", "stargazer")
        self.srv_name       = rospy.get_param("~add_object_service",
                                              "add_static_object")
        self.debug          = rospy.get_param("~enable_debug_logging", True)

        # ── 状态 ──────────────────────────────────────────────
        self.pylons: List[Pylon] = []

        # ── 发布已确认列表（latched） ──────────────────────────
        self.pub_list = rospy.Publisher("/pylon_detection/confirmed_pylons",
                                        PylonList, queue_size=1, latch=True)
        self._publish_list()   # 清空上轮缓存

        # ── ROS I/O ──────────────────────────────────────────
        rospy.Subscriber(self.topic, PointStamped, self._cb)
        self.marker_pub = rospy.Publisher("~debug/pylon_markers",
                                          MarkerArray, queue_size=1)

        rospy.wait_for_service(self.srv_name)
        self.cli = rospy.ServiceProxy(self.srv_name, AddStaticObject)

        if PRELOAD_PYLONS_FOR_TESTING:
            self._preload_static_pylons()
            self._publish_list()
            self._publish_markers()

        if self.debug:
            rospy.Timer(rospy.Duration(5), self._log)

        rospy.loginfo("Pylon manager node ready.")

    # ────────── 回调 ──────────
    def _cb(self, msg: PointStamped) -> None:
        if msg.header.frame_id != self.map_frame:
            rospy.logwarn_throttle(10,
                f"Pylon in '{msg.header.frame_id}', expected '{self.map_frame}'")
            return

        det = np.array([msg.point.x, msg.point.y])
        nearest, dist = self._closest(det)

        if nearest and dist < self.tol_radius:
            self._associate(nearest, msg.point, dist)
        else:
            if nearest:
                rospy.logwarn(
                    f"Create new pylon ({det[0]:.2f},{det[1]:.2f}); "
                    f"closest {dist:.2f} m > tol {self.tol_radius}")
            self._create(msg.point)

        self._publish_markers()

    # ────────── 关联 / 新建 ──────────
    def _closest(self, p: np.ndarray) -> Tuple[Optional[Pylon], float]:
        if not self.pylons:
            return None, float("inf")
        dists = [np.linalg.norm(p - (py.confirmed if py.is_confirmed else py.cur))
                 for py in self.pylons]
        idx = int(np.argmin(dists))
        return self.pylons[idx], dists[idx]

    def _associate(self, py: Pylon, point: Point, dist: float) -> None:
        rospy.logdebug(f"Associate pylon {py.id[:8]} dist {dist:.2f}")
        if py.add(point):                 # 缓冲满即 confirmed
            self._add_to_map(py)
            self._publish_list()

    def _create(self, point: Point) -> None:
        rospy.loginfo("New pylon tracker created.")
        self.pylons.append(Pylon(point, self.buf_size))
        self._publish_list()

    # ────────── 写入地图 ──────────
    def _add_to_map(self, py: Pylon) -> None:
        if py.added_to_map or py.confirmed is None:
            return
        try:
            req = AddStaticObjectRequest(layer="pylon",
                                         x=float(py.confirmed[0]),
                                         y=float(py.confirmed[1]))
            rospy.wait_for_service(self.srv_name, timeout=0.5)
            res = self.cli(req)
            if res.accepted:
                rospy.loginfo(f"Pylon {py.id[:8]} added to env map.")
            else:
                rospy.loginfo(f"Pylon {py.id[:8]} rejected by env map (dup?).")
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn(f"Add-static service failed: {e}")
        py.added_to_map = True

    # ────────── RViz Markers ──────────
    def _publish_markers(self) -> None:
        ma = MarkerArray()
        for i, py in enumerate(self.pylons):
            ma.markers.append(self._marker(py, i))
        self.marker_pub.publish(ma)

    def _marker(self, py: Pylon, mid: int) -> Marker:
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = rospy.Time.now()
        m.ns = "pylons"; m.id = mid
        m.type = Marker.CYLINDER; m.action = Marker.ADD

        pos = py.confirmed if py.is_confirmed else py.cur
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(pos[0]), float(pos[1]), 0.4
        m.pose.orientation.w = 1.0

        m.scale.x = m.scale.y = 0.5; m.scale.z = 0.8
        if py.is_confirmed: m.color.g = 1.0  # cyan → confirmed
        else:               m.color.r, m.color.g, m.color.b = 1.0, 0.35, 0.0
        m.color.a = 1.0; m.lifetime = rospy.Duration(5)
        return m

    # ────────── 发布列表 ──────────
    def _publish_list(self):
        msg = PylonList()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_frame
        msg.points = [Point(x=float(py.confirmed[0]), y=float(py.confirmed[1]), z=0.0)
                      for py in self.pylons if py.is_confirmed]
        self.pub_list.publish(msg)

    # ────────── Debug 日志 ──────────
    def _log(self, _evt) -> None:
        if not self.pylons:
            rospy.loginfo("[Debug] no pylons tracked."); return
        lines = [f"\n--- Pylon Status ({rospy.Time.now().to_sec():.2f}) ---"]
        for i, py in enumerate(self.pylons):
            lines.append(f"[{i+1}/{len(self.pylons)}] id={py.id[:8]} "
                         f"buf={len(py.buf)}/{self.buf_size} "
                         f"avg=({py.cur[0]:.2f},{py.cur[1]:.2f}) "
                         f"confirmed={py.is_confirmed}")
        lines.append("-------------------------------------------")
        rospy.loginfo("\n".join(lines))

    # ────────── 预加载（可选） ──────────
    def _preload_static_pylons(self):
        rospy.loginfo("Pre-loading static pylons for testing…")
        static_pylons = [(11.7, 3.1), (12.2, 3.2),
                         (12.7, 3.4), (12.7, 2.6), (12.1, 2.4)]
        for x, y in static_pylons:
            pylon = Pylon(Point(x=x, y=y, z=0),
                          self.buf_size)
            pylon.confirmed = np.array([x, y])
            pylon.is_confirmed = True
            self.pylons.append(pylon)
            self._add_to_map(pylon)
        rospy.loginfo(f"Loaded {len(static_pylons)} static pylons.")


def main() -> None:
    try:
        PylonManagerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()