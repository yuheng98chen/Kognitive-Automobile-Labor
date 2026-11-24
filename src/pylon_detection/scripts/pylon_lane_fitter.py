#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pylon_lane_fitter.py ── 外侧 pylon 连折线 + 滤波，生成中心线
"""

from __future__ import annotations
import rospy, numpy as np
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
from pylon_detection.msg import PylonList

# ───────── 参数 ────────────────────────────────────────
MIN_POINTS_FIT = 4
SAMPLE_STEP    = 0.10
SG_WIN, SG_DEG = 15, 3
OFFSET         = 0.40  # 沿右法向偏移中心线
# ──────────────────────────────────────────────────────

def resample(poly: np.ndarray, step=SAMPLE_STEP) -> np.ndarray:
    if len(poly) < 2:
        return poly.copy()
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s   = np.insert(np.cumsum(seg), 0, 0.)
    u   = s / s[-1]
    uu  = np.linspace(0., 1., max(int(s[-1]/step)+1, 2))
    fx  = interp1d(u, poly[:,0], kind='linear')
    fy  = interp1d(u, poly[:,1], kind='linear')
    return np.vstack((fx(uu), fy(uu))).T

def to_path(arr: np.ndarray, header) -> Path:
    msg = Path(); msg.header = header
    for x, y in arr:
        ps = PoseStamped(); ps.header = header
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        ps.pose.orientation = Quaternion(w=1.0)
        msg.poses.append(ps)
    return msg

def greedy_sort(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points.copy()
    used = np.zeros(len(points), dtype=bool)
    ordered = [0]
    used[0] = True
    for _ in range(1, len(points)):
        last = points[ordered[-1]]
        dists = np.linalg.norm(points - last, axis=1)
        dists[used] = np.inf
        next_idx = np.argmin(dists)
        ordered.append(next_idx)
        used[next_idx] = True
    return points[ordered]

class LaneFitter:
    def __init__(self):
        rospy.init_node("pylon_lane_fitter")
        self.start_time = rospy.Time.now()
        self.delay_sec = rospy.get_param("~startup_delay", 40.0)
        self.offset_y = rospy.get_param("~center_offset_y", -0.6)  # 新增：整体 y 方向偏移量

        self.pub_outer  = rospy.Publisher("/pylon_outer_path", Path, queue_size=1, latch=True)
        self.pub_center = rospy.Publisher("/pylon_center_path", Path, queue_size=1, latch=True)

        rospy.Subscriber("/pylon_detection/confirmed_pylons", PylonList, self._cb, queue_size=1)

        rospy.loginfo(f"Lane-fitter ready. Delaying processing for {self.delay_sec:.1f}s.")
        rospy.loginfo(f"Center line will offset downward by {self.offset_y:.2f} m in y direction.")
        rospy.spin()

    def _cb(self, msg: PylonList):
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        if elapsed < self.delay_sec:
            rospy.logdebug(f"Startup delay active: {self.delay_sec - elapsed:.1f}s remaining.")
            return

        pts = np.asarray([[p.x, p.y] for p in msg.points])
        if pts.shape[0] < MIN_POINTS_FIT:
            self._publish_empty(msg.header)
            return

        outer_rs, centre = self._process(pts)
        self.pub_outer.publish(to_path(outer_rs, msg.header))
        self.pub_center.publish(to_path(centre, msg.header))

    def _publish_empty(self, header):
        empty = to_path(np.empty((0,2)), header)
        self.pub_outer.publish(empty)
        self.pub_center.publish(empty)

    def _process(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # 1. PCA 主方向 & 横向方向
        pca = PCA(n_components=2).fit(pts)
        t = pca.components_[0]                # 主方向（前进方向）
        n = np.array([-t[1], t[0]])           # 正交方向（右法向）

        d = (pts - pca.mean_) @ n             # 横向分量
        if np.mean(d) > 0:
            outer_raw = pts[d > 0]            # 右边是外侧
        else:
            outer_raw = pts[d < 0]            # 左边是外侧

        if outer_raw.shape[0] < MIN_POINTS_FIT:
            return np.empty((0,2)), np.empty((0,2))

        # 2. 贪心排序
        outer_raw = greedy_sort(outer_raw)

        # 3. Savitzky–Golay 平滑
        if outer_raw.shape[0] >= SG_WIN:
            outer_raw[:,0] = savgol_filter(outer_raw[:,0], SG_WIN, SG_DEG)
            outer_raw[:,1] = savgol_filter(outer_raw[:,1], SG_WIN, SG_DEG)

        # 4. 重采样 + 第二次平滑
        outer_rs = resample(outer_raw)
        if outer_rs.shape[0] >= SG_WIN:
            outer_rs[:,0] = savgol_filter(outer_rs[:,0], SG_WIN, SG_DEG)
            outer_rs[:,1] = savgol_filter(outer_rs[:,1], SG_WIN, SG_DEG)

        # 5. 沿右法向偏移 + 全局 y 方向平移
        centre = []
        for i in range(len(outer_rs)):
            if 0 < i < len(outer_rs)-1:
                t_vec = outer_rs[i+1] - outer_rs[i-1]
            else:
                t_vec = outer_rs[min(i+1,len(outer_rs)-1)] - outer_rs[max(i-1,0)]
            if np.linalg.norm(t_vec) < 1e-6:
                n_vec = np.array([0., -1.])
            else:
                t_unit = t_vec / np.linalg.norm(t_vec)
                n_vec = np.array([-t_unit[1], t_unit[0]])

            pt_center = outer_rs[i] + OFFSET * n_vec
            pt_center[1] += self.offset_y  # ✅ y 方向偏移（负值代表“下移”）
            centre.append(pt_center)
        centre = np.asarray(centre)

        return outer_rs, centre

if __name__ == "__main__":
    try:
        LaneFitter()
    except rospy.ROSInterruptException:
        pass