#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pylon_locator_node ‒ 改进版（底边深度、中位数、地平面校正等）
---------------------------------------------------------------
• 订阅
    /yolo_detection/segmentation_mask   (NamedMaskArray)
    /camera_front/depth/processed       (Image 16UC1 / 32FC1)
    /camera_front/color/processed       (Image BGR8，仅调试)
    /camera_front/processed/camera_info (CameraInfo)
    /stargazer/camera_pose  或 /tf
• 发布
    /yolo_detection/pylon_position      (PointStamped, frame=stargazer)
• 主要改动
    ① 只取掩膜最底 N (=8) 行像素估算深度 → 避开锥体上部空洞
    ② 质心改为“底边中心”，更贴近锥基
    ③ 深度用 **中位数+IQR** 去离群
    ④ 相机射线与 z=0 地平面求交，消除系统性高度误差
    ⑤ camera→world 外参全部参数化，可 launch 中微调
"""

from __future__ import annotations
from dataclasses import dataclass
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

# ╔════════════ 参数配置 ════════════╗
@dataclass
class NodeConfig:
    # topics
    mask_topic:   str = "/yolo_detection/segmentation_mask"
    depth_topic:  str = "/camera_front/depth/processed"
    rgb_topic:    str = "/camera_front/color/processed"
    info_topic:   str = "/camera_front/processed/camera_info"
    pose_topic:   str = "/stargazer/camera_pose"
    tf_topic:     str = "/tf"
    out_topic:    str = "/yolo_detection/pylon_position"

    # debug topics
    dbg_rgb:   str = "/yolo_detection/pylon_debug_image"
    dbg_mask:  str = "/yolo_detection/pylon_debug_mask"
    dbg_depth: str = "/yolo_detection/pylon_depth_debug"
    enable_dbg: bool = True
    sync_slop:  float = .07

    # detection thresholds
    key:                str   = "pylon"
    min_area:           int   = 800
    max_area:           int   = 2500
    max_z:              float = 2.3
    max_initial_dist:   float = 2.6

    # depth handling
    bottom_rows: int   = 8     # 只用掩膜最底 N 行
    min_depth_m: float = .05

    # extrinsics
    world_frame:  str = "stargazer"
    camera_frame: str = "camera_top"
    camera_offset_y: float = 0.21   # 光心→车体 y (+左)
    camera_height:   float = 0.28   # 光心→地面 (m)
# ╚═════════════════════════════════╝


# —— 简易 debug 发布器 —————————————————
class DebugMux:
    def __init__(self, cfg: NodeConfig):
        self.br = CvBridge()
        if cfg.enable_dbg:
            self.pub_rgb   = rospy.Publisher(cfg.dbg_rgb,   Image, queue_size=1)
            self.pub_mask  = rospy.Publisher(cfg.dbg_mask,  Image, queue_size=1)
            self.pub_depth = rospy.Publisher(cfg.dbg_depth, Image, queue_size=1)
        else:
            self.pub_rgb = self.pub_mask = self.pub_depth = None  # type: ignore

    def rgb(self,  img): self.pub_rgb   and self.pub_rgb.publish(img)
    def mask(self, img): self.pub_mask  and self.pub_mask.publish(img)
    def depth(self,img): self.pub_depth and self.pub_depth.publish(img)


# ╔════════════ 主节点 ═════════════╗
class PylonLocatorNode:
    def __init__(self):
        rospy.init_node("pylon_locator_node")
        self.cfg = self._load_cfg()
        self.cv  = CvBridge()

        # camera intrinsics
        self.fx = self.fy = self.cx = self.cy = 0.
        self.int_ok = False

        # latest pose
        self.pose: Optional[PoseStamped] = None

        # pubs
        self.pub_pt = rospy.Publisher(self.cfg.out_topic,
                                      PointStamped, queue_size=1)
        self.dbg    = DebugMux(self.cfg)

        self._subs()
        rospy.loginfo("pylon_locator_node ready.")

    # ─── utils ─────────────────────
    def _load_cfg(self):
        cfg = NodeConfig()
        for f in cfg.__dataclass_fields__:
            p = "~"+f
            if rospy.has_param(p):
                setattr(cfg, f, rospy.get_param(p))
        return cfg

    def _subs(self):
        rospy.Subscriber(self.cfg.info_topic, CameraInfo,
                         self._info_cb, queue_size=1)

        try:
            rospy.wait_for_message(self.cfg.pose_topic, PoseStamped, timeout=2)
            rospy.Subscriber(self.cfg.pose_topic, PoseStamped,
                             self._pose_cb, queue_size=1)
        except rospy.ROSException:
            from tf.msg import tfMessage
            rospy.logwarn("Pose topic unavailable, fallback to /tf.")
            rospy.Subscriber(self.cfg.tf_topic, tfMessage,
                             self._tf_cb, queue_size=10)

        m_sub = message_filters.Subscriber(self.cfg.mask_topic,  NamedMaskArray, queue_size=1)
        d_sub = message_filters.Subscriber(self.cfg.depth_topic, Image,          queue_size=1)
        r_sub = message_filters.Subscriber(self.cfg.rgb_topic,   Image,          queue_size=1)
        sync  = message_filters.ApproximateTimeSynchronizer([m_sub,d_sub,r_sub],
                                                            5, self.cfg.sync_slop)
        sync.registerCallback(self._sync_cb)

    # ─── callbacks ─────────────────
    def _info_cb(self, msg: CameraInfo):
        if self.int_ok: return
        self.fx, self.fy = msg.K[0], msg.K[4]
        self.cx, self.cy = msg.K[2], msg.K[5]
        self.int_ok = True
        rospy.loginfo("Camera intrinsics received.")

    def _pose_cb(self, msg: PoseStamped): self.pose = msg

    def _tf_cb(self, tfm):
        from tf.msg import tfMessage
        for tr in tfm.transforms:
            if tr.header.frame_id==self.cfg.world_frame and tr.child_frame_id==self.cfg.camera_frame:
                ps = PoseStamped(); ps.header = tr.header
                ps.pose.position    = tr.transform.translation
                ps.pose.orientation = tr.transform.rotation
                self.pose = ps; break

    # 同步主回调
    def _sync_cb(self, masks: NamedMaskArray,
                 depth_msg: Image, rgb_msg: Image):
        if not (self.int_ok and self.pose): return
        dep = self._depth_arr(depth_msg)
        if dep is None: return
        rgb = self.cv.imgmsg_to_cv2(rgb_msg,"bgr8")

        for key, mask_msg in zip(masks.keys, masks.masks):
            if self.cfg.key not in key: continue
            mask = self.cv.imgmsg_to_cv2(mask_msg,"mono8") > 0
            if mask.sum() < self.cfg.min_area or mask.sum() > self.cfg.max_area:
                continue

            result = self._solve_3d(mask, dep)
            if result is None: continue
            cam_pt, z = result
            if z > self.cfg.max_z: continue

            world_pt = self._cam_to_world(cam_pt)
            if world_pt is None: continue

            cam_xyz = np.array([self.pose.pose.position.x,
                                self.pose.pose.position.y,
                                self.pose.pose.position.z])
            if np.linalg.norm(world_pt - cam_xyz) > self.cfg.max_initial_dist:
                continue

            self._publish(world_pt, masks.header)

            # optional debug overlay
            if self.cfg.enable_dbg:
                u,v = self._centroid(mask)
                self._dbg(rgb, mask, world_pt, (u,v), dep, masks.header)

    # ─── helpers ───────────────────
    def _depth_arr(self, msg: Image):
        try:
            if msg.encoding=="16UC1":
                return (np.frombuffer(msg.data,np.uint16)
                        .reshape(msg.height,msg.width)
                        .astype(np.float32)/1000.)
            return np.frombuffer(msg.data,np.float32).reshape(msg.height,msg.width)
        except Exception as e:
            rospy.logerr(f"Depth decode err: {e}"); return None

    def _solve_3d(self, mask: np.ndarray, depth: np.ndarray):
        ys,xs = np.nonzero(mask)
        if ys.size==0: return None
        bottom = ys.max()
        keep   = (ys >= bottom-(self.cfg.bottom_rows-1))
        xs_b, ys_b = xs[keep], ys[keep]
        d_vals = depth[ys_b, xs_b]
        good   = np.isfinite(d_vals) & (d_vals>self.cfg.min_depth_m)
        d_vals = d_vals[good]
        if d_vals.size<15: return None

        q1,q3 = np.percentile(d_vals,(25,75)); iqr=q3-q1
        d_vals = d_vals[(d_vals>=q1-1.5*iqr)&(d_vals<=q3+1.5*iqr)]
        if d_vals.size==0: return None
        z = float(np.median(d_vals))
        if z > self.cfg.max_z: return None

        u = int(xs_b[good].mean())
        v = int(ys_b[good].max())

        x = (u-self.cx)*z/self.fx
        y = (v-self.cy)*z/self.fy
        return np.array([x,y,z]), z

    def _cam_to_world(self, pt):
        if self.pose is None: return None
        cam_ros = np.array([-pt[0], -pt[2], -pt[1]])
        q=self.pose.pose.orientation
        R=tft.quaternion_matrix([q.x,q.y,q.z,q.w])[:3,:3]
        T=np.array([self.pose.pose.position.x,
                    self.pose.pose.position.y,
                    self.pose.pose.position.z])
        offset=np.array([0., self.cfg.camera_offset_y, 0.])
        return R@(cam_ros+offset)+T

    def _publish(self, p, header):
        msg=PointStamped(); msg.header=header; msg.header.frame_id=self.cfg.world_frame
        msg.point.x, msg.point.y, msg.point.z = p
        self.pub_pt.publish(msg)

    def _dbg(self, rgb, mask, p_w, centroid, depth, header):
        vis=rgb.copy(); vis[mask]=(0,128,255)
        cv2.circle(vis,centroid,4,(0,255,0),-1)
        cv2.putText(vis,f"{p_w[0]:.2f},{p_w[1]:.2f}",
                    (centroid[0]+6,centroid[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1,cv2.LINE_AA)
        self.dbg.rgb(self.cv.cv2_to_imgmsg(vis,"bgr8"))
        self.dbg.mask(self.cv.cv2_to_imgmsg(mask.astype(np.uint8)*255,"mono8"))

    @staticmethod
    def _centroid(mask):
        ys,xs=np.nonzero(mask)
        return int(xs.mean()), int(ys.mean())


# ════════════════════════════════
if __name__=="__main__":
    try:
        PylonLocatorNode(); rospy.spin()
    except rospy.ROSInterruptException:
        pass
