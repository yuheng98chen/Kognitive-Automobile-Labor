#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
import rospkg
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from kal_yolo_detection.msg import NamedMaskArray

# ------------------------------------------------------------------------------
# Optional Intel optimisations — *not* required
# ------------------------------------------------------------------------------
try:
    import intel_extension_for_pytorch as ipex  # type: ignore
    _IPEX_AVAILABLE = True
except Exception:                               # pragma: no cover
    _IPEX_AVAILABLE = False
# ------------------------------------------------------------------------------


class InferenceDevice(str, Enum):
    """Supported inference back-ends."""
    CPU = "cpu"
    CUDA0 = "cuda:0"


# ==============================================================================
# Parameter blocks
# ==============================================================================

@dataclass
class ColorPalette:
    """
    Fixed BGR colours used for debug overlays.

    Comment: OpenCV works in BGR, NOT RGB.
    """
    predefined: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "stop_sign": (0, 255, 255),     # Yellow
        "pylon": (0, 255, 0),           # Green
    })

    # ------------------------------------------------------------------
    def __call__(self, class_name: str) -> Tuple[int, int, int]:
        """
        Args:
            class_name: The YOLO class label.

        Returns:
            Tuple[int, int, int]: BGR colour.
        """
        return self.predefined.get(class_name, (255, 255, 255))  # default white


@dataclass
class NodeConfig:
    """
    All parameters that can be dynamically overridden via `rosparam`.
    """
    # --- model & inference -----------------------------------------------------
    model_path: str = "model/best_model_2.pt"         # relative to package root
    device: str = "auto"                            # "auto" | "cpu" | "cuda:0"
    confidence_threshold: float = 0.7               # global default
    class_conf_thresholds: Dict[str, float] = field(default_factory=dict)

    # --- ROS topics -----------------------------------------------------------
    input_topic: str = "/camera_front/color/processed"
    mask_topic: str = "/yolo_detection/segmentation_mask"
    debug_topic: str = "/yolo_detection/debug_image"

    # --- Debug overlay --------------------------------------------------------
    publish_debug: bool = True
    debug_overlay_classes: List[str] = field(
        default_factory=lambda: ["stop_sign", "pylon"])
    debug_alpha: float = 0.6
    debug_beta: float = 0.4

    # --------------------------------------------------------------------------
    @classmethod
    def from_ros_params(cls) -> "NodeConfig":
        """Create a ``NodeConfig`` populated from ``~``-prefixed ROS parameters.

        Returns:
            NodeConfig: Configuration with any ROS overrides applied.
        """
        return cls(
            model_path=rospy.get_param("~model_path", cls.model_path),
            device=rospy.get_param("~device", cls.device),
            confidence_threshold=rospy.get_param(
                "~confidence_threshold", cls.confidence_threshold),
            class_conf_thresholds=rospy.get_param(
                "~class_conf_thresholds", {}),
            input_topic=rospy.get_param("~input_topic", cls.input_topic),
            mask_topic=rospy.get_param("~mask_topic", cls.mask_topic),
            debug_topic=rospy.get_param("~debug_topic", cls.debug_topic),
            publish_debug=rospy.get_param("~publish_debug", cls.publish_debug),
            debug_overlay_classes=rospy.get_param(
                "~debug_overlay_classes",
                getattr(cls, "debug_overlay_classes",
                        ["stop_sign", "pylon"])
            ),
            debug_alpha=rospy.get_param("~debug_alpha", cls.debug_alpha),
            debug_beta=rospy.get_param("~debug_beta", cls.debug_beta),
        )


# ==============================================================================
# Model I/O helpers
# ==============================================================================

class ModelManager:
    """
    Wraps a Ultralytics-YOLO model instance and hides device selection.
    """

    def __init__(self, cfg: NodeConfig) -> None:
        self.cfg = cfg
        self.model: Optional[YOLO] = None
        self.class_names: Dict[int, str] = {}
        self.device: str = self._select_device()

    # ----------------------------------------------------------------------
    def _select_device(self) -> str:
        """Resolve the device string that will be handed to Ultralytics.

        Returns:
            str: Device identifier such as ``"cpu"`` or ``"cuda:0"``.
        """
        if self.cfg.device.lower() == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"

        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            rospy.logwarn(
                "CUDA requested but unavailable - falling back to CPU")
            return "cpu"

        return self.cfg.device

    # ----------------------------------------------------------------------
    def load(self, package_root: str) -> None:
        """
        Load model weights onto the resolved device.

        Args:
            package_root: Absolute path of `kal_yolo_detection` package.

        Raises:
            FileNotFoundError: When model file cannot be located.
            RuntimeError:     On any Ultralytics load error.
        """
        model_path = (
            self.cfg.model_path if os.path.isabs(self.cfg.model_path)
            else os.path.join(package_root, self.cfg.model_path)
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device).eval()

            # Optional CPU optimisations
            if self.device == "cpu":
                if _IPEX_AVAILABLE:
                    self.model = ipex.optimize(
                        self.model, dtype=torch.float32, inplace=True)

            self.class_names = getattr(self.model, "names", {})
        except Exception as ex:
            raise RuntimeError(f"YOLO load failed: {ex}") from ex

    # ----------------------------------------------------------------------
    def __call__(self, image_bgr: np.ndarray) -> Results:
        """
        Run inference.

        Args:
            image_bgr: OpenCV BGR image.

        Returns:
            Ultralytics `Results` object.
        """
        assert self.model is not None, "ModelManager.load() not called"
        with torch.no_grad():
            return self.model(image_bgr, device=self.device, verbose=False)[0]


# ==============================================================================
# Mask post-processing
# ==============================================================================

class MaskProcessor:
    """Converts Ultralytics masks → custom `NamedMaskArray` message."""

    def __init__(self, cfg: NodeConfig, class_names: Dict[int, str]) -> None:
        self.cfg, self.class_names = cfg, class_names
        self.bridge = CvBridge()

    # ------------------------------------------------------------------
    def process(
        self,
        result: Results,
        img_shape: Tuple[int, int],
        header
    ) -> NamedMaskArray:
        """
        Args:
            result:  Single-image YOLO `Results`.
            img_shape: (H, W) of the original frame.
            header:  Incoming `sensor_msgs/Image` header for stamp/frame_id
                     reuse.

        Returns:
            NamedMaskArray: Binary masks keyed by `<class>[_N]`.
        """
        masks = self._extract_masks(result, img_shape)
        return self._to_ros_msg(masks, header)

    # ------------------------------------------------------------------
    def _extract_masks(
        self,
        result: Results,
        full_shape: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """Extract and post-process masks that pass confidence thresholds.

        Args:
            result (Results): Ultralytics inference result for a single image.
            full_shape (Tuple[int, int]): Desired mask output shape ``(H, W)``.

        Returns:
            Dict[str, np.ndarray]: Binary masks keyed by unique class names.
        """
        if result.masks is None or result.boxes is None:
            return {}

        h, w = full_shape
        collected: Dict[str, np.ndarray] = {}

        for idx, mask_tensor in enumerate(result.masks.data):
            mask = mask_tensor.cpu().numpy()
            if mask.shape != full_shape:
                mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)

            # Confidence gate
            class_id = int(result.boxes.cls[idx])
            conf = float(result.boxes.conf[idx])
            name = self.class_names.get(class_id, f"class_{class_id}")

            thr = self.cfg.class_conf_thresholds.get(
                name, self.cfg.confidence_threshold)

            if conf < thr or mask.max() < 0.5:
                continue

            key = self._unique_key(name, collected)
            collected[key] = (mask > 0.5).astype(np.uint8) * 255

        return collected

    # ------------------------------------------------------------------
    @staticmethod
    def _unique_key(base: str, existing: Dict[str, Any]) -> str:
        """Generate a unique key for a mask when multiple instances exist.

        Args:
            base (str): Base class label.
            existing (Dict[str, Any]): Dictionary of existing mask keys.

        Returns:
            str: Unique key derived from ``base``.
        """
        if base not in existing:
            return base
        i = 1
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    # ------------------------------------------------------------------
    def _to_ros_msg(
        self,
        masks: Dict[str, np.ndarray],
        header
    ) -> NamedMaskArray:
        """Convert dict → `NamedMaskArray`."""
        msg = NamedMaskArray()
        msg.header = header
        msg.keys = list(masks.keys())

        for key in msg.keys:
            img = self.bridge.cv2_to_imgmsg(masks[key], encoding="mono8")
            img.header = header
            msg.masks.append(img)
        return msg


# ==============================================================================
# Debug visualisation (isolated block for easy removal)
# ==============================================================================

class DebugMux:
    """
    Convenience wrapper so debug publishing can be stripped by deleting
    this class plus two call-sites (`self.debug = DebugMux(...)` and
    `self.debug.publish(...)`).
    """

    def __init__(self, cfg: NodeConfig, colours: ColorPalette) -> None:
        self.cfg, self.colours = cfg, colours
        self.bridge = CvBridge()

        self.pub: Optional[rospy.Publisher] = (
            rospy.Publisher(cfg.debug_topic, Image, queue_size=1)
            if cfg.publish_debug else None
        )
        self.allowed = set(cfg.debug_overlay_classes)

    # ------------------------------------------------------------------
    def publish(
        self,
        src_bgr: np.ndarray,
        masks: Dict[str, np.ndarray],
        header
    ) -> None:
        """Blend the specified binary masks onto the RGB image and publish.

        Args:
            src_bgr (np.ndarray): Source BGR frame.
            masks (Dict[str, np.ndarray]): Mapping from mask key to binary mask.
            header: ROS header used to stamp the published image.

        Returns:
            None
        """
        if self.pub is None:
            return

        overlay = np.zeros_like(src_bgr)
        keys = self._filter(masks.keys())
        for k in keys:
            base = re.sub(r"_\d+$", "", k)
            colour = self.colours(base)
            overlay[masks[k] > 0] = colour

        blended = self._alpha_blend(src_bgr, overlay)
        msg = self.bridge.cv2_to_imgmsg(blended, encoding="bgr8")
        msg.header = header
        self.pub.publish(msg)

    # ------------------------------------------------------------------
    def _filter(self, keys) -> List[str]:
        """Filter mask keys based on configured debug overlay classes.

        Args:
            keys (List[str]): Input mask keys.

        Returns:
            List[str]: Keys permitted for overlay rendering.
        """
        if not self.allowed:
            return list(keys)
        out = []
        for k in keys:
            base = re.sub(r"_\d+$", "", k)
            if k in self.allowed or base in self.allowed:
                out.append(k)
        return list(set(out))

    # ------------------------------------------------------------------
    def _alpha_blend(self, img: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Alpha-blend an overlay onto the original image.

        Args:
            img (np.ndarray): Original BGR image.
            overlay (np.ndarray): BGR overlay image of identical shape.

        Returns:
            np.ndarray: Blended BGR image.
        """
        mask = np.any(overlay > 0, axis=-1)
        if not mask.any():
            return img
        result = img.copy()
        alpha, beta = self.cfg.debug_alpha, self.cfg.debug_beta
        result[mask] = (img[mask] * alpha + overlay[mask]
                        * beta).astype(np.uint8)
        return result


# ==============================================================================
# Main ROS Node
# ==============================================================================

class YoloSegmentationNode:
    """Starts subscribers, publishers and ties all helpers together."""

    def __init__(self) -> None:
        rospy.init_node("yolo_segmentation_node")

        # ------------------------------------------------------------------
        self.cfg = NodeConfig.from_ros_params()

        # Package absolute path (needed for relative model paths)
        pkg_root = rospkg.RosPack().get_path("kal_yolo_detection")

        # Helper singletons
        self.palette = ColorPalette()
        self.model = ModelManager(self.cfg)
        self.model.load(pkg_root)

        self.mask_proc = MaskProcessor(self.cfg, self.model.class_names)
        self.debug = DebugMux(self.cfg, self.palette)

        self.bridge = CvBridge()

        # ------------------------------------------------------------------
        # Publishers / Subscriber
        self.mask_pub = rospy.Publisher(
            self.cfg.mask_topic, NamedMaskArray, queue_size=1)

        rospy.Subscriber(
            self.cfg.input_topic, Image, self._on_image, queue_size=1)

        rospy.loginfo(f"Ready. Listening on {self.cfg.input_topic}")

    # ------------------------------------------------------------------
    def _on_image(self, msg: Image) -> None:
        """
        Per-frame callback.

        Steps:
        1. Convert to OpenCV.
        2. Run YOLO instance segmentation.
        3. Convert masks → `NamedMaskArray` and publish.
        4. Optionally create coloured debug overlay.
        """
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as ex:
            rospy.logerr_throttle(5.0, f"cv_bridge conversion error: {ex}")
            return

        # -------------------------------------- inference
        try:
            res = self.model(bgr)
        except Exception as ex:
            rospy.logerr_throttle(2.0, f"Inference error: {ex}")
            return

        # -------------------------------------- masks
        mask_msg = self.mask_proc.process(res, bgr.shape[:2], msg.header)
        self.mask_pub.publish(mask_msg)

        # Dict[str, np.ndarray] for debug overlay
        mask_arrays = {
            k: self.bridge.imgmsg_to_cv2(m, "mono8")
            for k, m in zip(mask_msg.keys, mask_msg.masks)
        }
        self.debug.publish(bgr, mask_arrays, msg.header)


# ==============================================================================
# Entrypoint
# ==============================================================================

def main() -> None:
    """Spawn node and enter the ROS spin loop."""
    try:
        YoloSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as ex:                  # hard failure during init
        rospy.logfatal(f"Node failed: {ex}")


if __name__ == "__main__":
    main()
