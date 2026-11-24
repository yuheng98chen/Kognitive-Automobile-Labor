# kal_yolo_detection

This package provides a ROS node, `yolo_inference_node.py`, for object detection and segmentation using YOLO models with the Ultralytics library (PyTorch backend).

The node subscribes to an image topic, performs inference, and publishes results as a `kal_yolo_detection/NamedMaskArray` message. For `pylon` objects, it publishes segmentation masks. For `stop_sign` objects, it publishes a rectangular mask from the detected bounding box.

## Installation (Intelex Support!)

```bash
pip install --upgrade torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install intel_extension_for_pytorch==2.3.100
```

## Node: `yolo_inference_node.py`

This node performs segmentation using a PyTorch (`.pt`) model file.

### Subscribed Topics

* `~input_topic` (type: `sensor_msgs/Image`, encoding: `rgb8`, default: `/camera_front/color/processed`)
  * Input RGB image for inference. The `utils` node provides this topic.

### Published Topics

* `~mask_topic` (type: `kal_yolo_detection/NamedMaskArray`, default: `/yolo_detection/segmentation_mask`)
  * Publishes an array of named binary masks. For pylons, this is a segmentation mask. For stop signs, this is a filled rectangle corresponding to the bounding box.
* `~debug_topic` (type: `sensor_msgs/Image`, default: `/yolo_detection/debug_image`)
  * Publishes the input image with debug visualizations (pylon masks, stop sign boxes). Published only if `~publish_debug` is `true`.

### Parameters

Parameters are loaded from `config/yolo_params.yaml` via the launch file. To change the node's behavior, modify this file.

* `~model_path` (string, default: `model/best_model.pt`)
  * Path to the YOLO `.pt` model file (relative to package or absolute).
* `~device` (string, default: `auto`)
  * Inference device: `auto`, `cpu`, `cuda:0`, etc.
* `~confidence_threshold` (float, default: `0.7`)
  * Global confidence threshold for filtering detections.
* `~class_conf_thresholds` (dict, default: `{}`)
  * Per-class confidence thresholds to override the global one.
* `~input_topic` (string, default: `/camera_front/color/processed`)
  * Input image topic.
* `~mask_topic` (string, default: `/yolo_detection/segmentation_mask`)
  * Output topic for the `NamedMaskArray`.
* `~debug_topic` (string, default: `/yolo_detection/debug_image`)
  * Output topic for the debug image.
* `~publish_debug` (bool, default: `true`)
  * Enables publishing the debug image.
* `~debug_overlay_classes` (list of strings, default: `["stop_sign", "pylon"]`)
  * Controls which visualizations are shown. Can contain "pylon" to show segmentation masks and/or "stop_sign" to show bounding boxes. If empty, no visualizations are shown.
* `~debug_alpha` (float, default: `0.6`)
  * Alpha blending for the original image when creating the pylon mask overlay.
* `~debug_beta` (float, default: `0.4`)
  * Alpha blending for the pylon mask overlay in the debug view.

## How to Launch

To run the node, use the provided launch file. This starts the node and loads all configuration from `config/yolo_params.yaml`.

```bash
roslaunch kal_yolo_detection yolo_inference.launch
```

## Custom Messages

* **`kal_yolo_detection/NamedMaskArray`**:
  * `std_msgs/Header header`
  * `string[] keys` (e.g., "pylon_0", "stop_sign_0")
  * `sensor_msgs/Image[] masks` (binary `mono8` images)
