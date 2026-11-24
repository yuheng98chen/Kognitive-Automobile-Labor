# utils Downscale Node Overview

## Purpose

Efficiently *undistorts* the front-camera colour stream and down-scales both the colour and the corresponding depth images, publishing geometry-consistent pairs at half resolution (configurable downscaling).

## Key Parameter

| Name | Type | Default | Effect |
| ---- | ---- | ------- | ------ |
| `~downscale_factor` | `int` | **2** | Divides input width / height and focal lengths by this factor before publishing. |

## **Input Topics**

| Topic | Message type | Encoding | Nominal resolution | Notes |
|-------|--------------|----------|--------------------|-------|
| `/camera_front/color/image_raw` | `sensor_msgs/Image` | `rgb8` | 1280 × 720 | 24-bit RGB. Node will throw if still `bgr8`. |
| `/camera_front/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | usually `16UC1` (unsigned mm) | 1280 × 720 | Any other encoding passes through but **must** match `CV_16UC1` for internal buffer. |

## **Output Topics**

| Topic | Message type | Encoding | Resolution (`downscale_factor` = 2) | Notes |
|-------|--------------|----------|--------------------------------------|-------|
| `/camera_front/color/processed` | `sensor_msgs/Image` | `rgb8` | 640 × 360 | Undistorted & resized with bilinear filtering. |
| `/camera_front/depth/processed` | `sensor_msgs/Image` | same as input (typically `16UC1`) | 640 × 360 | Nearest-neighbour resize to keep integer depth exact. |
| `/camera_front/processed/camera_info` | `sensor_msgs/CameraInfo` | — | matches outputs | Intrinsics *K* and *P* scaled, `binning_x/y = downscale_factor`. |

## Processing Pipeline

1. **Camera model load** – Distortion coefficients *D* and intrinsic matrix *K* are hard-coded at start-up.  
2. **Pre-computation** – `cv::initUndistortRectifyMap` builds remap tables for the half-size image once, using the adjusted *K*.  
3. **Colour callback**  
   * Converts to OpenCV with `cv_bridge` expecting `RGB8`.  
   * Single-step `cv::remap` undistorts **and** resizes.  
   * Creates fresh `sensor_msgs/Image` (`encoding="rgb8"`, `step = w × 3`) and publishes.  
   * Sends updated `CameraInfo` with identical timestamp.  
4. **Depth callback**  
   * Accepts any encoding but works with `CV_16UC1` buffer size.  
   * `cv::resize` with `INTER_NEAREST` avoids smoothing integer depth.  
   * Publishes a copy of the header/encoding, with `step` recomputed for new width.  

## Frame ID Convention

All published images keep the original header, so the optical frame remains `camera_front_optical_frame`. The dedicated `CameraInfo` message shares the same frame for easy synchronisation with downstream perception or VIO pipelines.
