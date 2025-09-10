# Camera View Dashboard (`camera_view`)

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Web Interface](#web-interface)
- [ROS2 Integration](#ros2-integration)
- [Monitoring](#monitoring)
- [Development](#development)
- [License](#license)
- [Citation](#citation)

---

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)  
![FastAPI](https://img.shields.io/badge/FastAPI-Ready-green)  
![Three.js](https://img.shields.io/badge/Three.js-WebGPU/WebGL-purple)  
![License](https://img.shields.io/badge/License-MIT-yellow)  
![Maintained](https://img.shields.io/badge/Maintained-Yes-success)  

**Author/Maintainer:** Sean Rice ([seanrice@umich.edu](mailto:seanrice@umich.edu))  
**License:** Creative Commons Attribution�NonCommercial 4.0 International (CC BY-NC 4.0)  
**Version:** 0.1.0  

This package provides a **web-based monitoring dashboard** for multi-camera and LiDAR ROS2 streams. It combines **FastAPI**, **WebSockets**, and **Three.js** for live visualization of camera feeds, annotated detections, DINOV3 embeddings, and LiDAR point clouds with IMU orientation.

---

## ?? Features

- **Multi-Camera Integration**
  - Subscribes to `/cameras/*/image_raw` and `/cameras/*/annotated`.
  - Displays both **raw** and **annotated** feeds in a responsive grid UI.
  - Supports fullscreen mode per camera.

- **Detection & Embedding Visualization**
  - Displays `vision_msgs/Detection2DArray` objects with class labels and bounding boxes.
  - Streams DINOV3 **embeddings** as JSON payloads.

- **LiDAR + IMU Integration**
  - Renders 3D LiDAR point clouds in **WebGPU** (or WebGL fallback).
  - IMU orientation visualized as an arrow helper.
  - Real-time packet loss statistics overlay.

- **Web Dashboard**
  - Tabbed interface for **Raw**, **Annotated**, **Detections**, **Embeddings**, **LiDAR**, and **Stats**.
  - Camera renaming and management via ROS2 `RenameCamera` service.
  - Session-based recording and playback support.

- **Secure by Default**
  - Runs over HTTPS/WSS (`uvicorn` with TLS).
  - Uses packaged SSL certs (`camera_view/ssl/cert.pem`, `camera_view/ssl/key.pem`).

---

## ?? Repository Structure

```
camera_view/
+-- __init__.py
+-- camera_view.py        # Main FastAPI + ROS2 node
+-- static/               # Dashboard assets (JS, CSS, HTML)
+-- ssl/                  # SSL cert.pem / key.pem
+-- package.xml           # ROS2 package manifest
+-- setup.py              # Python packaging
```

```mermaid
flowchart LR
    subgraph ROS2
        IMG[/cameras/*/image_raw/]
        ANN[/cameras/*/annotated]
        DET[/cameras/*/detections]
        EMB[/cameras/*/embeddings]
        LIDAR[/lidar_points/]
        IMU[/lidar_imu/]
    end

    IMG --> NODE["WebStreamNode (ROS2)"]
    ANN --> NODE
    DET --> NODE
    EMB --> NODE
    LIDAR --> NODE
    IMU --> NODE

    NODE -->|Publishes| WS["WebSocket Server (wss://:1080/ws)"]
    WS --> UI["Web Dashboard<br/>(FastAPI + Three.js)"]

    style NODE fill:#ff9,stroke:#333,stroke-width:1px
    style WS fill:#bfb,stroke:#333,stroke-width:1px
    style UI fill:#bbf,stroke:#333,stroke-width:1px
```

---

## ?? Installation

### Dependencies

- **System**
  - Ubuntu 22.04+
  - ROS2 Humble or newer
  - OpenCV (`opencv-python`)
  - FastAPI + Uvicorn

- **Python**
  ```bash
  pip install fastapi uvicorn opencv-python numpy transforms3d
  ```

### Build & Run

```bash
cd ~/multicamera_ros2/catkin_ws
colcon build
source install/setup.bash
```

---

## ?? Usage

Run the node:

```bash
ros2 run camera_view camera_view
```

Override SSL cert/key (optional):

```bash
export WVW_CERT=/path/to/cert.pem
export WVW_KEY=/path/to/key.pem
```

---

## ?? Web Interface

Open your browser:

```
https://<host>:1080/
```

Tabs available:
- **Raw** � Live camera feeds
- **Annotated** � Processed feeds with detections
- **Detections** � Class + confidence list
- **Embeddings** � DINOV3 JSON payloads
- **Stats** � Session stats, playback, and charts
- **LiDAR** � Interactive 3D point cloud + IMU
- **Cameras** � Manage and rename cameras

---

## ?? ROS2 Integration

### Subscribed Topics
- `/cameras/<id>/image_raw` (`sensor_msgs/Image`)
- `/cameras/<id>/annotated` (`sensor_msgs/Image`)
- `/cameras/<id>/embeddings` (`std_msgs/String`)
- `/lidar_points` (`sensor_msgs/PointCloud2`)
- `/lidar_imu` (`sensor_msgs/Imu`)
- `/lidar_packets_loss` (`std_msgs/Int32`)

### Services
- `rename_camera` (`camera_msgs/srv/RenameCamera`)

---

## ?? Monitoring

- WebSocket stream updates at ~30Hz for frames and metadata.
- Lightweight JSON envelopes for meta/detections, binary frames for LiDAR.
- Logs available via ROS2 console and FastAPI runtime.

---

## ?? Citation

If you use this work in your research, please cite:

```
@software{rice2025cameraview,
  author       = {Sean Rice},
  title        = {Camera View Dashboard for Multi-Camera + LiDAR ROS2},
  year         = {2025},
  institution  = {University of Michigan},
  email        = {seanrice@umich.edu},
  license      = {MIT}
}
```
