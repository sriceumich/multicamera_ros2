#!/usr/bin/env python3
import asyncio
import json
import base64
import threading
import time
import os
import os.path as osp
from importlib.resources import files
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
from sensor_msgs.msg import PointCloud2
import sys
import numpy as np
from sensor_msgs.msg import PointField
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import Imu
from std_msgs.msg import Int32
import transforms3d.euler as euler
from fastapi.staticfiles import StaticFiles
import random
import math
import struct
from fastapi import Body
from camera_msgs.srv import RenameCamera
from rclpy.task import Future

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, QoSPresetProfiles
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge

from fastapi import FastAPI, WebSocket, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

# ---------------- Config ----------------
app = FastAPI()
bridge = CvBridge()
static_dir = osp.join(osp.dirname(__file__), "static")
if osp.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"⚠️ static dir not found: {static_dir}")

# ---------------- Shared state ----------------
latest_raw_frames: Dict[str, Optional[str]] = {}  # {cam_id: b64_jpg}
latest_ann_frames: Dict[str, Optional[str]] = {}  # {cam_id: b64_jpg}
latest_detections: Dict[str, List[Dict[str, Any]]] = {}  # {cam_id: [{class,score,bbox}]}
latest_alerts: List[Dict[str, Any]] = []
known_cameras = set()

latest_embeddings: Dict[str, Any] = {}     # {cam_id: parsed_json_or_raw}

# --- Motion state (per camera) ---
_prev_small_raw = {}      # cam -> small grayscale frame (numpy)
_prev_small_ann = {}      # cam -> small grayscale frame (numpy)
_motion_until = {}        # cam -> unix_ts until which we consider "in motion"

latest_lidar_points = []
latest_lidar_imu = None       # {roll, pitch, yaw}
latest_lidar_loss = 0
LIDAR_MAGIC = b"LDR1"


def pack_lidar_frame(pts: np.ndarray) -> bytes:
    """
    pts: NumPy float32 array of shape (N, 4) in XYZI order, C-contiguous.
    Returns: header (8 bytes) + payload (N*16 bytes).
    """
    # Ensure expected dtype/layout with no copy unless needed
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    n = int(pts.shape[0])
    header = struct.pack("<4sI", LIDAR_MAGIC, n)  # magic + count
    return header + pts.tobytes(order="C")


def ts_filename(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y%m%d_%H%M%S_") + f"{int((ts - int(ts)) * 1000):03d}.jpg"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------------- ROS Node ----------------
class WebStreamNode(Node):
    def __init__(self):
        super().__init__("camera_view")
        self.bridge = CvBridge()
        self.subs = []
        self._known_topics = set()
        self._lidar_dtype = None   # cache built dtype
        self._use_pc2_numpy = hasattr(pc2, "read_points_numpy")

        # Initial subscribe + periodic refresh
        self._setup_initial_subs()
        self.create_timer(2.0, self._refresh_topics)

        # Alerts (reliable ok)
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.lidar_sub = self.create_subscription(PointCloud2, "/lidar_points", self.lidar_cb, qos)
        self.imu_sub   = self.create_subscription(Imu, "/lidar_imu", self.imu_cb, 10)
        self.loss_sub  = self.create_subscription(Int32, "/lidar_packets_loss", self.loss_cb, 10)

    def imu_cb(self, msg: Imu):
        global latest_lidar_imu
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]
        r, p, y = euler.quat2euler(quat)
        latest_lidar_imu = {"roll": r, "pitch": p, "yaw": y}

    def loss_cb(self, msg: Int32):
        global latest_lidar_loss
        latest_lidar_loss = msg.data

    @staticmethod
    def _build_numpy_dtype(msg):
        """
        Build a NumPy structured dtype honoring PointField offsets and point_step.
        Works for arbitrary field order/layout. Caches well.
        """
        np_map = {
            PointField.INT8:    np.int8,
            PointField.UINT8:   np.uint8,
            PointField.INT16:   np.int16,
            PointField.UINT16:  np.uint16,
            PointField.INT32:   np.int32,
            PointField.UINT32:  np.uint32,
            PointField.FLOAT32: np.float32,
            PointField.FLOAT64: np.float64,
        }
        names, formats, offsets = [], [], []
        for f in msg.fields:
            base = np_map[f.datatype]
            if f.count and f.count > 1:
                base = np.dtype((base, (f.count,)))
            names.append(f.name)
            formats.append(base)
            offsets.append(f.offset)
        return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})



    def lidar_cb(self, msg: PointCloud2):
        global latest_lidar_points

        t0 = time.time()

        # Fast path if available in your ROS 2 distro (Humble+ typically has this)
        if self._use_pc2_numpy:
            # Returns a dense float32 NxK array in one go, optionally filtering NaNs.
            pts = pc2.read_points_numpy(
                msg,
                field_names=("x", "y", "z", "intensity"),
                skip_nans=True  # same effect as your generator usage, but vectorized
            ).astype(np.float32, copy=False)
        else:
            # Portable zero-copy fallback using frombuffer with a structured dtype
            if self._lidar_dtype is None:
                self._lidar_dtype = self._build_numpy_dtype(msg)

            count = msg.width * msg.height
            arr = np.frombuffer(msg.data, dtype=self._lidar_dtype, count=count)

            # Handle endianness if needed
            if (msg.is_bigendian and sys.byteorder == "little") or (not msg.is_bigendian and sys.byteorder == "big"):
                arr = arr.byteswap().newbyteorder()

            # Extract the four columns; this creates a contiguous float32 Nx4 array
            # Using view/reshape is fastest when fields are contiguous; fall back to stack otherwise.
            try:
                # Works when x,y,z,intensity are 4 contiguous float32 fields (common case)
                pts = arr[['x', 'y', 'z', 'intensity']].view(np.float32).reshape(-1, 4)
            except ValueError:
                # Generic and still fast
                pts = np.column_stack((arr['x'].astype(np.float32, copy=False),
                                       arr['y'].astype(np.float32, copy=False),
                                       arr['z'].astype(np.float32, copy=False),
                                       arr['intensity'].astype(np.float32, copy=False)))

            # Vectorized NaN filtering to mimic skip_nans=True
            mask = np.isfinite(pts).all(axis=1)
            pts = pts[mask]

        t1 = time.time()
        dec_ms = (t1 - t0) * 1000.0

        latest_lidar_points = pts  # keep as a NumPy array (zero-copy to bytes if you stream)

        # Keep logs light; formatting strings is cheap but printing every frame isn’t.
        #self.get_logger().info(f"lidar_cb decode took {dec_ms:.1f} ms for {pts.shape[0]} pts")
        #self.get_logger().info(f"⚡ lidar_cb got {pts.shape[0]} pts → sending {pts.shape[0]}")


    # ---- Topic discovery & subscribe ----
    def _discover_topics(self):
        topics = {
            "raw": [],
            "annotated": [],
            "detections": [],
            "embeddings": [],
            "df_analysis": [],
            "df_annotated": [],
        }
        for name, types in self.get_topic_names_and_types():
            if name.startswith("/cameras/") and name.endswith("/image_raw") and "sensor_msgs/msg/Image" in types:
                topics["raw"].append(name)
            if name.startswith("/warehouse/annotated/") and "sensor_msgs/msg/Image" in types:
                topics["annotated"].append(name)
            #  DINOV3 embeddings (String JSON)
            if name.startswith("/warehouse/embeddings/") and "std_msgs/msg/String" in types:
                topics["embeddings"].append(name)

        return topics

    def _setup_initial_subs(self):
        topics = self._discover_topics()
        self._subscribe_sets(topics)
        self.get_logger().info(f"Initial discovery: {topics}")

    def _refresh_topics(self):
        topics = self._discover_topics()
        self._subscribe_sets(topics)

    @staticmethod
    def _try_json_loads(s: str) -> Any:
        try:
            return json.loads(s)
        except Exception:
            return s  # keep raw if not valid JSON

    # NEW: DINOV3 embeddings
    def emb_cb(self, msg: String, cam_id: str):
        try:
            payload = self._try_json_loads(msg.data or "")
            latest_embeddings[cam_id] = payload
        except Exception as e:
            self.get_logger().error(f"Embeddings parse error (cam={cam_id}): {e}")


    def _subscribe_sets(self, topics):
        # RAW images (BEST_EFFORT/SENSOR_DATA)
        for t in topics["raw"]:
            if t in self._known_topics:
                continue
            cam_id = self._cam_id_from_topic(t)
            self.get_logger().info(f"Subscribing RAW: {t} (id={cam_id})")
            self.subs.append(
                self.create_subscription(
                    Image, t,
                    lambda msg, cid=cam_id: self._handle_frame(msg, "raw", cid),
                    qos_profile=QoSPresetProfiles.SENSOR_DATA.value
                )
            )
            self._known_topics.add(t)
            known_cameras.add(cam_id)
            latest_raw_frames.setdefault(cam_id, None)

        # Annotated images (BEST_EFFORT/SENSOR_DATA)
        for t in topics["annotated"]:
            if t in self._known_topics:
                continue
            cam_id = self._cam_id_from_topic(t)
            self.get_logger().info(f"Subscribing ANN: {t} (id={cam_id})")
            self.subs.append(
                self.create_subscription(
                    Image, t,
                    lambda msg, cid=cam_id: self._handle_frame(msg, "annotated", cid),
                    qos_profile=QoSPresetProfiles.SENSOR_DATA.value
                )
            )
            self._known_topics.add(t)
            known_cameras.add(cam_id)
            latest_ann_frames.setdefault(cam_id, None)


        # NEW: DINOV3 embeddings (reliable ok)
        for t in topics["embeddings"]:
            if t in self._known_topics:
                continue
            cam_id = self._cam_id_from_topic(t)
            self.get_logger().info(f"Subscribing EMBEDDINGS: {t} (id={cam_id})")
            self.subs.append(
                self.create_subscription(
                    String, t, lambda msg, cid=cam_id: self.emb_cb(msg, cid), 10
                )
            )
            self._known_topics.add(t)
            known_cameras.add(cam_id)
            latest_embeddings.setdefault(cam_id, None)

            self._known_topics.add(t)
            known_cameras.add(cam_id)
            latest_df_ann_frames.setdefault(cam_id, None)
            
    # ---- cam_id parsing ----
    def _cam_id_from_topic(self, topic: str) -> str:
        if topic.startswith("/cameras/") and topic.endswith("/image_raw"):
            return topic.split("/")[-2]
        for prefix in (
            "/warehouse/annotated/",
            "/warehouse/detections/",
            "/warehouse/embeddings/",
        ):
            if topic.startswith(prefix):
                return topic.split("/")[-1]
        return topic.split("/")[-1]
        
    # ---- Callbacks ----
    def _handle_frame(self, msg: Image, stream_type: str, cam_id: str):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                return
            b64 = base64.b64encode(buf).decode("utf-8")
            if stream_type == "raw":
                latest_raw_frames[cam_id] = b64
            else:
                latest_ann_frames[cam_id] = b64

        except Exception as e:
            self.get_logger().error(f"Frame error ({cam_id}/{stream_type}): {e}")


# ---------------- HTTP UI (Tabbed Dashboard w/ Record & Playback & Stats) ----------------
@app.get("/")
async def index():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Camera/Lidar Dashboard</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#0f1115; color:#e5e7eb; margin:0; }
    header { padding:14px 18px; background:#151923; border-bottom:1px solid #232836; display:flex; align-items:center; gap:12px; }
    h1 { font-size:18px; margin:0; }
    .tabs { display:flex; gap:8px; margin-left:auto; flex-wrap:wrap;}
    .tab { padding:8px 12px; background:#1d2330; border:1px solid #2a3040; border-radius:8px; cursor:pointer; }
    .tab.active { background:#334155; }
    .section { display:none; padding:16px; }
    .section.active { display:block; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:12px; }
    .tile { background:#161b26; border:1px solid #232836; border-radius:10px; overflow:hidden; }
    .tile header { display:flex; align-items:center; justify-content:space-between; padding:8px 12px; background:#0f1420; border-bottom:1px solid #232836; }
    .tile h3 { margin:0; font-size:14px; color:#cbd5e1; }
    .imgwrap { background:#0b0e14; display:flex; align-items:center; justify-content:center; min-height:200px; }
    img { display:block; width:100%; height:auto; cursor:pointer; }
    .meta { font-size:12px; color:#a1a7b5; padding:8px 12px; }
    .alertbox { padding:12px; background:#2a0f10; border:1px solid #5b1d1f; border-radius:10px; margin:12px; }
    .pill { font-size:11px; padding:2px 8px; border-radius:999px; border:1px solid #374151; color:#cbd5e1; }
    .controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:8px 0; }
    input[type="text"], select { background:#0f1420; color:#e5e7eb; border:1px solid #2a3040; border-radius:8px; padding:6px 10px; }
    button { background:#1e293b; color:#e5e7eb; border:1px solid #2a3040; border-radius:8px; padding:6px 12px; cursor:pointer; }
    button.primary { background:#334155; }
    .row { display:flex; gap:12px; flex-wrap:wrap; }
    .col { flex:1 1 360px; }
    .timeline { width:100%; }
    svg { width:100%; height:320px; background:#0b0f17; border:1px solid #232836; border-radius:10px; }

    .tile {
      position: relative; /* for glow ball positioning */
      transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
    }

    /* Fullscreen helpers (works with the Fullscreen API) */
    .tile:fullscreen, .tile:-webkit-full-screen {
      width: 100vw; height: 100vh; border-radius: 0;
    }
    .tile:fullscreen .imgwrap, .tile:-webkit-full-screen .imgwrap {
      min-height: calc(100vh - 48px);
    }
    .tile .fs-hint {
      position:absolute; top:8px; right:10px; font-size:11px; opacity:.7;
    }

    #lidar {
      position: relative;
      width: 100vw;
      height: calc(100vh - 48px); /* minus header height */
      margin: 0;
      padding: 0;
    }

    #lidarCanvas {
      width: 100%;
      height: 100%;
      display: block;     /* remove default inline spacing */
    }

  </style>
</head>
<body>
  <header>
    <h1>Camera Monitoring Dashboard</h1>
    <div class="tabs">
      <div class="tab active" data-target="raw">Raw</div>
      <div class="tab" data-target="annotated">Annotated</div>
      <div class="tab" data-target="detections">Detections</div>
      <div class="tab" data-target="embeddings">Embeddings</div>
      <div class="tab" data-target="stats">Stats</div>
      <div class="tab" data-target="lidar">LiDAR</div>
      <div class="tab" data-target="cameras">Cameras</div>

    </div>
  </header>

  <section id="raw" class="section active"><div id="rawGrid" class="grid"></div></section>
  <section id="annotated" class="section"><div id="annGrid" class="grid"></div></section>
  <section id="detections" class="section"><div id="detWrap" style="padding:8px 12px;"></div></section>
  <section id="alerts" class="section"><div id="alertWrap" style="padding:8px 12px;"></div></section>
  <section id="embeddings" class="section"><div id="embGrid" class="grid"></div></section>      <!-- NEW -->
  <section id="lidar" class="section" style="position:relative;">
    <canvas id="lidarCanvas"></canvas>
    <div id="lidarOverlay" style="position:absolute;top:10px;left:10px;
        background:#1d2330;padding:6px 12px;border-radius:8px;
        font-size:12px;color:#cbd5e1;">
      Loss: 0
    </div>
  </section>

  <section id="cameras" class="section">
    <div id="camList" class="grid"></div>
    <div class="controls">
      <button class="primary" onclick="saveCameras()">Save Names</button>
    </div>
  </section>


  <script type="module">


    import * as THREE from "/static/js/three/build/three.webgpu.js";
    import * as WEBGPU from "/static/js/three/examples/jsm/capabilities/WebGPU.js";
    import * as Stats from "/static/js/three/examples/jsm/libs/stats.module.js";
    import { OrbitControls } from '/static/js/three/examples/jsm/controls/OrbitControls.js';

    import { Fn, wgslFn, positionLocal, scriptable, positionWorld, normalLocal, normalWorld, normalView, color, texture, uv, float, vec2, vec3, vec4, oscSine, triplanarTexture, screenUV, js, string, Loop, cameraProjectionMatrix, ScriptableNodeResources } from "/static/js/three/build/three.tsl.js";


    let lidarScene, lidarCamera, lidarRenderer, lidarPointsMesh, imuArrow, controls, lidarGeometry, lidarMaterial, renderer;
    let useWebGPU = false;
    let known_cameras = [];

    async function loadCameraList() {
      const grid = document.getElementById("camList");
      grid.innerHTML = "";
      const cams = Array.from(known_cameras || []);
      cams.forEach(id => {
        const div = document.createElement("div");
        div.className = "tile";
        div.innerHTML = `
          <header><h3>${id}</h3></header>
          <div class="meta">
            <label>Friendly name:
              <input type="text" id="camname-${id}" value="${id}" />
            </label>
          </div>`;
        grid.appendChild(div);
      });
    }

    async function saveCameras() {
      for (const id of known_cameras) {
        const newName = document.getElementById(`camname-${id}`).value.trim();
        if (newName && newName !== id) {
          const res = await fetch("/api/cameras/rename", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({current_id: id, new_name: newName})
          });
          const data = await res.json();
          console.log("Rename response", data);
        }
      }
      alert("Camera names updated. Restart MultiRtspCamNode for changes to take effect.");
    }

    async function initLidar() {
      const canvas = document.getElementById("lidarCanvas");
      useWebGPU = !!navigator.gpu;

      if (useWebGPU) {
        console.log("✅ WEBGPU supported, using WEBGPU.WebGPURenderer");
        lidarRenderer = new THREE.WebGPURenderer({ canvas, antialias: true });
        lidarRenderer.setPixelRatio( window.devicePixelRatio );
        lidarRenderer.setSize( window.innerWidth, window.innerHeight );
        await lidarRenderer.init();

        // WebGPU node-based material
        lidarMaterial = new THREE.PointsNodeMaterial({
          size: 0.01,
          sizeAttenuation: true,
          vertexColors: true
        });
      } else {
        console.log("⚠️ WebGPU not available, using THREE.WebGLRenderer");
        lidarRenderer = new THREE.WebGLRenderer({ canvas, antialias: true });

        // WebGL shader material fallback
        lidarMaterial = new THREE.ShaderMaterial({
          vertexColors: true,
          transparent: true,
          depthTest: true,
          blending: THREE.AdditiveBlending,
          uniforms: { pointSize: { value: 0.01 } },
          vertexShader: `
            uniform float pointSize;
            varying vec3 vColor;
            void main() {
              vColor = color;
              vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
              gl_PointSize = pointSize * (300.0 / -mvPosition.z);
              gl_Position = projectionMatrix * mvPosition;
            }
          `,
          fragmentShader: `
            varying vec3 vColor;
            void main() {
              vec2 c = gl_PointCoord - vec2(0.5);
              if (dot(c, c) > 0.25) discard;
              gl_FragColor = vec4(vColor, 1.0);
            }
          `
        });
      }

      lidarCamera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / (window.innerHeight - 48),
        0.1,
        1000
      );
      lidarCamera.position.set(0, 0, 50);

      // 🔥 Create scene + camera before adding meshes
      lidarScene = new THREE.Scene();
      lidarScene.background = new THREE.Color(0x000000);

      // Initialize geometry with dummy attributes so WebGPU compiles clean
      lidarGeometry = new THREE.BufferGeometry();
      lidarGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3));
      lidarGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array([1, 1, 1]), 3));
      lidarGeometry.computeBoundingSphere();

      lidarPointsMesh = new THREE.Points(lidarGeometry, lidarMaterial);
      lidarScene.add(lidarPointsMesh);

      // controls, grid, axes, etc.
      controls = new OrbitControls(lidarCamera, lidarRenderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;

      const grid = new THREE.GridHelper(100, 100, 0x444444, 0x222222);
      lidarScene.add(grid);
      const axes = new THREE.AxesHelper(5);
      lidarScene.add(axes);

      const dir = new THREE.Vector3(1,0,0);
      imuArrow = new THREE.ArrowHelper(dir, new THREE.Vector3(0,0,0), 2, 0x00ff00);
      lidarScene.add(imuArrow);

      animateLidar();
    }

    function animateLidar() {
      requestAnimationFrame(animateLidar);
      controls.update();                           // ✨ update controls each frame
      lidarRenderer.render(lidarScene, lidarCamera);
    }

    function updateIMU(imu) {
      if (!imu || !imuArrow) return;
      const {roll, pitch, yaw} = imu;

      // Build quaternion from roll/pitch/yaw
      const euler = new THREE.Euler(roll, pitch, yaw, 'XYZ');
      const quat = new THREE.Quaternion().setFromEuler(euler);

      const dir = new THREE.Vector3(1,0,0).applyQuaternion(quat).normalize();
      imuArrow.setDirection(dir);
    }

    function intensityToColor(intensity) {
      const t0 = performance.now();
      // normalize 0..255 → 0..1
      const t = Math.max(0, Math.min(1, intensity / 255.0));
      const t2 = performance.now();
      // simple "jet" style colormap: blue → cyan → yellow → red
      const r = t < 0.5 ? 0 : (t - 0.5) * 2;
      const g = t < 0.5 ? t * 2 : (1 - (t - 0.5) * 2);
      const b = t < 0.5 ? 1 - t * 2 : 0;
      const t1 = performance.now();
      //console.log(`intensityToColor: start ${(t1-t0).toFixed(1)} ms, finish ${(t2-t1).toFixed(1)} ms`);
      return [r, g, b];
      
    }

    function updateLidar(frameOrPoints) {
      // Accept either {count, data: Float32Array XYZI} or [[x,y,z,i], ...]
      if (!frameOrPoints) return;

      const t0 = performance.now();
      let N, vertices, colors;

      if (frameOrPoints.data instanceof Float32Array && Number.isInteger(frameOrPoints.count)) {
        // Fast path: zero-copy typed array from WebSocket binary frame
        const payload = frameOrPoints.data; // interleaved XYZI
        N = frameOrPoints.count | 0;

        vertices = new Float32Array(N * 3);
        colors   = new Float32Array(N * 3);

        for (let i = 0, k = 0, vi = 0; i < N; i++, vi += 3) {
          const x = payload[k++], y = payload[k++], z = payload[k++], intensity = payload[k++];

          // Rotate axes as you already do (adjust if your convention changes)
          const y2 = z, z2 = -y;

          vertices[vi + 0] = x;
          vertices[vi + 1] = y2;
          vertices[vi + 2] = z2;

          const t = Math.max(0, Math.min(1, intensity / 255));
          colors[vi + 0] = t;
          colors[vi + 1] = 1 - t;
          colors[vi + 2] = 0.5;
        }
      } else if (Array.isArray(frameOrPoints)) {
        // Legacy fallback: array of [x,y,z,i]
        const pts = frameOrPoints;
        N = pts.length;

        vertices = new Float32Array(N * 3);
        colors   = new Float32Array(N * 3);

        for (let i = 0, vi = 0; i < N; i++, vi += 3) {
          let [x, y, z, intensity] = pts[i];

          const y2 = z, z2 = -y;
          vertices[vi + 0] = x;
          vertices[vi + 1] = y2;
          vertices[vi + 2] = z2;

          const t = Math.max(0, Math.min(1, intensity / 255));
          colors[vi + 0] = t;
          colors[vi + 1] = 1 - t;
          colors[vi + 2] = 0.5;
        }
      } else {
        return; // unknown input shape
      }

      lidarGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
      lidarGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      lidarGeometry.computeBoundingSphere();

      if (!useWebGPU && lidarMaterial.uniforms) {
        lidarMaterial.uniforms.pointSize.value = 0.01;
        lidarMaterial.needsUpdate = true;
      }

      const t1 = performance.now();
      // console.log(`updateLidar: ${(t1 - t0).toFixed(1)} ms for ${N} points`);
      
      Object.assign(window, {
        // record/playback + stats actions that are used in HTML attributes:
          loadCameras, 
        showFrame
         });

    }


    
  window.updateLidar = updateLidar;
  window.updateIMU = updateIMU;
  initLidar();


  function makeJsonTile(id, label, objOrText) {                   
    const tile = document.createElement('div');
    tile.className = 'tile';
    tile.id = `tile-${label.toLowerCase()}-${id}`;
    const pretty = (typeof objOrText === 'string')
      ? objOrText
      : JSON.stringify(objOrText, null, 2);
    tile.innerHTML = `
      <header><h3>${id}</h3><span class="pill">${label}</span></header>
      <div class="meta" style="max-height:260px; overflow:auto;">
        <pre style="margin:0; white-space:pre-wrap;">${pretty ? escapeHtml(pretty) : '—'}</pre>
      </div>
    `;
    return tile;
  }

  function makeImageTile(kind, id, b64, label) {                      // NEW
    const tile = document.createElement('div');
    tile.className = 'tile';
    tile.id = `tile-${kind}-${id}`;
    tile.innerHTML = `
      <header><h3>${id}</h3><span class="pill">${label}</span><span class="fs-hint">click image to fullscreen</span></header>
      <div class="imgwrap">
        ${b64 ? `<img id="img-${kind}-${id}" data-camid="${id}" src="data:image/jpeg;base64,${b64}" />`
               : `<div class="meta">Waiting for frames…</div>`}
      </div>
      <div class="orbit-wrap"><div class="glow-ball"></div></div>
    `;
    const img = tile.querySelector('img');
    if (img) img.onclick = () => toggleFullscreen(tile);
    return tile;
  }

  function escapeHtml(s) {                                            // NEW
    return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function ensureJsonTile(gridEl, id, label, dataObj) {               // NEW
    const tid = `tile-${label.toLowerCase()}-${id}`;
    let tile = document.getElementById(tid);
    if (!tile) {
      tile = makeJsonTile(id, label, dataObj);
      gridEl.appendChild(tile);
    } else {
      const pre = tile.querySelector('pre');
      const pretty = (typeof dataObj === 'string') ? dataObj : JSON.stringify(dataObj, null, 2);
      if (pre) pre.textContent = pretty || '—';
    }
    return tile;
  }

  function ensureImageTile(gridEl, kind, id, label, b64) {
    const tid = `tile-${kind}-${id}`;
    let tile = document.getElementById(tid);
    if (!tile) {
      tile = makeImageTile(kind, id, b64, label);
      gridEl.appendChild(tile);
    } else if (b64) {
      const img = tile.querySelector('img');
      const newSrc = `data:image/jpeg;base64,${b64}`;
      if (img && img.src !== newSrc) img.src = newSrc;
    }
    return tile;
  }


  function toggleFullscreen(tile) {
    const el = tile;
    if (!document.fullscreenElement) {
      if (el.requestFullscreen) el.requestFullscreen();
      else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
    } else {
      if (document.exitFullscreen) document.exitFullscreen();
      else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
    }
  }
  function makeTile(id, label, b64) {
    const kind = (label || '').toLowerCase(); // "raw" or "annotated"
    const tile = document.createElement('div');
    tile.className = 'tile';
    tile.id = `tile-${kind}-${id}`;
    tile.innerHTML = `
      <header><h3>${id}</h3><span class="pill">${label}</span><span class="fs-hint">click image to fullscreen</span></header>
      <div class="imgwrap">
        ${b64 ? `<img id="img-${kind}-${id}" data-camid="${id}" src="data:image/jpeg;base64,${b64}" />`
              : `<div class="meta">Waiting for frames…</div>`}
      </div>
      <div class="orbit-wrap">
        <div class="glow-ball"></div>
      </div>
    `;
    const img = tile.querySelector('img');
    if (img) img.onclick = () => toggleFullscreen(tile);
    return tile;
  }


    // Tabs
    document.querySelectorAll('.tab').forEach(t => {
      t.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
        document.querySelectorAll('.section').forEach(x => x.classList.remove('active'));
        t.classList.add('active');
        document.getElementById(t.dataset.target).classList.add('active');
      });
    });

    let ws;

    async function handleMessage(event) {

      const t0 = performance.now();

      // Robust payload size for string / ArrayBuffer / Blob
      const size =
        (typeof event.data === "string")
          ? event.data.length
          : (event.data instanceof ArrayBuffer)
              ? event.data.byteLength
              : (event.data && typeof event.data.size === "number")
                  ? event.data.size
                  : 0;
      const payloadMB = (size / 1e6).toFixed(2);

      // ---- TEXT frames (JSON) ---------------------------------------------------
      if (typeof event.data === "string") {
        let msg;
        try {
          msg = JSON.parse(event.data);
        } catch (e) {
          console.error("Bad JSON frame:", e, event.data);
          return;
        }

        switch (msg.type) {
          case "meta": {
            // Build camera list from new envelope
            const cams = new Set(msg.known_cameras || []);
            Object.keys(msg.detections || {}).forEach(id => cams.add(id));
            const camList = Array.from(cams).sort();

            // ✅ Only rebuild if changed
            if (JSON.stringify(camList) !== JSON.stringify(known_cameras)) {
              known_cameras = camList;
              loadCameraList();   // re-render tiles
            }
            updateMeta(msg, camList);

            break;
          }

          case "frame": {
            // { type:'frame', kind:'raw'|'annotated'|'df_annotated', cam, b64 }
            // Uses your existing helper which ensures tiles and updates <img>
            updateFrame(msg.kind === "df_annotated" ? "df_annotated" : msg.kind, msg.cam, msg.b64);
            break;
          }

          case "embedding": {
            // { type:'embedding', cam, data }
            const grid = document.getElementById("embGrid");
            if (grid) ensureJsonTile(grid, msg.cam, "EMBEDDINGS", msg.data);
            break;
          }

          default:
            console.warn("Unknown WS msg.type:", msg?.type, msg);
        }

        const t1 = performance.now();
        //console.log(`onmessage(text): payload ${payloadMB} MB | handled in ${(t1 - t0).toFixed(1)} ms`);
        return;
      }

      // ---- Binary frames --------------------------------------------------------
      // Fast path if server set binaryType='arraybuffer'
      if (event.data instanceof ArrayBuffer) {
        handleLidarArrayBuffer(event.data);
        return;
      }

      // ---- Blob fallback (e.g., Safari) ----------------------------------------
      if (event.data instanceof Blob) {
        const buf = await event.data.arrayBuffer();
        handleLidarArrayBuffer(buf);
        return;
      }

      console.warn("Unknown WebSocket frame type:", typeof event.data, event.data);
    }

    function connectWS() {
      ws = new WebSocket("wss://" + location.host + "/ws");
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        console.log("✅ WebSocket connected");
      };

      ws.onclose = () => {
        console.warn("⚠️ WebSocket closed, retrying in 3s...");
        setTimeout(connectWS, 3000); // reconnect after 3s
      };

      ws.onerror = (err) => {
        console.error("WebSocket error", err);
        ws.close(); // triggers onclose → reconnect
      };

      ws.onmessage = (event) => {
        // 👇 Keep your existing onmessage handler here
        handleMessage(event);
      };
    }


    // Call once on load
    connectWS();

    ws.binaryType = "arraybuffer";

    function ensureTile(gridEl, id, label, initialB64=null) {
      const kind = label.toLowerCase();
      const tileId = `tile-${kind}-${id}`;
      let tile = document.getElementById(tileId);
      if (!tile) {
        tile = makeTile(id, label, initialB64); // use initial frame if available
        gridEl.appendChild(tile);
      }
      return tile;
    }


    function updateImage(kind, camId, b64) {
      if (!b64) return;

      const newSrc = `data:image/jpeg;base64,${b64}`;
      let img = document.getElementById(`img-${kind}-${camId}`);

      if (!img) {
        // no <img> yet -> replace the placeholder in this tile
        const tile = document.getElementById(`tile-${kind}-${camId}`);
        if (!tile) return;
        const wrap = tile.querySelector('.imgwrap');
        wrap.innerHTML = `<img id="img-${kind}-${camId}" data-camid="${camId}" />`;
        img = document.getElementById(`img-${kind}-${camId}`);
        img.onclick = () => toggleFullscreen(tile);
      }

      if (img.src !== newSrc) img.src = newSrc;
    }

    function updateMeta(msg, camList) {
      const T = () => performance.now();
      const m0 = T();

      // ---- Detections tab
      const detWrap = document.getElementById('detWrap');
      detWrap.innerHTML = '';
      if (msg.detections) {
        camList.forEach(id => {
          const dets = msg.detections[id] || [];
          const div = document.createElement('div');
          div.className = 'tile';
          div.innerHTML = `
            <header><h3>${id}</h3><span class="pill">DETECTIONS</span></header>
            <div class="meta">${
              dets.length
                ? dets.map(d => `${d.class} (${(d.score*100).toFixed(1)}%) [${d.bbox.map(x=>x.toFixed(1)).join(", ")}]`).join("<br>")
                : "No detections"
            }</div>
          `;
          detWrap.appendChild(div);
        });
      }
      const m1 = T();
      //console.log(`updateMeta: detections ${(m1-m0).toFixed(1)} ms`);


      const m2 = T();
      //console.log(`updateMeta: alerts ${(m2-m1).toFixed(1)} ms`);

      // ---- Lidar IMU
      if (msg.lidar_imu) {
        const im0 = T();
        updateIMU(msg.lidar_imu);
        const im1 = T();
        //console.log(`updateMeta: imu ${(im1-im0).toFixed(1)} ms`);
      }

      // ---- Lidar loss overlay
      if (typeof msg.lidar_loss !== "undefined") {
        const ov0 = T();
        document.getElementById("lidarOverlay").textContent = "Loss: " + msg.lidar_loss;
        const ov1 = T();
        //console.log(`updateMeta: overlay ${(ov1-ov0).toFixed(1)} ms`);
      }

      const m3 = T();
      //console.log(`updateMeta: total ${(m3-m0).toFixed(1)} ms`);
    }

    // Before connecting:
    ws.binaryType = "arraybuffer";

    const LIDAR_MAGIC = "LDR1";
    function handleLidarArrayBuffer(buf) {
      if (!(buf instanceof ArrayBuffer)) {
        console.warn("LIDAR handler expected ArrayBuffer");
        return;
      }
      if (buf.byteLength < 8) {
        console.warn("LIDAR frame too small:", buf.byteLength);
        return;
      }
      const view = new DataView(buf);
      const magic =
        String.fromCharCode(view.getUint8(0)) +
        String.fromCharCode(view.getUint8(1)) +
        String.fromCharCode(view.getUint8(2)) +
        String.fromCharCode(view.getUint8(3));

      if (magic !== LIDAR_MAGIC) {
        console.warn("Unknown binary frame magic:", magic);
        return;
      }
      const count = view.getUint32(4, true); // little-endian
      const expectedBytes = 8 + count * 16; // 4 floats (16 bytes) per point
      if (buf.byteLength !== expectedBytes) {
        console.warn("LIDAR payload length mismatch", {
          got: buf.byteLength,
          expected: expectedBytes,
          count
        });
        return;
      }
      // Float32Array view over payload (starts after 8-byte header)
      const payload = new Float32Array(buf, 8); // length = count * 4
      updateLidar({ count, data: payload });
    }



    function updateFrame(kind, cam, b64) {
      const grid =
        kind === 'raw'         ? document.getElementById('rawGrid') :
        kind === 'annotated'   ? document.getElementById('annGrid') :
        null;

      if (!grid) return;
      const tile = ensureTile(grid, cam, kind.toUpperCase(), b64);
      updateImage(kind, cam, b64);
    }


    async function loadSessions() {
      const sel  = document.getElementById('sessionSel');
      const sSel = document.getElementById('statsSessionSel');
      const cSel = document.getElementById('cameraSel');

      sel.innerHTML = '';
      sSel.innerHTML = '';
      if (cSel) cSel.innerHTML = '';

      try {
        const res = await fetch('/api/sessions');
        if (!res.ok) throw new Error(`GET /api/sessions -> ${res.status}`);
        const data = await res.json();
        const sessions = Array.isArray(data?.sessions) ? data.sessions : [];

        // populate selects
        sessions.forEach(s => {
          const o = document.createElement('option'); o.value = s; o.textContent = s; sel.appendChild(o);
          const o2 = document.createElement('option'); o2.value = s; o2.textContent = s; sSel.appendChild(o2);
        });

        if (sessions.length > 0) {
          sel.value = sessions[sessions.length - 1];
          sSel.value = sel.value;
          await loadCameras();   // only call when we actually have a selected session
        } else {
          // no sessions -> clear playback UI gracefully
          const slider = document.getElementById('timeline');
          const playImg = document.getElementById('playImg');
          const frameTime = document.getElementById('frameTime');
          if (slider) { slider.min = 0; slider.max = 0; slider.value = 0; }
          if (playImg) playImg.src = '';
          if (frameTime) frameTime.textContent = 'Time: —';
          const hits = document.getElementById('hits');
          if (hits) hits.textContent = 'No sessions found yet.';
        }
      } catch (err) {
        console.error('loadSessions failed:', err);
      }
    }

    async function loadCameras() {
      const ses  = document.getElementById('sessionSel')?.value || '';
      const csel = document.getElementById('cameraSel');
      if (!csel) return;

      csel.innerHTML = '';

      if (!ses) {
        console.warn('No session selected; skipping /api/session/{session}/cameras');
        return; // nothing to load yet
      }

      try {
        const url = `/api/session/${encodeURIComponent(ses)}/cameras`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`GET ${url} -> ${res.status}`);
        const data = await res.json();
        const cams = Array.isArray(data?.cameras) ? data.cameras : [];

        cams.forEach(c => {
          const o = document.createElement('option'); o.value = c; o.textContent = c; csel.appendChild(o);
        });

        if (cams.length > 0) {
          csel.value = cams[0];
          await loadTimeline();
        } else {
          // clear timeline if no cameras in this session
          const slider = document.getElementById('timeline');
          const playImg = document.getElementById('playImg');
          const frameTime = document.getElementById('frameTime');
          if (slider) { slider.min = 0; slider.max = 0; slider.value = 0; }
          if (playImg) playImg.src = '';
          if (frameTime) frameTime.textContent = 'Time: —';
        }
      } catch (err) {
        console.error('loadCameras failed:', err);
      }
    }

    let frameIndex = 0;
    let frameList = [];



    async function showFrame() {
      const slider = document.getElementById('timeline');
      frameIndex = parseInt(slider.value || '0');
      const meta = frameList[frameIndex];
      const img = document.getElementById('playImg');
      const tdiv = document.getElementById('frameTime');
      if (!meta) { img.src=''; tdiv.textContent='Time: —'; return; }
      img.src = '/media?path='+encodeURIComponent(meta.path);
      tdiv.textContent = 'Time: ' + new Date(meta.ts*1000).toLocaleString();
    }



    function drawChart(data) {
      // data: { camera1: {classA: count, classB: count}, camera2: {...} }
      const svg = document.getElementById('chart');
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      const cams = Object.keys(data).sort();
      const classes = Array.from(new Set(cams.flatMap(c => Object.keys(data[c])))).sort();

      const pad = 40, W = svg.clientWidth||800, H = svg.clientHeight||320;
      const innerW = W - pad*2, innerH = H - pad*2;

      const camCount = Math.max(1, cams.length);
      const classCount = Math.max(1, classes.length);
      const groupW = innerW / camCount;
      const barW = Math.max(6, groupW / (classCount+1));
      const maxVal = Math.max(1, Math.max(...cams.map(c => Math.max(1, ...Object.values(data[c]||{_0:0})))));
      const scaleY = v => pad + innerH - (v / maxVal) * innerH;

      // axes
      const axis = (x1,y1,x2,y2) => {
        const l = document.createElementNS("http://www.w3.org/2000/svg", "line");
        l.setAttribute("x1", x1); l.setAttribute("y1", y1);
        l.setAttribute("x2", x2); l.setAttribute("y2", y2);
        l.setAttribute("stroke", "#3b4254");
        svg.appendChild(l);
      };
      axis(pad, pad, pad, pad+innerH);
      axis(pad, pad+innerH, pad+innerW, pad+innerH);

      // bars
      cams.forEach((cam, ci) => {
        classes.forEach((cls, ki) => {
          const v = (data[cam] && data[cam][cls]) ? data[cam][cls] : 0;
          const x = pad + ci*groupW + ki*barW + 4;
          const y = scaleY(v);
          const h = pad + innerH - y;
          const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          r.setAttribute("x", x); r.setAttribute("y", y);
          r.setAttribute("width", barW-6); r.setAttribute("height", h);
          r.setAttribute("fill", "#4b5563");
          svg.appendChild(r);
        });
        // cam label
        const tx = document.createElementNS("http://www.w3.org/2000/svg", "text");
        tx.setAttribute("x", pad + ci*groupW + groupW/2);
        tx.setAttribute("y", pad + innerH + 16);
        tx.setAttribute("text-anchor", "middle");
        tx.setAttribute("fill", "#a1a7b5");
        tx.setAttribute("font-size", "11");
        tx.textContent = cam;
        svg.appendChild(tx);
      });

      // simple legend
      classes.forEach((cls, i) => {
        const lx = pad + i*100, ly = 16;
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", lx); rect.setAttribute("y", ly);
        rect.setAttribute("width", 14); rect.setAttribute("height", 10);
        rect.setAttribute("fill", "#4b5563");
        svg.appendChild(rect);
        const lt = document.createElementNS("http://www.w3.org/2000/svg", "text");
        lt.setAttribute("x", lx+20); lt.setAttribute("y", ly+10);
        lt.setAttribute("fill", "#cbd5e1");
        lt.setAttribute("font-size", "12");
        lt.textContent = cls;
        svg.appendChild(lt);
      });
    }
    // init
    loadSessions();
    
  </script>

</body>
</html>
""")

# ---------------- WebSocket: periodic live push ----------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            now = time.time()
            motion = {cid: (now < _motion_until.get(cid, 0)) for cid in known_cameras}

            # 1. Send light JSON metadata (fast to parse)
            meta = {
                "type": "meta",
                "known_cameras": list(set(known_cameras) | set(latest_raw_frames.keys()) | set(latest_ann_frames.keys())),
                "detections": latest_detections,
                "lidar_imu": latest_lidar_imu,
                "lidar_loss": latest_lidar_loss,
            }
            await ws.send_text(json.dumps(meta))

            # 2. Send raw frames separately
            for cam_id, b64 in latest_raw_frames.items():
                if b64:
                    await ws.send_json({
                        "type": "frame",
                        "kind": "raw",
                        "cam": cam_id,
                        "b64": b64
                    })

            # 3. Send annotated frames
            for cam_id, b64 in latest_ann_frames.items():
                if b64:
                    await ws.send_json({
                        "type": "frame",
                        "kind": "annotated",
                        "cam": cam_id,
                        "b64": b64
                    })

            # 5. Send embeddings & DF analysis (JSON objects, not base64)
            for cam_id, obj in latest_embeddings.items():
                await ws.send_json({
                    "type": "embedding",
                    "cam": cam_id,
                    "data": obj
                })

            # 6. Send lidar separately (single binary frame)
            pts = latest_lidar_points
            if isinstance(pts, np.ndarray) and pts.size:
                try:
                    await ws.send_bytes(pack_lidar_frame(pts))
                except Exception as e:
                    # Optional: throttle log/ignore transient disconnects
                    # logger.warning(f"ws lidar send failed: {e}")
                    pass

            await asyncio.sleep(0.03)
    except Exception:
        pass

@app.post("/api/cameras/rename")
async def api_camera_rename(body: Dict[str, Any] = Body(...)):
    """
    Body: { "current_id": "7845582FD303_0", "new_name": "barn_cam" }
    """
    node = app.state.ros_node
    cli = node.create_client(RenameCamera, "rename_camera")

    if not cli.wait_for_service(timeout_sec=2.0):
        return JSONResponse({"ok": False, "error": "rename_camera service not available"}, status_code=503)

    req = RenameCamera.Request()
    req.current_id = body.get("current_id", "")
    req.new_name = body.get("new_name", "")

    future: Future = cli.call_async(req)
    await asyncio.wrap_future(future)

    if future.result() and future.result().success:
        return JSONResponse({"ok": True, "message": future.result().message})
    else:
        return JSONResponse({
            "ok": False,
            "message": getattr(future.result(), "message", "unknown error")
        }, status_code=400)


@app.get("/media")
async def api_media(path: str):
    # Basic security: ensure within RECORD_ROOT
    full = osp.realpath(path)
    root = osp.realpath(RECORD_ROOT)
    if not full.startswith(root):
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not osp.exists(full):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(full, media_type="image/jpeg")

# ---------------- Spin ROS + HTTP ----------------
def ros_thread(app_ref):
    rclpy.init()
    node = WebStreamNode()
    app_ref.state.ros_node = node
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main():
    # Resolve packaged SSL files
    ssl_root = files('camera_view') / 'ssl'
    cert_path = str(ssl_root / 'cert.pem')
    key_path  = str(ssl_root / 'key.pem')

    #  allow override via env vars
    cert_path = osp.abspath(os.environ.get('WVW_CERT', cert_path))
    key_path  = osp.abspath(os.environ.get('WVW_KEY',  key_path))

    t = threading.Thread(target=ros_thread, args=(app,), daemon=True)
    t.start()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=1080,
        ssl_certfile=cert_path,
        ssl_keyfile=key_path,
    )

if __name__ == "__main__":
    main()
