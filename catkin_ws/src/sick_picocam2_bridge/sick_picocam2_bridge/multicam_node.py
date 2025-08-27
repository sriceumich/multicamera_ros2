#!/usr/bin/env python3

import os
import threading


# Must be set before importing cv2 / initializing GStreamer
os.environ.setdefault("GST_DEBUG", "3,rtsp*:6,rtspsrc*:6")
os.environ.setdefault("GST_DEBUG_NO_COLOR", "1")
# Write to a file so ROS logging does not swallow messages
os.environ.setdefault("GST_DEBUG_FILE", "/tmp/gst.log")
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)
from .AppsinkWorker import AppsinkWorker

import re
import socket
import time
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.parameter import ParameterValue

try:
    from cv_bridge import CvBridge
except ImportError:
    raise RuntimeError("cv_bridge not found. Install: sudo apt install ros-$ROS_DISTRO-cv-bridge")

# ---------------------- helpers: naming & rtsp probing ----------------------

def sanitize_name(name: str) -> str:
    """
    Make a ROS topic/TF-friendly name:
    - lowercase
    - spaces -> underscores
    - remove invalid chars
    - ensure starts with a letter
    """
    name = name.strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]+", "", name)
    if not name or not name[0].isalpha():
        name = f"cam_{name}" if name else "cam"
    return name

def fix_scheme(uri: str) -> str:
    """
    Normalize common RTSP scheme typos:
    - 'rtsp:/host'      -> 'rtsp://host'
    - 'rtsp//host'      -> 'rtsp://host'
    - 'rtsps//host'     -> 'rtsps://host'
    - tolerate missing colon (e.g. 'rtsps//...') by inserting it
    Also trims spaces.
    """
    u = uri.strip()

    # If it already looks fine, keep it
    if re.match(r"^(rtsp|rtsps)://", u, flags=re.IGNORECASE):
        return u

    # Missing colon? e.g. 'rtsps//host'
    m = re.match(r"^(rtsp|rtsps)//(.+)$", u, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()}://{m.group(2)}"

    # Single slash after colon? e.g. 'rtsp:/host'
    u = re.sub(r"^(rtsp[s]?):/([^/])", r"\1://\2", u, flags=re.IGNORECASE)

    # No slashes after colon? e.g. 'rtsp:host'
    u = re.sub(r"^(rtsp[s]?):(?!//)", r"\1://", u, flags=re.IGNORECASE)

    return u

def _read_exact(sock: socket.socket, n: int, timeout: float) -> bytes:
    sock.settimeout(timeout)
    chunks = []
    remaining = n
    while remaining > 0:
        b = sock.recv(remaining)
        if not b:
            break
        chunks.append(b)
        remaining -= len(b)
    return b"".join(chunks)

def gst_rtsp_appsink(uri: str, latency_ms=50, hw=None, width=None, height=None, fps=10):
    dec = "nvh264dec" if hw == "nvidia" else "avdec_h264"
    caps_parts = []
    if width and height:
        caps_parts.append(f"width={width},height={height}")
    if fps:
        caps_parts.append(f"framerate={fps}/1")
    caps = "" if not caps_parts else " ! video/x-raw," + ",".join(caps_parts)
    

    return (
        f"rtspsrc location={uri} protocols=tcp latency={latency_ms} ! "
        "rtph264depay ! h264parse ! "
        f"{dec} ! videoconvert ! video/x-raw,format=BGR ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "appsink name=appsink0 emit-signals=true sync=false max-buffers=1 drop=true"
    )

def gst_aravis_appsink(device: str, width=None, height=None, fps=10,
                       packet_size=8192, packet_delay_ns=1000):
    caps_parts = []
    if width and height:
        caps_parts.append(f"width={width},height={height}")
    if fps:
        caps_parts.append(f"framerate={fps}/1")
    caps = "" if not caps_parts else ",".join(caps_parts)

    return (
        f"aravissrc camera-name={device} packet-size={packet_size} ! "
        f"capsfilter caps=video/x-raw,{caps if caps else 'format=BGR'} ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=appsink0 emit-signals=true sync=false max-buffers=1 drop=true"
    )

def probe_rtsp_name(uri: str, timeout: float = 2.0) -> Optional[str]:
    """
    Minimal RTSP DESCRIBE to fetch SDP and extract 's=' session name.
    Returns None on any failure.
    """
    try:
        uri = fix_scheme(uri)
        parsed = urlparse(uri)
        if parsed.scheme not in ("rtsp", "rtsps"):
            return None

        host = parsed.hostname
        port = parsed.port or (322 if parsed.scheme == "rtsps" else 554)  # default RTSP=554; rtsps often 322/5554/etc
        path = parsed.geturl()  # full URI for DESCRIBE line

        # TLS is out-of-scope here; most UniFi RTSP are plain rtsp on LAN.
        # Use TCP socket:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            req = (
                f"DESCRIBE {uri} RTSP/1.0\r\n"
                f"CSeq: 1\r\n"
                f"User-Agent: multi-rtsp-cam/1.0\r\n"
                f"Accept: application/sdp\r\n"
                f"\r\n"
            ).encode("ascii")
            sock.sendall(req)

            sock.settimeout(timeout)
            # Read headers
            header_bytes = b""
            while b"\r\n\r\n" not in header_bytes:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                header_bytes += chunk
                # safety guard: header shouldn't be huge
                if len(header_bytes) > 65536:
                    break

            header, _, rest = header_bytes.partition(b"\r\n\r\n")
            if not header:
                return None

            # parse status
            if not header.startswith(b"RTSP/1.0 200"):
                # not OK; bail
                return None

            # content length
            m = re.search(br"Content-Length:\s*(\d+)", header, flags=re.IGNORECASE)
            length = int(m.group(1)) if m else 0

            sdp = rest
            if length and len(rest) < length:
                sdp += _read_exact(sock, length - len(rest), timeout)

            # SDP text
            try:
                sdp_text = sdp.decode("utf-8", errors="ignore")
            except Exception:
                return None

            # find 's=' line
            for line in sdp_text.splitlines():
                line = line.strip()
                if line.startswith("s="):
                    name = line[2:].strip()
                    if name:
                        return name
            return None
    except Exception:
        return None

def name_from_uri(uri: str) -> str:
    """Fallback name: last path segment or full host-port if no path."""
    uri = fix_scheme(uri)
    parsed = urlparse(uri)
    last = (parsed.path or "").rstrip("/").split("/")[-1]
    if last:
        return last
    host = parsed.hostname or "camera"
    port = parsed.port
    return f"{host}_{port}" if port else host

def is_aravis_uri(u: str) -> bool:
    u = u.strip()
    return u.lower().startswith(("aravis:", "gige:", "mac:"))

def open_capture_from_uri(uri: str, use_gst: bool, latency_ms: int,
                          hwaccel: Optional[str], width: Optional[int],
                          height: Optional[int], fps: Optional[int]):
    flags = backend_flags(use_gst)
    if is_aravis_uri(uri):
        # Strip scheme if present to feed camera-name=…
        device = uri.split(":", 1)[1] if ":" in uri else uri
        pipe = build_aravis_pipeline(device=device, width=width, height=height, fps=fps)
        return cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    # otherwise fall back to RTSP/H264
    pipe = build_gst_pipeline(uri, latency_ms, hwaccel, width, height, fps) if use_gst else uri
    return cv2.VideoCapture(pipe, flags)

def build_aravis_pipeline(device: str,
                          width: Optional[int] = None,
                          height: Optional[int] = None,
                          fps: Optional[int] = None,
                          to_bgr: bool = True,
                          packet_size: int = 9000) -> str:
    """
    device can be:
      - 'GigE:192.168.0.102'
      - 'MAC:00:11:1C:AA:BB:CC'
      - the camera name shown by `arv-tool-0.8`
    Requires gstreamer + aravis plugins installed.
    """
    caps = []
    if width and height:
        caps.append(f"width={width},height={height}")
    if fps:
        caps.append(f"framerate={fps}/1")
    caps_str = ""
    if caps:
        caps_str = " ! video/x-raw," + ",".join(caps)

    convert = "videoconvert ! video/x-raw,format=BGR" if to_bgr else "videoconvert"

    # Drop/sync=false keeps latency small; tune packet-timeout if needed.
    # You can also set e.g. exposure via 'camera-gv-feature=value' style properties (advanced).
    pipeline = (
        f"aravissrc camera-name={device} packet-size=8192 ! "
        f"video/x-raw{caps_str},width={width},height={height}  ! {convert} ! {caps} !"
        "appsink emit-signals=false sync=false max-buffers=1 drop=true"
    )
    return pipeline


def build_gst_pipeline(uri: str, latency_ms: int = 50,
                       hw: Optional[str] = None,
                       width: Optional[int] = None,
                       height: Optional[int] = None,
                       fps: Optional[int] = None) -> str:
    if hw == "nvidia":
        decoder = "nvh264dec"
        convert = "videoconvert ! video/x-raw,format=BGR"
    else:
        decoder = "avdec_h264"
        convert = "videoconvert"

    caps = []
    if width and height:
        caps.append(f"width={width},height={height}")
    if fps:
        caps.append(f"framerate={fps}/1")

    pipeline = (
        f"aravissrc location={uri} protocols=tcp latency={latency_ms} ! "
        "application/x-rtp,media=video,encoding-name=H264 ! {caps} !"
        "rtph264depay ! h264parse ! "
        f"{decoder} ! {convert} ! "
        "appsink emit-signals=false sync=false max-buffers=1 drop=true"
    )
    return pipeline

def backend_flags(use_gst: bool) -> int:
    if use_gst:
        return cv2.CAP_GSTREAMER
    return cv2.CAP_FFMPEG if hasattr(cv2, "CAP_FFMPEG") else 0

# ---------------------- worker thread ----------------------

class CameraWorker(threading.Thread):
    def __init__(self,
                 node: Node,
                 index: int,
                 uri: str,
                 topic_ns: str,
                 use_gst: bool,
                 latency_ms: int,
                 hwaccel: Optional[str],
                 width: Optional[int],
                 height: Optional[int],
                 fps: Optional[int],
                 encoding: str,
                 drop_old_frames: bool,
                 stats_interval: float = 5.0,
                 frame_id: Optional[str] = None):
        super().__init__(daemon=True)
        self.node = node
        self.index = index
        self.uri = uri
        self.topic_ns = topic_ns
        self.use_gst = use_gst
        self.latency_ms = latency_ms
        self.hwaccel = hwaccel
        self.width = width
        self.height = height
        self.fps = fps
        self.encoding = encoding
        self.drop_old_frames = drop_old_frames
        self.stats_interval = stats_interval
        self.frame_id = frame_id or f"camera_{index}"

        qos_image = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        qos_info = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.pub_img = self.node.create_publisher(Image, f"{self.topic_ns}/image_raw", qos_image)
        self.pub_info = self.node.create_publisher(CameraInfo, f"{self.topic_ns}/camera_info", qos_info)

        self.bridge = CvBridge()
        self._stop_evt = threading.Event()
        self.cap = None
        self.frames_pub = 0
        self.frames_recv = 0
        self.last_stat_t = time.time()

    def stop(self):
        self._stop_evt.set()

    def _open_capture(self):
        if is_aravis_uri(self.uri):
            device = self.uri.split(":", 1)[1] if ":" in self.uri else self.uri
            pipe = build_aravis_pipeline(device=device, width=self.width, height=self.height, fps=fps)
            self.node.get_logger().info(f"[{self.topic_ns}] Aravis pipeline: {pipe}")
            self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        elif self.use_gst:
            pipe = build_gst_pipeline(self.uri, self.latency_ms, self.hwaccel, self.width, self.height, self.fps)
            self.node.get_logger().info(f"[{self.topic_ns}] GStreamer RTSP pipeline: {pipe}")
            self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        else:
            self.node.get_logger().info(f"[{self.topic_ns}] OpenCV backend: {self.uri}")
            self.cap = cv2.VideoCapture(self.uri, backend_flags(False))

        if not self.cap or not self.cap.isOpened():
            self.node.get_logger().warn(f"[{self.topic_ns}] Open failed, retrying in 0.5s…")
            time.sleep(0.5)
            return self._open_capture()

        # 🔎 Now log the negotiated caps
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])

        self.node.get_logger().info(
            f"[{self.topic_ns}] Negotiated caps: {w}x{h} @ {fps:.2f} FPS, codec={codec}"
        )

        # Optional width/height/fps hints
        if self.width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:    self.cap.set(cv2.CAP_PROP_FPS, self.fps)


    def _publish_camera_info(self, header: Header, w: int, h: int):
        msg = CameraInfo()
        msg.header = header
        msg.width = w
        msg.height = h

        self.pub_info.publish(msg)

    def run(self):
        try:
            self._open_capture()
            self.node.get_logger().info(f"[{self.topic_ns}] Connected.")
        except Exception as e:
            self.node.get_logger().error(f"[{self.topic_ns}] Open failed: {e}")
            return

        try:
            # Main loop
            
            while not self._stop_evt.is_set() and rclpy.ok():
                try:
                    if self.drop_old_frames and hasattr(self.cap, "grab"):
                        if not self.cap.grab():
                            time.sleep(0.01)
                            continue
                        ret, frame = self.cap.retrieve()
                    else:
                        ret, frame = self.cap.read()

                    if not ret or frame is None:
                        time.sleep(0.005)
                        continue


                    self.frames_recv += 1
                    # 👇 Debug log before publishing
                    self.node.get_logger().info(
                        f"[{self.topic_ns}] received frame #{self.frames_recv} "
                        f"size={frame.shape[1]}x{frame.shape[0]} dtype={frame.dtype}"
                    )
                    now = self.node.get_clock().now().to_msg()
                    header = Header()
                    header.stamp = now
                    header.frame_id = self.frame_id

                    if self.encoding.lower() == "rgb8":
                        # Convert to RGB if requested
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        ros_img = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
                    else:
                        # Default: publish as BGR8 (OpenCV default)
                        ros_img = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

                    ros_img.header = header

                    # Safety check: validate data layout
                    expected = ros_img.height * ros_img.step
                    if len(ros_img.data) != expected:
                        self.node.get_logger().error(
                            f"[{self.topic_ns}] Invalid Image msg: "
                            f"len(data)={len(ros_img.data)} vs expected={expected}, "
                            f"w={ros_img.width}, h={ros_img.height}, step={ros_img.step}"
                        )
                        continue  # Skip publishing bad frame

                    self.pub_img.publish(ros_img)
                    self._publish_camera_info(header, frame.shape[1], frame.shape[0])
                    self.frames_pub += 1

                    if (time.time() - self.last_stat_t) >= self.stats_interval:
                        self.node.get_logger().info(
                            f"[{self.topic_ns}] frames_recv={self.frames_recv} frames_pub={self.frames_pub}"
                        )
                        self.frames_recv = 0
                        self.frames_pub = 0
                        self.last_stat_t = time.time()

                except Exception as e:
                    self.node.get_logger().warn(f"[{self.topic_ns}] Loop error: {e}")
                    time.sleep(0.02)
        finally:
            try:
                if self.cap:
                    self.node.get_logger().info(f"[{self.topic_ns}] Releasing RTSP session...")
                    self.cap.release()
                    self.cap = None
            except Exception as e:
                self.node.get_logger().warn(f"[{self.topic_ns}] Release failed: {e}")

            self.node.get_logger().info(f"[{self.topic_ns}] Stopped.")

# ---------------------- main node ----------------------

class MultiRtspCamNode(Node):
    def __init__(self):
        super().__init__("multi_rtsp_cam_pub")

        self.workers = []

        self.declare_parameter(
            "streams",
            ParameterValue(
                string_array_value=[],
                type=ParameterType.PARAMETER_STRING_ARRAY
            ),
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY
            ),
        )
        self.declare_parameter("topic_prefix", "/cameras")
        self.declare_parameter("use_gstreamer", True)
        self.declare_parameter("latency_ms", 50)
        self.declare_parameter("hardware_accel", "")
        self.declare_parameter("width", 0)
        self.declare_parameter("height", 0)
        self.declare_parameter("fps", 0)
        self.declare_parameter("encoding", "bgr8")
        self.declare_parameter("drop_old_frames", True)
        self.declare_parameter("stats_interval", 5.0)

        streams: List[str] = list(self.get_parameter("streams").get_parameter_value().string_array_value)
        topic_prefix: str = self.get_parameter("topic_prefix").value
        use_gst: bool = bool(self.get_parameter("use_gstreamer").value)
        latency_ms: int = int(self.get_parameter("latency_ms").value)
        hwaccel: str = (self.get_parameter("hardware_accel").value or "").strip().lower() or None
        width: int = int(self.get_parameter("width").value)
        height: int = int(self.get_parameter("height").value)
        fps: int = int(self.get_parameter("fps").value)
        encoding: str = str(self.get_parameter("encoding").value)
        drop_old_frames: bool = bool(self.get_parameter("drop_old_frames").value)
        stats_interval: float = float(self.get_parameter("stats_interval").value)
        self.declare_parameter("use_ptp", False)   # default = False
        use_ptp: bool = bool(self.get_parameter("use_ptp").value)        

        # Normalize URIs (fix schemes) up front
        streams = [fix_scheme(u) for u in streams]

        if not streams:
            self.get_logger().error("No RTSP streams provided. Set the 'streams' parameter to a list of URIs.")
            raise SystemExit(2)
        if len(streams) > 32:
            self.get_logger().warn(f"Requested {len(streams)} streams.")

        # Probe names
        discovered = []
        for i, uri in enumerate(streams):
            cam_name = probe_rtsp_name(uri, timeout=2.0)
            if not cam_name:
                # fallback to last segment token
                cam_name = name_from_uri(uri)
            safe_name = sanitize_name(cam_name)
            ns = f"{topic_prefix}/{safe_name}"
            self.get_logger().info(f"Camera[{i}] uri={uri} -> name='{cam_name}' -> ns='{ns}'")
            discovered.append((uri, ns, safe_name))

        # Spin up workers
        self.workers: List[threading.Thread] = []
        for i, (uri, ns, safe_name) in enumerate(discovered):
            if is_aravis_uri(uri):
                dev = uri.split(":", 1)[1] if ":" in uri else uri
                pipe = gst_aravis_appsink(
                    device=dev,
                    width=(width or None),
                    height=(height or None),
                    fps=fps,
                    packet_size=8192,          # tune with your MTU (use 8192 with MTU=9000)
                    packet_delay_ns=2000       # stagger if multiple cameras share a NIC
                )
            else:
                pipe = gst_rtsp_appsink(
                    uri=uri,
                    latency_ms=latency_ms,
                    hw=hwaccel,
                    width=(width or None),
                    height=(height or None),
                    fps=fps
                )

            w = AppsinkWorker(
                node=self,
                topic_ns=ns,
                frame_id=safe_name,
                pipeline_str=pipe,
                stats_interval=stats_interval,
            )
            self.workers.append(w)

        self.get_logger().info(f"Starting {len(self.workers)} camera workers")
        for w in self.workers:
            w.start()
            time.sleep(0.3) 


    def destroy_node(self):
        for w in getattr(self, "workers", []):
            w.stop()
        for w in getattr(self, "workers", []):
            w.join(timeout=2.0)
        super().destroy_node()

def main():
    rclpy.init()
    node = None
    try:
        node = MultiRtspCamNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
