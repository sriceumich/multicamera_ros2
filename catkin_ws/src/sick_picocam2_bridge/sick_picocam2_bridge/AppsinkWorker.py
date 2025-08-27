import gi, time
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp

import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header

Gst.init(None)

class AppsinkWorker(threading.Thread):
    def __init__(self, node, topic_ns, frame_id, pipeline_str, stats_interval=5.0):
        super().__init__(daemon=True)
        self.node = node
        self.topic_ns = topic_ns
        self.frame_id = frame_id
        self.stats_interval = stats_interval

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        self.pub_img = node.create_publisher(Image, f"{topic_ns}/image_raw", qos)
        self.pub_info = node.create_publisher(CameraInfo, f"{topic_ns}/camera_info", qos)

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("appsink0")
        if self.appsink is None:
            raise RuntimeError(f"{topic_ns}: appsink0 not found")

        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.connect("new-sample", self.on_new_sample)

        self._stop_evt = threading.Event()
        self.frames_pub = 0
        self.last_stat_t = time.time()

    def stop(self):
        self._stop_evt.set()

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        width = s.get_value('width')
        height = s.get_value('height')
        fmt = s.get_string('format')  # e.g. BGR, RGB, GRAY8
        fmt_map = {
            "BGR": "bgr8",
            "RGB": "rgb8",
            "GRAY8": "mono8",
            "GRAY16_LE": "mono16",
        }

        ros_encoding = fmt_map.get(fmt, "bgr8")  # default fallback

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        try:
            # Build ROS Image directly
            msg = Image()
            msg.header = Header()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.height = height
            msg.width = width
            msg.encoding = ros_encoding
            msg.is_bigendian = 0
            msg.step = width * (len(mapinfo.data) // (width * height))
            msg.data = mapinfo.data  # already bytes
            self.pub_img.publish(msg)

            cinfo = CameraInfo()
            cinfo.header = msg.header
            cinfo.width = width
            cinfo.height = height
            self.pub_info.publish(cinfo)

            self.frames_pub += 1
            if (time.time() - self.last_stat_t) >= self.stats_interval:
                self.node.get_logger().info(
                    f"[{self.topic_ns}] frames_pub={self.frames_pub}, caps={width}x{height}, fmt={fmt}"
                )
                self.frames_pub = 0
                self.last_stat_t = time.time()
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def run(self):
        self.node.get_logger().info(f"[{self.topic_ns}] starting pipeline")
        self.pipeline.set_state(Gst.State.PLAYING)
        bus = self.pipeline.get_bus()
        while not self._stop_evt.is_set():
            msg = bus.timed_pop_filtered(100*Gst.MSECOND,
                                         Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    self.node.get_logger().error(f"[{self.topic_ns}] GST error: {err} / {dbg}")
                    break
                elif msg.type == Gst.MessageType.EOS:
                    break
        self.pipeline.set_state(Gst.State.NULL)
        self.node.get_logger().info(f"[{self.topic_ns}] stopped")
