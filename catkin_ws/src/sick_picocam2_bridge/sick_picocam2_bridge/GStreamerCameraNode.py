#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class GStreamerCameraNode(Node):
    def __init__(self, camera_name, topic_name, node_name, fps=10):
        super().__init__(node_name)
        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()
        self.fps = fps

        self.pipeline = (
            f"aravissrc camera-name={camera_name} ! "
            "videoconvert ! "
            "appsink drop=true sync=false max-buffers=1"
        )

        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera: {camera_name}")

        # FPS tracking
        self.last_time = time.time()
        self.frame_count = 0

        # Timer to pull frames
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)

    def timer_callback(self):



        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame grab failed")
            return
        # If the camera gave grayscale (single channel), convert to BGR
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self.frame_count += 1
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher_.publish(msg)

        # Log FPS every 5 seconds
        now = time.time()
        if now - self.last_time >= 5.0:
            fps = self.frame_count / (now - self.last_time)
            self.get_logger().info(
                f"{self.get_name()} publishing at {fps:.2f} FPS, shape={frame.shape}"
            )
            self.last_time = now
            self.frame_count = 0

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):



    rclpy.init(args=args)

    cam1 = GStreamerCameraNode(
        "SICK-I2D303C-2RCA11-0025310004",
        "camera1/image_raw",
        "gstreamer_camera_node_1",
        fps=10,
    )
    cam2 = GStreamerCameraNode(
        "SICK-I2D303C-2RCA11-0025310021",
        "camera2/image_raw",
        "gstreamer_camera_node_2",
        fps=10,
    )

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(cam1)
    executor.add_node(cam2)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        cam1.destroy_node()
        cam2.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
