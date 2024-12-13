import cv2
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.get_logger().info("CameraNode initialized.")
        
        # Publisher for camera images
        self.image_publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        self.bridge = CvBridge()

        # Timer to capture and publish images at a fixed rate
        self.timer = self.create_timer(0.1, self.capture_and_publish_image)  # Adjust the frequency as needed
        self.get_logger().info("CameraNode is ready to capture and publish images.")

    def capture_and_publish_image(self):
        """
        Capture an image using the webcam and publish it as a ROS topic.
        """
        try:
            cam = cv2.VideoCapture(0)  # Access the default camera
            if not cam.isOpened():
                self.get_logger().error("Unable to access the camera")
                return

            # Capture a single frame
            ret, frame = cam.read()
            cam.release()

            if not ret:
                self.get_logger().error("Failed to capture image")
                return

            # Publish the image
            self.publish_image(frame)

        except Exception as e:
            self.get_logger().error(f"Error capturing image: {e}")

    def publish_image(self, frame):
        """
        Publish the captured image to the ROS topic.
        """
        try:
            # Convert the OpenCV image to a ROS Image message
            image_message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.image_publisher.publish(image_message)
            self.get_logger().info("Image published successfully.")
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
