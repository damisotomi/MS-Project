import cv2
import os
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

        # Flag to ensure only one image is published
        self.image_published = False

        # Directory to save the captured image
        self.image_save_path = "/home/sotomi/ms_project/src/vfm_pkg/images/captured_image.jpg"

        # Timer to check and publish only once
        self.timer = self.create_timer(0.1, self.capture_and_publish_image)

    def capture_and_publish_image(self):
        """
        Capture an image using the webcam and publish it as a ROS topic.
        """
        if self.image_published:
            # Stop the timer once the image is published
            self.timer.cancel()
            self.get_logger().info("Image already published. Stopping the timer.")
            return

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

            # Publish the image and save it
            self.publish_and_save_image(frame)

        except Exception as e:
            self.get_logger().error(f"Error capturing image: {e}")

    def publish_and_save_image(self, frame):
        """
        Publish the captured image to the ROS topic and save it locally.
        """
        try:
            # Save the image locally
            cv2.imwrite(self.image_save_path, frame)
            self.get_logger().info(f"Image saved at {self.image_save_path}")

            # Convert the OpenCV image to a ROS Image message
            image_message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.image_publisher.publish(image_message)
            self.get_logger().info("Image published successfully.")
            self.image_published = True  # Set the flag to true after publishing

        except Exception as e:
            self.get_logger().error(f"Error publishing or saving image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


