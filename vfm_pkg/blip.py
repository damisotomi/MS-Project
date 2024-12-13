import rclpy
from rclpy.node import Node
from transformers import BlipProcessor, BlipForConditionalGeneration
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from PIL import Image as PILImage
import numpy as np
import io


class BlipCaptionNode(Node):
    def __init__(self):
        super().__init__('blip_caption_node')
        self.get_logger().info("BLIP Caption Node Initialized")

        # Load the BLIP processor and model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # ROS Subscriptions and Publications
        self.image_subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        self.caption_publisher = self.create_publisher(
            String,
            'blip/caption',
            10
        )

        # CV Bridge for converting ROS Image messages
        self.bridge = CvBridge()

        self.get_logger().info("BLIP Caption Node ready and waiting for images...")

    def image_callback(self, msg):
        """
        Callback function to process the incoming image message.
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert OpenCV image to PIL Image for BLIP
            pil_image = PILImage.fromarray(cv_image)

            # Generate caption for the image
            caption = self.generate_caption(pil_image)
            self.get_logger().info(f"Generated Caption: {caption}")

            # Publish the caption
            self.publish_caption(caption)
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

    def generate_caption(self, pil_image):
        """
        Generates a caption for a given PIL Image.
        """
        try:
            inputs = self.processor(images=pil_image, return_tensors="pt")
            output = self.model.generate(**inputs)
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            self.get_logger().error(f"Error generating caption: {e}")
            return "Error generating caption"

    def publish_caption(self, caption):
        """
        Publishes the generated caption to the ROS topic.
        """
        try:
            caption_msg = String()
            caption_msg.data = caption
            self.caption_publisher.publish(caption_msg)
            self.get_logger().info("Caption published successfully.")
        except Exception as e:
            self.get_logger().error(f"Error publishing caption: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = BlipCaptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()











