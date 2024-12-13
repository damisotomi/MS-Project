import rclpy
from rclpy.node import Node
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

class BlipCaption(Node):
    def __init__(self):
        super().__init__('blip_captioning_node')
        self.get_logger().info("BLIP Captioning Node Initialized")
        
        # Load the BLIP processor and model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Set a relative image path in the current project folder
        default_image_path = os.path.join(
            os.path.dirname(__file__), 
            'images', 'image1.jpg'
        )
        
        self.image_path = self.declare_parameter(
            "image_path", default_image_path
        ).get_parameter_value().string_value

        # Generate a caption for the example image
        self.caption = self.generate_caption(self.image_path)
        self.get_logger().info(f"Generated Caption: {self.caption}")

    def generate_caption(self, image_path):
        """
        Generates a caption for a given image path.
        """
        if not os.path.exists(image_path):
            self.get_logger().error(f"Image not found: {image_path}")
            return "Image not found"

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            output = self.model.generate(**inputs)
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            self.get_logger().error(f"Error generating caption: {e}")
            return "Error generating caption"

def main(args=None):
    rclpy.init(args=args)
    node = BlipCaption()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

