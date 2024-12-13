import os
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

# --- Grad-CAM Implementation ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()
        score = output[:, class_idx]

        # Backward pass
        self.model.zero_grad()
        score.backward()

        # Generate the heatmap
        gradients = self.gradients.detach().numpy()[0]
        activations = self.activations.detach().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.size(-1), input_tensor.size(-2)))
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam

# --- Heatmap Node ---
class HeatmapNode(Node):
    def __init__(self):
        super().__init__('heatmap_node')
        
        # Initialize the pretrained model
        self.cam_model = models.resnet50(pretrained=True)
        self.cam_model.eval()
        self.bridge = CvBridge()

        # Subscriber to camera topic
        self.image_subscriber = self.create_subscription(
            RosImage,
            'camera/image_raw',
            self.image_callback,
            10
        )

        # Publisher for heatmap summary
        self.summary_publisher = self.create_publisher(String, 'heatmap/summary', 10)

        self.get_logger().info("HeatmapNode initialized and subscribed to 'camera/image_raw'.")

    def preprocess_image(self, image):
        """
        Preprocess the image for ResNet and Grad-CAM.
        """
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(image).unsqueeze(0)

    def generate_heatmap_summary(self, image):
        """
        Generate a heatmap for the given image and overlay it for visualization.
        """
        input_tensor = self.preprocess_image(image)
        grad_cam = GradCAM(self.cam_model, self.cam_model.layer4[1])  # Use ResNet's layer for Grad-CAM
        heatmap = grad_cam.generate_heatmap(input_tensor)

        # Overlay heatmap
        overlay_path = "/home/sotomi/ms_project/src/vfm_pkg/images/heatmap_overlay.jpg"
        plt.imshow(image)
        plt.imshow(cv2.resize(heatmap, (image.size[0], image.size[1])), cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.title("Heatmap Overlay")
        plt.savefig(overlay_path)
        self.get_logger().info(f"Heatmap overlay saved at {overlay_path}")

        # Generate summary
        high_attention_ratio = np.mean(heatmap > 0.7) * 100
        heatmap_summary = f"High attention area covers {high_attention_ratio:.2f}% of the image."
        self.get_logger().info(f"Heatmap Summary: {heatmap_summary}")

        # Publish the heatmap summary
        self.publish_summary(heatmap_summary)

    def publish_summary(self, summary):
        """
        Publish the heatmap summary to a ROS topic.
        """
        try:
            msg = String()
            msg.data = summary
            self.summary_publisher.publish(msg)
            self.get_logger().info("Heatmap summary published successfully.")
        except Exception as e:
            self.get_logger().error(f"Error publishing heatmap summary: {e}")

    def image_callback(self, msg):
        """
        Callback function to process images received from the camera.
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert OpenCV image to PIL format
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Generate the heatmap summary
            self.generate_heatmap_summary(pil_image)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = HeatmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
