import os
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import rclpy
from rclpy.node import Node

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

        self.get_logger().info("HeatmapNode initialized.")
        self.run_heatmap_pipeline()

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

    def generate_heatmap_summary(self, image_path):
        """
        Generate a heatmap for the given image and overlay it for visualization.
        """
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess_image(image)
        grad_cam = GradCAM(self.cam_model, self.cam_model.layer4[1])  # Use ResNet's layer for Grad-CAM
        heatmap = grad_cam.generate_heatmap(input_tensor)

        # Overlay heatmap
        overlay_path = "heatmap_overlay.jpg"
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

    def run_heatmap_pipeline(self):
        """
        Simulate a pipeline to process an image for heatmap generation.
        """
        try:
            # Simulate capturing or loading an image
            image_path = "images/image1.jpg"  # Replace with a valid path
            self.get_logger().info(f"Processing image at: {image_path}")

            # Generate Heatmap and Summary
            self.generate_heatmap_summary(image_path)
        except Exception as e:
            self.get_logger().error(f"Error in heatmap pipeline: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = HeatmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
