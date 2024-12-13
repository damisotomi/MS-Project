import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from openai import AzureOpenAI
import matplotlib.pyplot as plt
import cv2
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')
        self.get_logger().info("LLM Node initialized.")

        # Azure OpenAI Configuration
        self.endpoint = "https://msproject.openai.azure.com/"
        self.deployment = "gpt-35-turbo"
        self.subscription_key = openai_key 

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
            api_version="2024-05-01-preview",
        )

        # Attributes to store data from subscribed topics
        self.caption = None
        self.heatmap_summary = None

        # Subscribers
        self.caption_subscriber = self.create_subscription(
            String, 'blip/caption', self.caption_callback, 10
        )
        self.heatmap_subscriber = self.create_subscription(
            String, 'heatmap/summary', self.heatmap_callback, 10
        )

        # Timer to check if data is ready and process it
        self.timer = self.create_timer(1.0, self.check_and_generate_summary)

    def caption_callback(self, msg):
        """
        Callback to handle data received from the Blip Node.
        """
        self.caption = msg.data
        self.get_logger().info(f"Received Caption: {self.caption}")

    def heatmap_callback(self, msg):
        """
        Callback to handle data received from the Heatmap Node.
        """
        self.heatmap_summary = msg.data
        self.get_logger().info(f"Received Heatmap Summary: {self.heatmap_summary}")

    def check_and_generate_summary(self):
        """
        Check if both caption and heatmap summary are available, then process.
        """
        if self.caption and self.heatmap_summary:
            self.get_logger().info("Data from both topics received. Generating LLM response...")
            response = self.generate_llm_summary(self.caption, self.heatmap_summary)
            # Display and save the result and heatmap
            self.display_and_save_result(response)
            # Clear the data to avoid reprocessing the same inputs
            self.caption = None
            self.heatmap_summary = None

    def generate_llm_summary(self, caption, heatmap_summary):
        """
        Generate a textual summary using Azure OpenAI LLM.
        """
        try:
            # Prepare the chat prompt
            chat_prompt = [
                {"role": "system", "content": "You are a mobile robot explaining observations."},
                {"role": "user", "content": f"The image caption is: '{caption}'. The heatmap analysis shows: '{heatmap_summary}'. Provide a short accurate description of this image. Do not mention heatmap details explicitly."}
            ]

            # Send request to Azure OpenAI
            self.get_logger().info("Sending request to Azure OpenAI...")
            completion = self.client.chat.completions.create(
                model=self.deployment,
                messages=chat_prompt,
                max_tokens=100,  # Adjust tokens as needed
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )

            # Log the response
            response = completion.choices[0].message.content
            self.get_logger().info(f"LLM Response: {response}")
            return response

        except Exception as e:
            self.get_logger().error(f"Error while generating response: {e}")
            return "Error generating response."

    def display_and_save_result(self, llm_response):
        """
        Display and save the LLM response and heatmap image.
        """
        heatmap_path = "/home/sotomi/ms_project/src/vfm_pkg/images/heatmap_overlay.jpg"  # Path to saved heatmap image
        output_figure_path = "/home/sotomi/ms_project/src/vfm_pkg/images/heatmap_llm_result.jpg"  # Path to save the figure

        if not os.path.exists(heatmap_path):
            self.get_logger().error(f"Heatmap image not found at: {heatmap_path}")
            return

        # Load the heatmap image
        heatmap_image = cv2.imread(heatmap_path)
        heatmap_image = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

        # Create the figure with larger dimensions
        plt.figure(figsize=(18, 10))  # Increased size for better visibility

        # Display the heatmap on one side
        plt.subplot(121)
        plt.imshow(heatmap_image)
        plt.axis('off')
        plt.title("Heatmap Overlay", fontsize=20)  # Larger title

        # Display the LLM response on the other side with larger text
        plt.subplot(122)
        plt.text(0.5, 0.5, llm_response, fontsize=18, ha='center', va='center', wrap=True)  # Larger font size
        plt.axis('off')
        plt.title("LLM Response", fontsize=20)  # Larger title

        # Adjust layout
        plt.subplots_adjust(wspace=0.4)  # Add space between the two plots

        # Save the figure
        plt.savefig(output_figure_path, bbox_inches='tight')
        self.get_logger().info(f"Figure saved at {output_figure_path}")

        # Show the figure
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
