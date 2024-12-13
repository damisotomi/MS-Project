import os
import rclpy
from rclpy.node import Node
from openai import AzureOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')
        self.endpoint = os.getenv("ENDPOINT_URL", "https://msproject.openai.azure.com/")
        self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo")
        self.subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "1seVCftIG83MnRYvMias4VCGLlzH4fYens2NZAZ7gEjvcTZpxKKsJQQJ99AJACYeBjFXJ3w3AAABACOGI05R")
        
        # Initialize the Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
            api_version="2024-05-01-preview",
        )
        
        # Example chat prompt
        self.chat_prompt = [
            {"role": "system", "content": "You are an AI helping to answer basic questions"},
            {"role": "user", "content": "when is the next US presidential election"}
        ]
        
        # Run the main process
        self.run_llm()

    def run_llm(self):
        try:
            # Generate completion
            self.get_logger().info("Sending request to Azure OpenAI...")
            completion = self.client.chat.completions.create(
                model=self.deployment,
                messages=self.chat_prompt,
                max_tokens=800,  # Adjust tokens as needed
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
        except Exception as e:
            self.get_logger().error(f"Error while generating response: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
