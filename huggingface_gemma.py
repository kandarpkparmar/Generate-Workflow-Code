import os
from typing import Optional
import requests
from huggingface_hub import HfApi, InferenceClient
from dotenv import load_dotenv

#class defining gemma client
class HuggingFaceGemmaClient:
    """
    Client to Interact with Gemma Model
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initiate Client with api_key.

        Args:
            api_key: Hugging Face Access Token.
        """
        #load the environment
        load_dotenv()

        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("No API Key Provided")
        
        self.model_id = "mistralai/Mistral-7B-instruct-v0.1"
        self.api = HfApi(token = self.api_key)
        self.client = InferenceClient(
            model = self.model_id,
            token = self.api_key
        )

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using Gemma-27B.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Generated response text
        """
        try:
            # Format prompt according to Gemma's instruction format
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            response = self.client.text_generation(
                formatted_prompt,
                max_new_tokens=10000,  # Increased for 27B model
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,  # Slightly reduced for 27B model
                do_sample=True,
                top_k=50  # Added top-k sampling for better output diversity
            )
            return response
            
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")