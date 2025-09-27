#!/usr/bin/env python3
"""
DeepSeek API Client for medical knowledge graph recommendation system
"""

import requests
import json
import logging
from typing import Optional, Dict, Any
from config import MODEL_CONFIG


class DeepSeekAPIClient:
    """DeepSeek API client for LLM interactions"""
    
    def __init__(self, api_key: str = None):
        self.api_url = MODEL_CONFIG["api_url"]
        self.api_key = api_key or MODEL_CONFIG["api_key"]
        self.model_name = MODEL_CONFIG["model_name"]
        self.max_tokens = MODEL_CONFIG["max_tokens"]
        self.temperature = MODEL_CONFIG["temperature"]
        self.top_p = MODEL_CONFIG["top_p"]
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
    
    def generate_text(self, prompt: str, max_tokens: int = None, 
                     temperature: float = None, top_p: float = None) -> Optional[str]:
        """
        Generate text using DeepSeek API
        
        Args:
            prompt: Input prompt for text generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text or None if failed
        """
        try:
            # Use provided parameters or fall back to config defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature if temperature is not None else self.temperature
            top_p = top_p if top_p is not None else self.top_p
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False
            }
            
            # Set headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            self.logger.info(f"Generating text with DeepSeek API, prompt length: {len(prompt)}")
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Check response status
            if response.status_code == 200:
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    generated_text = result["choices"][0]["message"]["content"]
                    self.logger.info(f"Text generation successful, response length: {len(generated_text)}")
                    return generated_text
                else:
                    self.logger.error("No choices in API response")
                    return None
            else:
                self.logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request exception: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in text generation: {e}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_prompt = "Hello, this is a test message."
            response = self.generate_text(test_prompt, max_tokens=10)
            
            if response:
                self.logger.info("API connection test successful")
                return True
            else:
                self.logger.error("API connection test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"API connection test exception: {e}")
            return False


def main():
    """Test the DeepSeek API client"""
    client = DeepSeekAPIClient()
    
    # Test connection
    if client.test_connection():
        print("✅ DeepSeek API connection successful")
        
        # Test text generation
        test_prompt = "Please explain what is emergency triage in medical context."
        response = client.generate_text(test_prompt)
        
        if response:
            print("✅ Text generation successful")
            print(f"Response: {response[:200]}...")
        else:
            print("❌ Text generation failed")
    else:
        print("❌ DeepSeek API connection failed")


if __name__ == "__main__":
    main()
