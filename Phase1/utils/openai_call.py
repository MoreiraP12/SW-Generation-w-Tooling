import os
import requests
import json
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
project_id = os.getenv("VERTEX_AI_PROJECT_ID")
location = os.getenv("VERTEX_AI_LOCATION", "us-central1")  # Default to us-central1
model_name = os.getenv("VERTEX_AI_MODEL_NAME", "gemma-7b-it")  # Default to gemma-7b-it
service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Create the full URL for Vertex AI
url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_name}:predict"

def get_token():
    """
    Get OAuth token using service account credentials
    """
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    credentials.refresh(Request())
    return credentials.token

def call_vertex_ai_gemma(prompt, system_message="You are a helpful assistant.", max_tokens=150):
    """
    Function to call Vertex AI Gemma API
    """
    # Headers for authorization
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_token()}"
    }
    
    # Structure request for Gemma
    data = {
        "instances": [
            {
                "prompt": f"{system_message}\n\nUser: {prompt}\nAssistant:",
            }
        ],
        "parameters": {
            "temperature": 0.7,
            "maxOutputTokens": max_tokens,
            "topK": 40,
            "topP": 0.95
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}

# Example usage
if __name__ == "__main__":
    user_prompt = "Can you provide a summary of the latest advancements in AI?"
    result = call_vertex_ai_gemma(user_prompt)
    
    if "error" not in result:
        try:
            print("Response:")
            print(result["predictions"][0]["content"])
        except KeyError:
            print("Unexpected response format:", result)
    else:
        print(f"Error {result['error']}: {result['message']}")