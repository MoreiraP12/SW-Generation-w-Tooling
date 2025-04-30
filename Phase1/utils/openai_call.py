import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Create the full URL
url = f"{endpoint}/openai/deployments/{deployment}/chat/completions"
params = {"api-version": "2024-02-01"}

# Headers
headers = {
    "Content-Type": "application/json",
    "api-key": api_key
}

def call_azure_openai(prompt, system_message="You are a helpful assistant.", max_tokens=150):
    """
    Function to call Azure OpenAI API with environment variables
    """
    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }
    
    response = requests.post(url, params=params, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}

# Example usage
if __name__ == "__main__":
    user_prompt = "Can you provide a summary of the latest advancements in AI?"
    result = call_azure_openai(user_prompt)
    
    if "error" not in result:
        print("Response:")
        print(result["choices"][0]["message"]["content"])
    else:
        print(f"Error {result['error']}: {result['message']}")