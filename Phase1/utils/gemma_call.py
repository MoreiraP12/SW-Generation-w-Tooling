import os
import requests
import json
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from dotenv import load_dotenv

try:
    from google.cloud import aiplatform
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value
except ImportError:
    print("Error: google-cloud-aiplatform library not found.")
    print("Please install using: pip install google-cloud-aiplatform")
    aiplatform = None

# Load environment variables from .env file
load_dotenv()

# Vertex AI Gemma Configuration
VERTEX_AI_PROJECT_ID = os.getenv('VERTEX_AI_PROJECT_ID')
VERTEX_AI_ENDPOINT_ID = os.getenv('VERTEX_AI_ENDPOINT_ID')
VERTEX_AI_LOCATION = os.getenv('VERTEX_AI_LOCATION', 'us-central1')

def call_vertex_ai_gemma_api(prompt: str, project_id: str, endpoint_id: str, location: str, task_type: str, max_tokens: int = 256, temperature: float = 0.2, max_retry_attempts: int = 3):
    
    if not (project_id and endpoint_id and location and aiplatform):
        print("--- Skipping Vertex AI Gemma API Call (Missing Configuration) ---")
        return None, "Skipped: Missing configuration (project_id, endpoint_id, location, or aiplatform library)"
    
    # Initialize the Vertex AI client
    aiplatform.init(project=project_id, location=location)
    
    # Construct the full endpoint resource name
    endpoint_name = f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
    
    # Create an Endpoint object
    endpoint = aiplatform.Endpoint(endpoint_name)
    
    # Format instances as per Vertex AI requirements
    instances = [{"prompt": prompt}]
    
    # Add parameters for controlling generation
    parameters = {
        "max_output_tokens": max_tokens,  # Control maximum response length
        "temperature": temperature,  # Lower for more deterministic outputs
        "top_p": 0.95,
        "top_k": 40
    }
    
    # Call the endpoint with parameters
    prediction = endpoint.predict(instances=instances, parameters=parameters)
    print("Prediction response:", prediction)
    
    if prediction:
        try:
            # From the error message, we can see the prediction is coming back as a string
            # The actual response seems to be directly in predictions[0]
            response_text = prediction.predictions[0]
            print(f"Generated text:\n{response_text}")

            # Extract the words after "Output:"
            if "Output:" in response_text:
                response_text = response_text.split("Output:")[1].strip()
            return response_text, None
        except (KeyError, IndexError, AttributeError) as e:
            # If there's any error accessing the prediction, try to print more debug info
            print(f"Error processing prediction: {e}")
            print("Prediction type:", type(prediction))
            if hasattr(prediction, 'to_dict'):
                print("Raw response:")
                print(json.dumps(prediction.to_dict(), indent=2))
            else:
                print("Raw prediction:", prediction)
            return str(prediction), None

def main():
    """
    Main function to test the Vertex AI Gemma API call functionality.
    """
    # Test prompt
    test_prompt = "Tell me about the history of Portugal"
    
    # Configure generation parameters
    max_tokens = int(os.getenv('MAX_TOKENS', '4096'))  # Default to 256 if not specified
    temperature = float(os.getenv('TEMPERATURE', '0.2'))  # Default to 0.2 if not specified
    
    # Print configuration for debugging
    print("=== Vertex AI Gemma API Test ===")
    print(f"Project ID: {VERTEX_AI_PROJECT_ID}")
    print(f"Endpoint ID: {VERTEX_AI_ENDPOINT_ID}")
    print(f"Location: {VERTEX_AI_LOCATION}")
    
    # Check if required configuration is available
    if not (VERTEX_AI_PROJECT_ID and VERTEX_AI_ENDPOINT_ID):
        print("\nError: Missing required environment variables.")
        print("Please set VERTEX_AI_PROJECT_ID and VERTEX_AI_ENDPOINT_ID in your .env file.")
        return
    
    print("\nSending test prompt to Gemma model...")
    print(f"Prompt: '{test_prompt}'")
    
    try:
        # Call the Gemma API
        response_text, error_message = call_vertex_ai_gemma_api(
            prompt=test_prompt,
            project_id=VERTEX_AI_PROJECT_ID,
            endpoint_id=VERTEX_AI_ENDPOINT_ID,
            location=VERTEX_AI_LOCATION,
            task_type="text-generation",
            max_tokens=max_tokens,  # Use the environment variable or default
            temperature=temperature  # Use the environment variable or default
        )
        
        if response_text:
            print("\n=== API Response ===")
            print(f"Generated text:\n{response_text}")
        else:
            print(f"\nAPI call skipped or failed: {error_message}")
    
    except Exception as e:
        print(f"\nError calling Vertex AI Gemma API: {str(e)}")

if __name__ == "__main__":
    main()