from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import json # Import the json library

# BEFORE RUNNING THIS SCRIPT MAKE SURE TO: gcloud auth application-default login 
# On the account where this model was built

def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    location: str,
    instances: list # Changed type hint to list
    # instances: dict | list # More flexible type hint if needed
):
    """
    Calls a Vertex AI endpoint for prediction.

    Args:
        project: Your Google Cloud project ID.
        endpoint_id: The ID of the Vertex AI Endpoint.
        location: The region where the endpoint is located (e.g., "us-central1").
        instances: A list of instances for prediction. Each instance should be a
                   dictionary formatted according to your model's input requirements.
                   Example: [{"feature1": value1, "feature2": value2}, ...]
                   For the user's example {"instance_key_1": "value"},
                   this function expects it wrapped in a list:
                   [{"instance_key_1": "value"}]
    """
    # Initialize the Vertex AI client
    aiplatform.init(project=project, location=location)

    # Construct the full endpoint resource name
    endpoint_name = f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"

    # Create an Endpoint object
    endpoint = aiplatform.Endpoint(endpoint_name)

    print(f"Calling endpoint: {endpoint_name}")
    print(f"With instances: {json.dumps(instances, indent=2)}") # Use json for nice printing

    try:
        # The predict method expects a list of instances.
        # If your input 'instances' is a single dictionary, wrap it in a list.
        # If 'instances' is already a list of dictionaries, pass it directly.
        # This example assumes 'instances' passed to the function is ALREADY the list.
        prediction = endpoint.predict(instances=instances)

        print("\nPrediction response:")
        # The prediction object contains deployed_model_id and predictions
        print(f"  Deployed Model ID: {prediction.deployed_model_id}")
        print("  Predictions:")
        # Predictions are often returned as a list of dictionaries or values
        for pred in prediction.predictions:
             # Print nicely - might need adjustment depending on prediction format
            if isinstance(pred, dict):
                print(f"    {json.dumps(pred, indent=2)}")
            else:
                 print(f"    {pred}")

        return prediction

    except Exception as e:
        print(f"\nError during prediction: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    PROJECT_ID = "561066832551"
    ENDPOINT_ID = "7050478675136872448"
    LOCATION = "us-central1"

    # IMPORTANT: Structure this dictionary EXACTLY as your deployed model expects.
    # The example from your prompt suggests a single instance might look like this:
    # instance_data = {"instance_key_1": "some_input_value", "instance_key_2": 123}
    # If calling predict for just *one* item, wrap it in a list:
    # instances_list = [instance_data]

    # Replace this with your actual instance data, formatted correctly
    # for your specific model's input tensor(s).
    # This example assumes your model expects a dictionary with a key 'prompt'.
    # Adjust keys and values based on YOUR model's requirements.
    example_instances = [
        { "prompt": "Write a short poem about a rainy day." }
        # Add more dictionaries here if you want to send multiple instances in one call
        # ,{ "prompt": "Explain the concept of recursion." }
    ]

    # If your model expects a different structure (e.g., just raw values in a list per instance):
    # example_instances = [
    #     [1.0, 2.5, 3.0],
    #     [0.5, 1.8, 9.0]
    # ]

    print("--- Starting Prediction ---")
    predict_custom_trained_model_sample(
        project=PROJECT_ID,
        endpoint_id=ENDPOINT_ID,
        location=LOCATION,
        instances=example_instances
    )
    print("\n--- Prediction Finished ---")