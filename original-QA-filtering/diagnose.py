import os
import json
import time
from dotenv import load_dotenv

# --- SDK Imports ---
# Attempt imports and print message if they fail
try:
    import google.generativeai as genai
    print("Successfully imported google.generativeai")
except ImportError:
    print("WARNING: google-generativeai library not found. Cannot test Gemini.")
    print("         Install using: pip install google-generativeai")
    genai = None

try:
    import requests
    print("Successfully imported requests")
except ImportError:
    print("WARNING: requests library not found. Cannot test OpenRouter.")
    print("         Install using: pip install requests")
    requests = None

try:
    from openai import OpenAI, APIError, AuthenticationError, NotFoundError, RateLimitError
    print("Successfully imported openai")
except ImportError:
    print("WARNING: openai library not found. Cannot test NVIDIA.")
    print("         Install using: pip install openai")
    OpenAI = None
    # Define dummy exceptions if import failed
    APIError = AuthenticationError = NotFoundError = RateLimitError = BaseException

# --- Load Environment Variables ---
load_dotenv()
print("\nAttempted to load variables from .env file.")

# --- Configuration (Match your main script's intended settings) ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

# Use the specific model IDs you intend to use in the main script
GEMINI_MODEL_ID = os.getenv('GEMINI_MODEL_ID', 'gemini-2.0-flash')
GEMMA_MODEL_ID = os.getenv('GEMMA_MODEL_ID', 'google/gemma-2-4b-it:free') # Using 9b free tier again
NVIDIA_MODEL_ID = os.getenv('NVIDIA_MODEL_ID', 'deepseek-ai/deepseek-r1') # Back to this ID

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_REFERER = os.getenv('OPENROUTER_REFERER', '') # Optional
OPENROUTER_TITLE = os.getenv('OPENROUTER_TITLE', '')     # Optional

# --- Simple Test Prompt ---
TEST_PROMPT = "What is 1 + 1?"
MAX_TOKENS_TEST = 10 # Keep it very small

# --- Test Functions ---

def test_gemini():
    print("\n--- Testing Gemini API ---")
    if not genai:
        print("SKIPPED: google.generativeai library not imported.")
        return
    if not GOOGLE_API_KEY:
        print("SKIPPED: GOOGLE_API_KEY not found in environment variables or .env file.")
        return

    try:
        print(f"Configuring Gemini with API Key and Model: {GEMINI_MODEL_ID}")
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_ID)
        print("Gemini client configured successfully.")

        print(f"Sending test prompt: '{TEST_PROMPT}'")
        # Use minimal generation config
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=MAX_TOKENS_TEST,
            temperature=0.1
        )
        # Add a timeout to the request
        response = model.generate_content(
            TEST_PROMPT,
            generation_config=generation_config,
            request_options={'timeout': 60} # 60 second timeout
        )

        # Check for blocks first
        if response.prompt_feedback and response.prompt_feedback.block_reason:
             print(f"ERROR: Gemini API blocked the prompt. Reason: {response.prompt_feedback.block_reason}")
             return

        # Check for content
        if hasattr(response, 'text') and response.text:
            print(f"SUCCESS: Gemini responded.")
            print(f"  -> Response Text (partial): '{response.text[:100]}'")
        else:
            print(f"ERROR: Gemini response received, but no text content found.")
            print(f"  -> Full Response Object: {response}")

    except Exception as e:
        print(f"ERROR: An exception occurred during Gemini test: {type(e).__name__}")
        print(f"  -> Details: {e}")
        if "API key not valid" in str(e):
            print("  -> HINT: Double-check your GOOGLE_API_KEY.")
        elif "permission" in str(e).lower() or "quota" in str(e).lower():
             print("  -> HINT: Check your Google AI Studio project quota/billing/API enablement.")
        elif "404" in str(e) or "could not find model" in str(e).lower():
             print(f"  -> HINT: Model ID '{GEMINI_MODEL_ID}' might be incorrect or unavailable.")

def test_openrouter():
    print("\n--- Testing OpenRouter API (Gemma) ---")
    if not requests:
        print("SKIPPED: requests library not imported.")
        return
    if not OPENROUTER_API_KEY:
        print("SKIPPED: OPENROUTER_API_KEY not found.")
        return

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    if OPENROUTER_REFERER: headers["HTTP-Referer"] = OPENROUTER_REFERER
    if OPENROUTER_TITLE: headers["X-Title"] = OPENROUTER_TITLE

    payload = {
        "model": GEMMA_MODEL_ID,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": MAX_TOKENS_TEST,
        "temperature": 0.1
    }

    try:
        print(f"Sending request to OpenRouter URL: {OPENROUTER_API_URL}")
        print(f"  -> Model: {GEMMA_MODEL_ID}")
        print(f"  -> Payload (partial): {json.dumps(payload)[:100]}...")
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=60 # 60 second timeout
        )

        print(f"  -> Response Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                response_data = response.json()
                if response_data.get("choices") and len(response_data["choices"]) > 0:
                    message = response_data["choices"][0].get("message", {})
                    content = message.get("content")
                    if content:
                        print(f"SUCCESS: OpenRouter responded.")
                        print(f"  -> Response Content (partial): '{content[:100]}'")
                    else:
                        print(f"ERROR: OpenRouter response OK (200), but 'content' missing in response structure.")
                        print(f"  -> Response JSON: {response_data}")
                else:
                    print(f"ERROR: OpenRouter response OK (200), but 'choices' missing or empty.")
                    # Check if there's an error structure instead
                    if "error" in response_data:
                        print(f"  -> Error reported in JSON: {response_data['error']}")
                    else:
                        print(f"  -> Response JSON: {response_data}")
            except json.JSONDecodeError:
                print(f"ERROR: OpenRouter response OK (200), but failed to decode JSON response.")
                print(f"  -> Raw Response Text: {response.text[:200]}...")
        else:
            # Specific error handling based on status code
            print(f"ERROR: OpenRouter returned status code {response.status_code}.")
            try:
                error_data = response.json() # Try to get JSON error details
                print(f"  -> Error Details (JSON): {error_data}")
            except json.JSONDecodeError:
                print(f"  -> Error Details (Raw Text): {response.text[:200]}...") # Show raw text if not JSON

            if response.status_code == 401:
                print("  -> HINT: Check your OPENROUTER_API_KEY (Unauthorized).")
            elif response.status_code == 404:
                print(f"  -> HINT: Model ID '{GEMMA_MODEL_ID}' might be incorrect or unavailable on OpenRouter (Not Found).")
            elif response.status_code == 429:
                print("  -> HINT: You've hit the OpenRouter rate limit (Too Many Requests). Wait before retrying.")
            elif 400 <= response.status_code < 500:
                 print("  -> HINT: This is a client-side error (e.g., bad request format, invalid arguments).")
            elif 500 <= response.status_code < 600:
                 print("  -> HINT: This is a server-side error on OpenRouter's end. Try again later.")

    except requests.exceptions.Timeout:
        print("ERROR: The request to OpenRouter timed out.")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: An exception occurred during OpenRouter request: {type(e).__name__}")
        print(f"  -> Details: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected exception occurred during OpenRouter test: {type(e).__name__}")
        print(f"  -> Details: {e}")


def test_nvidia():
    print("\n--- Testing NVIDIA API ---")
    if not OpenAI:
        print("SKIPPED: openai library not imported.")
        return
    if not NVIDIA_API_KEY:
        print("SKIPPED: NVIDIA_API_KEY not found.")
        return

    try:
        print(f"Configuring NVIDIA client with Base URL: {NVIDIA_API_BASE_URL}")
        print(f"  -> Target Model: {NVIDIA_MODEL_ID}")
        client = OpenAI(
            base_url=NVIDIA_API_BASE_URL,
            api_key=NVIDIA_API_KEY
        )
        print("NVIDIA client configured successfully.")

        print(f"Sending test prompt: '{TEST_PROMPT}'")
        completion = client.chat.completions.create(
          model=NVIDIA_MODEL_ID,
          messages=[{"role":"user","content": TEST_PROMPT}],
          temperature=0.1,
          max_tokens=MAX_TOKENS_TEST,
          stream=False,
          timeout=60.0 # 60 second timeout
        )

        # Check response structure
        if completion.choices and len(completion.choices) > 0:
            message = completion.choices[0].message
            if message and message.content:
                print(f"SUCCESS: NVIDIA API responded.")
                print(f"  -> Response Content (partial): '{message.content[:100]}'")
                print(f"  -> Finish Reason: {completion.choices[0].finish_reason}")
            else:
                print(f"ERROR: NVIDIA response structure seems OK, but message content is missing.")
                print(f"  -> Completion Object: {completion}")
        else:
            print(f"ERROR: NVIDIA response structure invalid ('choices' missing or empty).")
            print(f"  -> Completion Object: {completion}")

    # Catch specific OpenAI/NVIDIA errors
    except AuthenticationError as e:
        print(f"ERROR: NVIDIA Authentication Failed (401).")
        print(f"  -> Details: {e}")
        print(f"  -> HINT: Check your NVIDIA_API_KEY.")
    except NotFoundError as e:
         print(f"ERROR: NVIDIA Resource Not Found (404).")
         print(f"  -> Details: {e}")
         print(f"  -> HINT: Check the NVIDIA_API_BASE_URL ('{NVIDIA_API_BASE_URL}') and Model ID ('{NVIDIA_MODEL_ID}'). Is the model available at this endpoint?")
    except RateLimitError as e:
         print(f"ERROR: NVIDIA Rate Limit Exceeded (429).")
         print(f"  -> Details: {e}")
         print(f"  -> HINT: You are sending requests too quickly for your NVIDIA plan.")
    except APIError as e: # Catch other NVIDIA API errors (e.g., 400 Bad Request, 5xx Server Errors)
        print(f"ERROR: NVIDIA API Error (Status Code: {getattr(e, 'status_code', 'N/A')}).")
        print(f"  -> Details: {e}")
        if getattr(e, 'status_code', 0) == 400:
             print(f"  -> HINT: This is often a Bad Request (e.g., invalid input/parameters). Check model compatibility/payload.")
    except Exception as e:
        print(f"ERROR: An unexpected exception occurred during NVIDIA test: {type(e).__name__}")
        print(f"  -> Details: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting API Diagnostics...")
    test_gemini()
    time.sleep(1) # Small delay between tests
    test_openrouter()
    time.sleep(1)
    test_nvidia()
    print("\nDiagnostics finished.")
    print("Review the SUCCESS/ERROR messages above for each service.")
    print("Common issues: Incorrect API keys, invalid Model IDs, network problems, quota limits.")