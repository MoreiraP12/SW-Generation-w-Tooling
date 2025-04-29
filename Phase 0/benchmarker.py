# Install necessary libraries if you don't have them
# pip install datasets pandas google-generativeai python-dotenv requests openai argparse google-cloud-aiplatform

import datasets
import pandas as pd
import random
import time  # For delays and backoff
import os  # To securely get configuration from environment variables, and check file existence
import re  # For parsing the LLM response
import requests  # For calling OpenRouter API
import json  # For OpenRouter API payload
from dotenv import load_dotenv  # To load environment variables from .env file
import argparse  # For command-line arguments
from functools import wraps  # For the decorator
from datetime import datetime  # For timestamped directories

# --- SDK Imports ---
try:
    import google.generativeai as genai
    # Import specific exceptions for backoff
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, GoogleAPIError
except ImportError:
    print("Error: google-generativeai library not found.")
    print("Please install using: pip install google-generativeai")
    # Set exceptions to BaseException if import fails to avoid NameError later
    ResourceExhausted = ServiceUnavailable = GoogleAPIError = BaseException
    genai = None  # Ensure genai is None if import failed

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError, InternalServerError
except ImportError:
    print("Error: openai library not found.")
    print("Please install using: pip install openai")
    # Set exceptions to BaseException if import fails
    APIError = RateLimitError = APIConnectionError = InternalServerError = BaseException
    OpenAI = None  # Ensure OpenAI is None if import failed

# --- Import Vertex AI for Vertex AI Gemma ---
try:
    from google.cloud import aiplatform
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value
except ImportError:
    print("Error: google-cloud-aiplatform library not found.")
    print("Please install using: pip install google-cloud-aiplatform")
    aiplatform = None

# --- Load Environment Variables ---
# Load variables from a .env file if it exists
load_dotenv()
print("Attempted to load variables from .env file.")

# --- Configuration ---

# API Keys from .env
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

# Model IDs from .env (or defaults) - Using stable IDs identified previously
GEMINI_MODEL_ID = os.getenv('GEMINI_MODEL_ID', 'gemini-2.0-flash')
GEMMA_MODEL_ID = os.getenv('GEMMA_MODEL_ID', 'google/gemma-3-4b-it:free')
NVIDIA_MODEL_ID = os.getenv('NVIDIA_MODEL_ID', 'deepseek-ai/deepseek-r1')

# Vertex AI Gemma Configuration
VERTEX_AI_PROJECT_ID = os.getenv('VERTEX_AI_PROJECT_ID')
VERTEX_AI_ENDPOINT_ID = os.getenv('VERTEX_AI_ENDPOINT_ID')
VERTEX_AI_LOCATION = os.getenv('VERTEX_AI_LOCATION', 'us-central1')

# API Endpoints / Base URLs
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Optional OpenRouter headers
OPENROUTER_REFERER = os.getenv('OPENROUTER_REFERER', '')
OPENROUTER_TITLE = os.getenv('OPENROUTER_TITLE', '')

# --- Backoff Configuration ---
MAX_RETRIES = 5  # Maximum number of retry attempts
INITIAL_DELAY = 8.0  # Initial delay in seconds
BACKOFF_FACTOR = 2.0  # Multiplier for the delay (exponential)
JITTER_FACTOR = 0.5  # Factor for randomization (0 to 1). 0.5 means +/- 50% of delay

# Define retryable exceptions for each service
# Using tuples for isinstance checks
RETRYABLE_GEMINI_EXCEPTIONS = (ResourceExhausted, ServiceUnavailable)
RETRYABLE_OPENROUTER_EXCEPTIONS = (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                                   requests.exceptions.ChunkedEncodingError)  # Add others if needed
RETRYABLE_NVIDIA_EXCEPTIONS = (RateLimitError, APIConnectionError, InternalServerError)
# For Vertex AI, we'll just use general exceptions since we don't have specific imports
RETRYABLE_VERTEX_EXCEPTIONS = (Exception,)

# --- Dataset Configurations ---
DATASET_CONFIGS = {
    "medmcqa": {
        "hf_path": "medmcqa",
        "hf_config": None,
        "split": "train",  # <<< CORRECTED SPLIT TO 'train' >>>
        "task_type": "mcqa",
        "question_field": "question",
        "options_fields": ["opa", "opb", "opc", "opd"],  # Specific fields (List of keys)
        "answer_field": "cop",  # 0-based index based on exploration
        "id_field": "id",
        "context_field": None,
        "explanation_field": "exp",
    },
    "mmlu_clinical_knowledge": {
        "hf_path": "cais/mmlu",
        "hf_config": "clinical_knowledge",
        "split": "test",  # MMLU typically evaluated on test
        "task_type": "mcqa",
        "question_field": "question",
        "options_fields": "choices",  # Field containing a list (String key)
        "answer_field": "answer",  # 0-based index
        "id_field": None,  # No specific ID field? Use index.
        "context_field": None,
        "explanation_field": None,
    },
    "pubmedqa_pqa_l": {
        "hf_path": "qiaojin/PubMedQA",
        "hf_config": "pqa_labeled",  # Corrected config name
        "split": "train",  # Using train split as it's labeled
        "task_type": "yesno",
        "question_field": "question",  # Corrected based on exploration
        "context_field": "CONTEXTS",  # Actual field is nested sample['context']['contexts']
        "answer_field": "final_decision",  # yes/no/maybe
        "id_field": "PUBMED_ID",  # Actual field is 'pubid'
        "options_fields": None,
        "explanation_field": None,
    },
    "medxpertqa": {
        "hf_path": "TsinghuaC3I/MedXpertQA",
        "hf_config": "Text",
        "split": "test",
        "task_type": "medXpert",
        "question_field": "question",
        "options_fields": "options",
        "answer_field": "label",
        "id_field": "id",
        "context_field": None,
        "explanation_field": None,
    },
    "pubmedqa_pqa_artificial": {
        "hf_path": "qiaojin/PubMedQA",
        "hf_config": "pqa_artificial",
        "split": "train",
        "task_type": "yesno",
        "question_field": "question",  # Corrected based on exploration
        "context_field": "CONTEXTS",  # Actual field is nested sample['context']['contexts']
        "answer_field": "final_decision",  # yes/no/maybe
        "id_field": "PUBMED_ID",  # Actual field is 'pubid'
        "options_fields": None,
        "explanation_field": "LONG_ANSWER",  # Actual field is 'long_answer'
    },
}

# Define available models
AVAILABLE_MODELS = ["gemini", "gemma", "nvidia", "vertex_gemma"]


# --- Helper Functions ---

def create_timestamped_directory():
    """
    Creates a directory with timestamp (YYYY-MM-DD_HH-MM-SS) and returns its path.
    """
    # Get current date and time
    now = datetime.now()
    # Format: 2025-04-22_14-30-45
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Create directory name
    dir_name = f"results_{timestamp}"

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created output directory: {dir_name}")

    return dir_name


# Unchanged from your provided script
def retry_with_exponential_backoff(retryable_exceptions):
    """
    Decorator for retrying a function with exponential backoff and jitter.

    Args:
        retryable_exceptions (tuple): A tuple of exception classes that should trigger a retry.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = INITIAL_DELAY
            last_exception = None
            for attempt in range(MAX_RETRIES + 1):  # +1 to allow the initial try
                try:
                    return func(*args, **kwargs)  # Attempt the function call
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == MAX_RETRIES:
                        print(f"  -> Max retries ({MAX_RETRIES}) reached for {func.__name__}. Last error: {e}")
                        # Return the standard error format expected by the calling code
                        return None, f"Retry Error: Max retries reached. Last error: {e}"

                    # Calculate delay with jitter
                    jitter = random.uniform(-JITTER_FACTOR * delay, JITTER_FACTOR * delay)
                    wait_time = max(0.1, delay + jitter)  # Ensure min wait time 0.1s

                    # Truncate long error messages for cleaner logs
                    error_msg = str(e)
                    if len(error_msg) > 150: error_msg = error_msg[:150] + "..."
                    print(
                        f"  -> Retryable error in {func.__name__}: {error_msg}. Retrying in {wait_time:.2f} seconds (Attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)

                    # Increase delay for next attempt
                    delay *= BACKOFF_FACTOR

                except Exception as e:
                    # Handle non-retryable exceptions immediately
                    print(f"  -> Non-retryable error in {func.__name__}: {e}")
                    # Return the standard error format
                    return None, f"API Error: {e}"

            # This part should theoretically not be reached if MAX_RETRIES >= 0
            # but added for safety. Return error if loop finishes unexpectedly.
            return None, f"Retry Error: Loop finished unexpectedly. Last error: {last_exception}"

        return wrapper

    return decorator


# Unchanged from your provided script
def load_data(config_key: str):
    """Loads dataset based on the configuration key."""
    if config_key not in DATASET_CONFIGS:
        print(f"Error: Unknown dataset configuration key '{config_key}'")
        return None, None

    config = DATASET_CONFIGS[config_key]
    print(
        f"\nLoading dataset '{config_key}': Path='{config['hf_path']}', Config='{config['hf_config']}', Split='{config['split']}'...")
    try:
        # Suppress verbose logging from datasets if desired
        # datasets.logging.set_verbosity_error()
        dataset = datasets.load_dataset(
            config["hf_path"],
            name=config["hf_config"],  # Use name parameter for config
            split=config["split"]
            # , trust_remote_code=True # Might be needed for some datasets, use with caution
        )
        # datasets.logging.set_verbosity_warning() # Restore default
        print("Dataset loaded successfully.")
        return dataset, config  # Return both dataset and its config
    except Exception as e:
        print(f"Error loading dataset '{config_key}': {e}")
        # Add hint for config name error based on user feedback
        if "BuilderConfig" in str(e) and "not found" in str(e):
            print(
                f"  -> HINT: Check if the config name '{config['hf_config']}' is correct. Available configs might be different (e.g., 'pqa_labeled' instead of 'pqa_l').")
        # Add hint for split name error
        if "Unknown split" in str(e):
            print(f"  -> HINT: Check if the split name '{config['split']}' is correct for this dataset/config.")
        return None, None


def format_question_for_llm(question_data, config):
    """Formats the prompt and returns extracted question text."""
    task_type = config["task_type"]
    question_field = config["question_field"]
    question_value = question_data.get(question_field)
    context = None
    options = None  # Initialize options dictionary
    prompt = ""

    # Determine actual question text or N/A
    raw_question_text = "N/A"
    if question_value is not None and str(question_value).strip() != "":
        raw_question_text = str(question_value)

    # Add context if available
    if config["context_field"]:
        context_config_key = config["context_field"]
        context_data = question_data.get(context_config_key)

        if config['hf_path'] == 'qiaojin/PubMedQA' and isinstance(question_data.get('context'), dict):
            context_list = question_data.get('context', {}).get('contexts', [])
            if isinstance(context_list, list):
                context = "\n".join(filter(None, context_list))
        elif isinstance(context_data, list):
            context = "\n".join(filter(None, context_data))
        elif isinstance(context_data, str):
            context = context_data
        else:
            context = None

        if context:
            max_context_len = 3000
            if len(context) > max_context_len:
                context = context[:max_context_len] + "... (truncated)"
            prompt += f"Context:\n{context}\n\n"

    # Add question
    prompt += f"Question: {raw_question_text}\n\n"

    # Add options for MCQA tasks
    if task_type == "mcqa":
        prompt += "Options:\n"
        options_fields_config = config.get("options_fields")

        if isinstance(options_fields_config, list):
            if len(options_fields_config) >= 4:
                options = {
                    'A': question_data.get(options_fields_config[0], ""),
                    'B': question_data.get(options_fields_config[1], ""),
                    'C': question_data.get(options_fields_config[2], ""),
                    'D': question_data.get(options_fields_config[3], ""),
                }
                for key, value in options.items():
                    prompt += f"{key}. {value}\n"
            else:
                prompt += "(Error: 'options_fields' list in config has fewer than 4 elements)\n"

        elif isinstance(options_fields_config, str):
            options_data = question_data.get(options_fields_config)

            # New: handle dict-style options
            if isinstance(options_data, dict):
                options = {k: options_data.get(k, "") for k in ['A', 'B', 'C', 'D']}
                for key, value in options.items():
                    prompt += f"{key}. {value}\n"

            # Fallback: handle list-style options
            elif isinstance(options_data, list) and len(options_data) >= 4:
                options = {k: v for k, v in zip('ABCD', options_data)}
                for key, value in options.items():
                    prompt += f"{key}. {value}\n"
            else:
                prompt += f"(Error: Could not format options - expected a list or dict in field '{options_fields_config}')\n"

        elif options_fields_config is None:
            prompt += "(Error: 'options_fields' not defined in config for MCQA task)\n"
        else:
            prompt += "(Error: Invalid 'options_fields' configuration type)\n"

    return prompt, options, raw_question_text


# <<< RENAMED & CORRECTED MedMCQA Mapping >>>
def get_ground_truth_answer(question_data, config):
    """Gets the ground truth answer string/letter based on dataset config."""
    task_type = config["task_type"]
    answer_field = config["answer_field"]
    ground_truth_raw = question_data.get(answer_field)

    if ground_truth_raw is None:
        return "N/A"

    # Special handling for MedXpertQA (uses string letters directly)
    if config["hf_path"] == "TsinghuaC3I/MedXpertQA":
        if isinstance(ground_truth_raw, str) and ground_truth_raw.strip().upper() in "ABCDEFGHIJ":
            return ground_truth_raw.strip().upper()
        else:
            return f"Invalid Raw GT: {ground_truth_raw}"

    if task_type == "mcqa":
        # Use try-except for safer integer conversion
        try:
            answer_index = int(ground_truth_raw)
        except (ValueError, TypeError):
            return f"Invalid Raw GT: {ground_truth_raw}"

        if config["hf_path"] == "medmcqa":
            mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            return mapping.get(answer_index, f"Invalid GT Index: {answer_index}")
        elif config["hf_path"] == "cais/mmlu":
            mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            return mapping.get(answer_index, f"Invalid GT Index: {answer_index}")
        else:
            return str(ground_truth_raw)

    elif task_type == "yesno":
        return str(ground_truth_raw).lower()

    return str(ground_truth_raw)


# <<< MODIFIED: Improved Yes/No Parsing + Typo check, returns None on failure >>>
def parse_llm_response(response_text: str, task_type: str):
    """
    Parses the LLM's text response. Improved for Yes/No/Maybe & MCQA.
    Returns parsed choice or None.
    """
    if not response_text:
        return None

    response_clean = response_text.strip()
    response_lower = response_clean.lower()

    if task_type == "yesno":
        # Priority 1: Exact match (case-insensitive)
        if response_lower == "yes": return "yes"
        if response_lower == "lyes": return "yes"  # <<< ADDED TYPO CHECK >>>
        if response_lower == "no": return "no"
        if response_lower == "lno": return "no"  # <<< ADDED TYPO CHECK >>>
        if response_lower == "maybe": return "maybe"
        # if response_lower == "lmaybe": return "maybe" # Less likely typo

        # Priority 2: Check for word boundaries (e.g., "Yes.", "Answer: No")
        if re.search(r'\b(yes)\b', response_lower): return "yes"
        if re.search(r'\b(no)\b', response_lower): return "no"
        if re.search(r'\b(maybe)\b', response_lower): return "maybe"

        # Priority 3: Check if the response *starts* with the word
        if response_lower.startswith("yes"): return "yes"
        if response_lower.startswith("no"): return "no"
        if response_lower.startswith("maybe"): return "maybe"

        # Priority 4: Check for bracketed answer (common for specific prompts)
        match_bracket = re.search(r'\s*\[\s*(yes|no|maybe)\s*\]', response_clean,
                                  re.IGNORECASE)  # Added whitespace allowance
        if match_bracket: return match_bracket.group(1).lower()

    elif task_type == "mcqa":
        # Priority 1: Bracketed answer
        match_bracket = re.search(r'\s*\[\s*([A-D])\s*\]', response_clean, re.IGNORECASE)  # Added whitespace allowance
        if match_bracket: return match_bracket.group(1).upper()

        # Priority 2: Standalone letter (case-insensitive)
        match_standalone = re.search(r'\b([A-D])\b', response_clean, re.IGNORECASE)
        if match_standalone: return match_standalone.group(1).upper()

        # Priority 3: Starts with letter (allowing punctuation, case-insensitive)
        match_start = re.match(r'^\s*([A-D])[\s\.\):]*', response_clean, re.IGNORECASE)
        if match_start: return match_start.group(1).upper()

        # Priority 4: Ends with letter (allowing preceding space, case-insensitive)
        match_end = re.search(r'\s([A-D])$', response_clean, re.IGNORECASE)
        if match_end: return match_end.group(1).upper()

    # If no specific format found after checks, return None
    # Warning printing moved to where it's called if needed
    else:
        return response_text
    return None


# --- API Call Functions ---

# Unchanged logic from original, but uses updated parse_llm_response
@retry_with_exponential_backoff(retryable_exceptions=RETRYABLE_GEMINI_EXCEPTIONS)
def call_gemini_api(prompt: str, model: genai.GenerativeModel, task_type: str):
    """
    Calls Gemini API with retry logic handled by the decorator.
    Adapts system prompt based on task type. Uses updated parser.
    """
    raw_response_text = "N/A"
    parsed_choice = None  # Initialize
    if not model:
        print("--- Skipping Gemini API Call (Not Initialized) ---")
        return None, "Skipped: Model not initialized"

    # System prompt based on task type
    if task_type == "mcqa":
        system_instruction = "Respond ONLY with the single letter."
    elif task_type == "yesno":
        system_instruction = "Respond ONLY with one word: Yes, No, or Maybe."
    else:
        system_instruction = "Respond ONLY with the single letter."

    # The try/except for retryable errors is now handled by the decorator
    # We still need to catch non-retryable API issues or handle the response structure
    try:
        generation_config = genai.types.GenerationConfig(max_output_tokens=400, temperature=0.1, top_p=0.95)
        print(f"  -> Calling Gemini ({getattr(model, 'model_name', '?')})...")  # Added model name log
        response = model.generate_content([system_instruction, prompt], generation_config=generation_config,
                                          request_options={'timeout': 60})  # Added timeout

        # Check for blocked content *before* trying to access .text
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raw_response_text = f"Blocked by API: {response.prompt_feedback.block_reason}"
            print(f"  -> Gemini Error: {raw_response_text}")
            return None, raw_response_text  # Return None for parsed_choice, and the error as raw_response

        # Try getting text attribute first
        raw_text_content = getattr(response, 'text', None)  # Safely get text attribute
        if raw_text_content is not None:
            raw_response_text = raw_text_content.strip()
            if not raw_response_text:
                raw_response_text = "Empty Response"  # Indicate empty explicitly
                # Don't print warning here, let outcome logic decide
            else:
                print(f"  -> Gemini Raw: '{raw_response_text[:100]}...'")  # Log if not empty
            parsed_choice = parse_llm_response(raw_response_text, task_type)  # Attempt parsing
        # Fallback to parts if text was None or empty
        elif hasattr(response, 'parts') and response.parts:
            raw_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
            if not raw_response_text:
                raw_response_text = "Empty Response from Parts"
            else:
                print(f"  -> Gemini Raw (from parts): '{raw_response_text[:100]}...'")
            parsed_choice = parse_llm_response(raw_response_text, task_type)
        else:  # Neither .text nor .parts yielded content
            raw_response_text = f"API Error: No text/parts found in response structure: {response}"
            print(f"  -> Gemini Error: {raw_response_text[:150]}...")
            # parsed_choice remains None

        # Optional: Log parsing failure here if needed for fine-grained debugging
        # if parsed_choice is None and "Error" not in raw_response_text and "Empty" not in raw_response_text and "Blocked" not in raw_response_text:
        #    print(f"  -> Gemini Parsing Failed for raw response: '{raw_response_text[:100]}...'")

        return parsed_choice, raw_response_text  # Always return both

    except (GoogleAPIError, Exception) as e:
        raw_response_text = f"Gemini Call Exception: {e}"
        print(f"  -> Gemini Error during call: {e}")
        return None, raw_response_text  # Return None for parsed, error for raw


# Unchanged logic from original, but uses updated parse_llm_response
@retry_with_exponential_backoff(retryable_exceptions=RETRYABLE_OPENROUTER_EXCEPTIONS + (requests.exceptions.HTTPError,))
def call_openrouter_gemma_api(prompt: str, api_key: str, model_id: str, task_type: str):
    """
    Calls OpenRouter API with retry logic handled by the decorator.
    Adapts system prompt based on task type. Uses updated parser.
    """
    raw_response_text = "N/A"
    parsed_choice = None  # Initialize
    if not api_key:
        print("--- Skipping OpenRouter API Call (No API Key) ---")
        return None, "Skipped: No API Key"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if OPENROUTER_REFERER: headers["HTTP-Referer"] = OPENROUTER_REFERER
    if OPENROUTER_TITLE: headers["X-Title"] = OPENROUTER_TITLE

    if task_type == "mcqa":
        system_prompt = "Respond ONLY with the single letter (A, B, C, or D)."
    elif task_type == "yesno":
        system_prompt = "Respond ONLY with one word: Yes, No, or Maybe."
    else:
        system_prompt = "Please answer the question."

    payload = {"model": model_id,
               "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
               "max_tokens": 2000, "temperature": 0.1}

    try:
        print(f"  -> Calling OpenRouter ({model_id})...")  # Added model name log
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload),
                                 timeout=60)  # Added timeout

        # Check for non-retryable client errors first
        # Note: Original backoff doesn't specifically handle 429 here if requests raises HTTPError
        if 400 <= response.status_code < 500:
            raw_response_text = f"API Client Error: {response.status_code} - {response.text}"
            print(f"  -> OpenRouter Client Error {response.status_code}: {response.text[:150]}...")
            return None, raw_response_text  # Return None for parsed, error for raw

        response.raise_for_status()  # Handle 5xx or other HTTP errors

        response_data = response.json()
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            message = response_data["choices"][0].get("message", {})
            content = message.get("content")
            if content is not None:
                raw_response_text = content.strip()
                if not raw_response_text:
                    raw_response_text = "Empty Response"
                else:
                    print(f"  -> OpenRouter Raw: '{raw_response_text[:100]}...'")  # Log raw response
                parsed_choice = parse_llm_response(raw_response_text, task_type)  # Attempt parsing
            else:
                raw_response_text = f"API Error: 'content' missing in response: {response_data}"
                print(f"  -> OpenRouter Error: {raw_response_text[:150]}...")
                # parsed_choice remains None
        else:  # No choices field or empty choices
            raw_response_text = f"API Error: 'choices' missing or empty: {response_data}"
            # Check for specific error message from OpenRouter
            if "error" in response_data: raw_response_text += f" | Error Field: {response_data['error']}"
            print(f"  -> OpenRouter Error: {raw_response_text[:150]}...")
            # parsed_choice remains None

        # Optional: Log parsing failure here if needed
        # if parsed_choice is None and "API Error" not in raw_response_text and "Empty Response" not in raw_response_text:
        #     print(f"  -> OpenRouter Parsing Failed for raw response: '{raw_response_text[:100]}...'")

        return parsed_choice, raw_response_text  # Always return both

    except requests.exceptions.RequestException as e:
        # This catches network errors, timeouts not handled by decorator, etc.
        raw_response_text = f"OpenRouter RequestException: {e}"
        print(f"  -> OpenRouter Request Error: {e}")
        return None, raw_response_text  # Return None for parsed, error for raw
    except Exception as e:
        # Catch JSONDecodeError or other unexpected issues
        raw_response_text = f"OpenRouter Unexpected Error: {e}"
        print(f"  -> OpenRouter Unexpected Error: {e}")
        return None, raw_response_text  # Return None for parsed, error for raw


# <<< MODIFIED: Using improved NVIDIA prompt/parsing & increased max_tokens >>>
@retry_with_exponential_backoff(retryable_exceptions=RETRYABLE_NVIDIA_EXCEPTIONS)
def call_nvidia_api(prompt: str, client: OpenAI, model_id: str, task_type: str):
    """
    Calls NVIDIA API with retry logic handled by the decorator.
    Uses improved system prompt and parsing logic.
    """
    raw_response_text = "N/A"
    parsed_choice = None  # Initialize

    if not client:
        print("--- Skipping NVIDIA API Call (Not Initialized) ---")
        return None, "Skipped: Client not initialized"

    # System prompt based on task type, requesting bracket format
    if task_type == "mcqa":
        system_prompt = "State final answer putting ONLY the single letter (A, B, C, or D) in square brackets at the end: [A]."
        expected_pattern = r'\[([A-D])\]'
    elif task_type == "yesno":
        system_prompt = "Analyze context/question. Respond ONLY with Yes, No, or Maybe inside square brackets at the end. Example: [Yes]."
        expected_pattern = r'\[(Yes|No|Maybe)\]'  # Case-insensitive search pattern needed in re.search
    else:
        system_prompt = (
            "INSTRUCTION: Answer the following multiple-choice question with ONLY the letter of the correct option.\n\n"
            "FORMAT: Your entire response must be exactly 'Answer: X' where X is just the letter.\n\n"
            "IMPORTANT: DO NOT repeat any question text or options in your response.\n\n")
        expected_pattern = r'Answer:\s*(\S)'

    # The try/except for retryable errors is now handled by the decorator
    try:
        print(f"  -> Calling NVIDIA ({model_id})...")  # Added model name log
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0.1, top_p=0.95,
            max_tokens=4096,  # <<< KEPT INCREASED TOKENS >>>
            stream=False,
            timeout=60.0  # Added timeout
        )

        if completion.choices and len(completion.choices) > 0:
            message = completion.choices[0].message
            finish_reason = completion.choices[0].finish_reason  # Get finish reason

            if message and message.content is not None:
                raw_response_text = message.content.strip()  # Strip whitespace
                if not raw_response_text:
                    raw_response_text = "Empty Response"
                    print(f"  -> NVIDIA Warn: {raw_response_text} (Finish: {finish_reason})")
                    # Fall through to parsing, which will return None
                else:

                    print(
                        f"  -> NVIDIA Raw (Finish: {finish_reason}): '{raw_response_text[:100]}...'")  # Log raw response and finish reason

                # --- Specific Parsing for [X] format ---
                if expected_pattern:
                    # Find *all* matches (case-insensitive) and take the last one
                    matches = re.findall(expected_pattern, raw_response_text, re.IGNORECASE)
                    if matches:
                        extracted_answer = matches[-1]  # Get the last bracketed answer found
                        parsed_choice = extracted_answer.upper() if task_type == "mcqa" or "medXpert" else extracted_answer.lower()
                        print(f"  -> NVIDIA Parsed (Bracket): {parsed_choice}")
                    else:
                        # Fallback if bracket format not found
                        # print("  -> NVIDIA: Bracket format not found, attempting fallback parsing...") # Less verbose
                        parsed_choice = parse_llm_response(raw_response_text, task_type)  # Use the general parser
                        if parsed_choice: print(
                            f"  -> NVIDIA Parsed (Fallback): {parsed_choice}")  # Log if fallback succeeded
                else:  # No specific format expected
                    parsed_choice = parse_llm_response(raw_response_text, task_type)
                    if parsed_choice: print(
                        f"  -> NVIDIA Parsed (General): {parsed_choice}")  # Log if general parsing succeeded

                # Report parsing failure if it occurred and response wasn't empty
                if parsed_choice is None and "Empty Response" not in raw_response_text:
                    print(f"  -> NVIDIA Parsing Failed.")
                    if finish_reason == 'length': print(
                        "  -> Warning: NVIDIA response likely truncated (max_tokens reached).")

                # Return parsed_choice (which might be None) and the raw_response_text
                return parsed_choice, raw_response_text
            else:
                # Message object exists but content is None
                raw_response_text = f"API Error: 'content' missing. Finish Reason: {finish_reason}. Completion: {completion}"
                print(f"  -> NVIDIA Response Error: {raw_response_text[:150]}...")
                return None, raw_response_text  # Return None for parsed, details for raw
        else:
            # No choices in response
            raw_response_text = f"API Error: 'choices' missing or empty. Completion: {completion}"
            print(f"  -> NVIDIA Response Error: {raw_response_text[:150]}...")
            return None, raw_response_text  # Return None for parsed, details for raw

    except (APIError, Exception) as e:
        # Catch API errors and other exceptions during the call
        raw_response_text = f"NVIDIA Call Exception: {e}"
        print(f"  -> Error calling NVIDIA API: {e}")
        return None, raw_response_text  # Return None for parsed, error for raw


# Modified function to retry if parsed answer is invalid
# Modified function to retry if parsed answer isn't a single letter for MCQA
@retry_with_exponential_backoff(retryable_exceptions=RETRYABLE_VERTEX_EXCEPTIONS)
def call_vertex_ai_gemma_api(prompt: str, project_id: str, endpoint_id: str, location: str, task_type: str, max_retry_attempts: int = 3):
    """
    Calls Vertex AI Gemma endpoint with retry logic.
    Formats prompt based on task type and parses response.
    Retries if the API returns anything other than a single letter for MCQA.

    Args:
        prompt: The formatted prompt to send
        project_id: Google Cloud project ID
        endpoint_id: Vertex AI endpoint ID
        location: GCP region (e.g., "us-central1")
        task_type: Type of task (mcqa, yesno)
        max_retry_attempts: Maximum number of retries for invalid answers

    Returns:
        Tuple of (parsed_answer, raw_response)
    """
    if not (project_id and endpoint_id and location and aiplatform):
        print("--- Skipping Vertex AI Gemma API Call (Missing Configuration) ---")
        return None, "Skipped: Missing configuration (project_id, endpoint_id, location, or aiplatform library)"

    # Define expected answer format based on task type
    if task_type == "mcqa":
        system_instruction = ("INSTRUCTION: Answer the following multiple-choice question with ONLY the letter of the correct option.\n\n"
            "FORMAT: Your entire response must be exactly 'Answer: X' where X is just the letter.\n\n"
            "IMPORTANT: DO NOT repeat any question text or options in your response.\n\n")
        # Define valid answer pattern for MCQA - single letters only
        valid_answer_pattern = re.compile(r'^[A-Z]$')
    elif task_type == "yesno":
        system_instruction = "Respond ONLY with one word: Yes, No, or Maybe. No explanation, just one word after 'Answer:'. Example: 'Answer: Yes'."
        # Define valid answer pattern for Yes/No - only Yes, No, or Maybe
        valid_answer_pattern = re.compile(r'^(Yes|No|Maybe)$', re.IGNORECASE)
    else:
        system_instruction = ("INSTRUCTION: Answer the following multiple-choice question with ONLY the letter of the correct option.\n\n"
            "FORMAT: Your entire response must be exactly 'Answer: X' where X is just the letter.\n\n"
            "IMPORTANT: DO NOT repeat any question text or options in your response.\n\n")
        valid_answer_pattern = re.compile(r'^[A-Z]$')

    # Format the full prompt with system instruction
    original_full_prompt = f"{str(system_instruction)}\n\n{prompt}"
    full_prompt = original_full_prompt

    raw_response_text = None

    for attempt in range(max_retry_attempts):
        try:
            # Initialize the Vertex AI client
            aiplatform.init(project=project_id, location=location)

            # Construct the full endpoint resource name
            endpoint_name = f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"

            # Create an Endpoint object
            endpoint = aiplatform.Endpoint(endpoint_name)

            print(f"  -> Calling Vertex AI Gemma (Endpoint ID: {endpoint_id}) - Attempt {attempt + 1}/{max_retry_attempts}...")

            # Format instances as per Vertex AI requirements
            instances = [{"prompt": full_prompt}]

            # Call the endpoint
            prediction = endpoint.predict(instances=instances)

            # Check if we have predictions
            if hasattr(prediction, 'predictions') and prediction.predictions:
                # Extract the raw response
                if isinstance(prediction.predictions[0], dict) and "text" in prediction.predictions[0]:
                    raw_response_text = prediction.predictions[0]["text"].strip()
                elif isinstance(prediction.predictions[0], str):
                    raw_response_text = prediction.predictions[0].strip()
                else:
                    raw_response_text = str(prediction.predictions[0]).strip()

                # Handle empty response
                if not raw_response_text:
                    print(f"  -> Vertex AI Gemma Warn: Empty response - Attempt {attempt + 1}")
                    continue

                print(f"  -> Vertex AI Gemma Raw: '{raw_response_text[:100]}...' - Attempt {attempt + 1}")

                # Extract the answer part
                answer_match = re.search(r'Answer:\s*([A-Za-z]+)', raw_response_text)

                if "Output:" in raw_response_text:
                    output_parts = raw_response_text.split("Output:")
                    if len(output_parts) > 1:
                        output_text = output_parts[1].strip()
                        answer_match = re.search(r'Answer:\s*([A-Za-z]+)', output_text)

                # If we found an Answer: pattern
                if answer_match:
                    extracted_answer = answer_match.group(1).strip()
                    print(f"  -> Extracted answer: '{extracted_answer}' - Attempt {attempt + 1}")

                    # Parse the response
                    parsed_choice = parse_llm_response(extracted_answer, task_type)

                    # Now validate if the parsed answer meets our format requirements
                    is_valid_format = False
                    if parsed_choice:
                        # For MCQA, check if it's just a single letter
                        if task_type == "mcqa":
                            is_valid_format = valid_answer_pattern.match(parsed_choice) is not None
                        # For Yes/No, check if it's Yes, No, or Maybe only
                        elif task_type == "yesno":
                            is_valid_format = valid_answer_pattern.match(parsed_choice) is not None
                        else:
                            is_valid_format = valid_answer_pattern.match(parsed_choice) is not None

                    # If valid format, return the result
                    if is_valid_format:
                        print(f"  -> Valid answer format found: '{parsed_choice}' - Attempt {attempt + 1}")
                        return parsed_choice, raw_response_text
                    else:
                        print(f"  -> Invalid answer format: '{parsed_choice}' - Attempt {attempt + 1}")

                        # For subsequent attempts, strengthen the instruction by emphasizing the format
                        if attempt < max_retry_attempts - 1:
                            # Create a stronger system instruction for retry
                            enhanced_system_instruction = system_instruction + (
                                "\nPLEASE READ CAREFULLY: Your previous response was incorrect. "
                                "For MCQA questions, you MUST provide ONLY a single letter (A, B, C, etc.) as your answer. "
                                "DO NOT provide explanations, reasoning, or context. "
                                "ONLY the letter. Example: 'Answer: A'\n\n"
                            )

                            # Use the stronger instruction for the retry
                            full_prompt = f"{enhanced_system_instruction}\n\n{prompt}"
                else:
                    print(f"  -> Could not extract answer with 'Answer:' pattern - Attempt {attempt + 1}")

                    # Create more explicit instruction for retry
                    if attempt < max_retry_attempts - 1:
                        enhanced_system_instruction = (
                            "INSTRUCTION: You MUST answer with EXACTLY this format: 'Answer: X' where X is just ONE LETTER.\n\n"
                            "IMPORTANT: I need ONLY the letter, nothing else. No explanations or additional text.\n\n"
                            "BAD: 'I think the answer is A because...'\n"
                            "BAD: 'Answer: The patient with...'\n"
                            "GOOD: 'Answer: A'\n\n"
                        )
                        full_prompt = f"{enhanced_system_instruction}\n\n{prompt}"
            else:
                # No predictions in response
                print(f"  -> No predictions in response - Attempt {attempt + 1}")
                raw_response_text = f"API Error: No predictions in response: {prediction}"
                continue

            # Add delay between attempts
            if attempt < max_retry_attempts - 1:
                time.sleep(1 * (attempt + 1))

        except Exception as e:
            raw_response_text = f"Vertex AI Gemma Exception: {e}"
            print(f"  -> Vertex AI Gemma Error: {e} - Attempt {attempt + 1}")
            # Let the retry_with_exponential_backoff decorator handle retryable exceptions
            raise

    # If we reach here, all attempts failed
    print(f"  -> All {max_retry_attempts} attempts failed to get a valid answer format.")
    # Return whatever was last parsed, even if invalid
    if 'parsed_choice' in locals() and parsed_choice:
        return parsed_choice, raw_response_text
    return None, raw_response_text if 'raw_response_text' in locals() else "All attempts failed"

# <<< RENAMED & MODIFIED: More general function to save results (not just failures) >>>
def save_result_to_csv(data_dict: dict, output_path: str, column_order: list, model_name: str):
    """
    Appends a single row of result data (success or failure) to a CSV file.
    Creates the file and writes the header if it doesn't exist.

    Args:
        data_dict (dict): Dictionary containing the data for one question.
        output_path (str): Path to the CSV file.
        column_order (list): Desired order of columns in the CSV.
        model_name (str): Name of the model (e.g., "Gemini") for logging.
    """
    try:
        # Use column names when accessing the dict
        parsed_ans_key = f'{model_name.lower()}_parsed_answer'
        raw_resp_key = f'{model_name.lower()}_raw_response'
        outcome_key = f'{model_name.lower()}_outcome'

        # Ensure basic types for CSV
        temp_dict = data_dict.copy()  # Work on a copy
        for key, value in temp_dict.items():
            if pd.isna(value):
                temp_dict[key] = ''  # Replace pandas NA with empty string
            elif not isinstance(value, (str, int, float, bool)):
                temp_dict[key] = str(value)  # Convert others to string

        df_row = pd.DataFrame([temp_dict])
        file_exists = os.path.exists(output_path)

        # Ensure all defined columns exist in the DataFrame before saving
        for col in column_order:
            if col not in df_row.columns:
                df_row[col] = ''  # Add missing columns with empty string
        df_row = df_row[column_order]  # Enforce column order using the passed list

        # Append to CSV
        df_row.to_csv(output_path, mode='a', header=not file_exists, index=False, encoding='utf-8')

        # Log using the correct keys from temp_dict
        outcome = temp_dict.get(outcome_key, 'UnknownOutcome')
        model_ans = temp_dict.get(parsed_ans_key, 'N/A')
        ground_truth = temp_dict.get('ground_truth_answer', 'N/A')
        raw_resp_short = str(temp_dict.get(raw_resp_key, ''))[:70] + "..."
        # Updated log to indicate success/failure status
        if outcome == "Correct":
            print(f"  -> Logged {model_name} Success (GT: {ground_truth}, Parsed: {model_ans}) to {output_path}")
        else:
            print(
                f"  -> Logged {model_name} {outcome} (GT: {ground_truth}, Parsed: {model_ans}, Raw: '{raw_resp_short}') to {output_path}")

    except Exception as e:
        print(f"  -> Error saving data row for {model_name} to {output_path}: {e}")


# <<< MODIFIED: Save all results (successes and failures) and use timestamped directory >>>
def evaluate_questions(dataset, config,
                       results_dir,  # New parameter for output directory
                       models_to_run,  # New parameter for which models to run
                       start_index=0, # New parameter: Starting index
                       end_index=None, # New parameter: Ending index (inclusive)
                       max_questions=None, # Max questions to process
                       gemini_model=None,
                       gemma_api_key=None,
                       gemma_model_id=None,
                       nvidia_client=None,
                       nvidia_model_id=None,
                       vertex_project_id=None,
                       vertex_endpoint_id=None,
                       vertex_location=None):
    """
    Evaluates LLM performance on the loaded dataset based on its config.
    Only runs the models specified in models_to_run list.
    Saves ALL results (successes and failures) to separate CSV files in a timestamped directory.
    """
    if dataset is None or config is None:
        print("Invalid dataset or config provided.")
        return

    # Construct output paths with the results directory, but only for selected models
    output_paths = {}
    if "gemini" in models_to_run:
        output_paths["gemini"] = os.path.join(results_dir, f"gemini_results_{args.dataset}.csv")
    if "gemma" in models_to_run:
        output_paths["gemma"] = os.path.join(results_dir, f"gemma_results_{args.dataset}.csv")
    if "nvidia" in models_to_run:
        output_paths["nvidia"] = os.path.join(results_dir, f"nvidia_results_{args.dataset}.csv")
    if "vertex_gemma" in models_to_run:
        output_paths["vertex_gemma"] = os.path.join(results_dir, f"vertex_gemma_results_{args.dataset}.csv")

    task_type = config["task_type"]
    processed_count = 0
    skipped_count = 0  # Count skipped items (missing Q or invalid GT)

    # Track counts for each selected model and outcome
    model_stats = {}
    for model in models_to_run:
        model_stats[model] = {
            "correct_count": 0,
            "incorrect_count": 0,
            "error_count": 0
        }

    print(f"\nStarting evaluation for dataset '{args.dataset}' (Task Type: {task_type}).")
    print(f"Processing from index {start_index} to {end_index or 'end'} (Max questions to process: {max_questions or 'unlimited'})")
    print(f"Running models: {', '.join(models_to_run)}")
    print(f"ALL results will be saved iteratively to:")
    for model, path in output_paths.items():
        print(f"  - {model.capitalize()}: {path}")
    print(
        f"Using Exponential Backoff: Max Retries={MAX_RETRIES}, Initial Delay={INITIAL_DELAY}s, Factor={BACKOFF_FACTOR}, Jitter={JITTER_FACTOR}")

    # --- Define base columns and model-specific column orders ---
    base_columns = ['id', 'dataset', 'question', 'ground_truth_answer']
    # Handle potential differences in explanation field name vs config key
    explanation_config_key = config.get("explanation_field")
    actual_exp_field = None  # This will store the actual key to use for getting data
    if explanation_config_key:
        # Use specific known field name if PubMedQA, otherwise use config key
        actual_exp_field = 'long_answer' if config['hf_path'] == 'qiaojin/PubMedQA' else config.get("explanation_field",
                                                                                                    None)
        # Special case for medmcqa which uses 'exp'
        if config['hf_path'] == 'medmcqa': actual_exp_field = 'exp'

        if actual_exp_field:  # Check if we determined a valid field name
            base_columns.append('ground_truth_explanation')

    # Define column names for CSV output using model prefixes
    column_orders = {
        "gemini": base_columns + ['gemini_model', 'gemini_parsed_answer', 'gemini_raw_response', 'gemini_outcome'],
        "gemma": base_columns + ['gemma_model', 'gemma_parsed_answer', 'gemma_raw_response', 'gemma_outcome'],
        "nvidia": base_columns + ['nvidia_model', 'nvidia_parsed_answer', 'nvidia_raw_response', 'nvidia_outcome'],
        "vertex_gemma": base_columns + ['vertex_gemma_model', 'vertex_gemma_parsed_answer', 'vertex_gemma_raw_response',
                                        'vertex_gemma_outcome']
    }
    # --- End Column Definition ---

    # Iterate through the dataset using indices
    data_len = len(dataset)
    if end_index is None:
        end_index = data_len - 1 # process until the end of dataset
    else:
        end_index = min(end_index, data_len - 1) # End index should not be longer than the dataset size

    for i in range(start_index, end_index + 1): # Iterate up to and *including* end_index
        if max_questions is not None and processed_count >= max_questions:
            print("Reached maximum number of valid questions. Stopping.")
            break
        try:
            question_data = dataset[i]
            items_checked = i + 1 # track how many items we checked

            # Determine ID (handle potential differences in field name)
            id_config_key = config.get("id_field")
            actual_id_field = 'pubid' if config['hf_path'] == 'qiaojin/PubMedQA' else id_config_key  # Use 'pubid' for PubMedQA
            if config['hf_path'] == 'medmcqa': actual_id_field = 'id'  # Use 'id' for medmcqa
            # Use index if ID field is missing or not found in data
            q_id = question_data.get(actual_id_field) if actual_id_field else None
            if q_id is None: q_id = f'index_{items_checked - 1}'  # Fallback to index

            print(f"\n--- Checking Item {items_checked} (Index: {i}, ID: {q_id}) ---")

            # 1. Format prompt AND get extracted question text
            # This function now returns "N/A" if the question is missing/empty
            prompt, options_dict, raw_question_text = format_question_for_llm(question_data, config)

            # --- CRITICAL CHECK 1: Skip if question is 'N/A' ---
            if raw_question_text == "N/A":
                print(
                    f"  -> SKIPPING Item ID {q_id} (Dataset Index: {items_checked - 1}): Question field ('{config['question_field']}') missing/empty.")
                skipped_count += 1
                continue  # Move to the next item in the dataset

            # --- Process Results (Get GT Answer BEFORE potentially skipping) ---
            ground_truth_ans = get_ground_truth_answer(question_data, config)
            print(f"Ground Truth Answer: {ground_truth_ans}")  # Print GT for checking

            # --- CRITICAL CHECK 2: Skip if Ground Truth is Invalid or Missing ---
            if ground_truth_ans == "N/A" or (
                    isinstance(ground_truth_ans, str) and ground_truth_ans.startswith("Invalid")):
                print(
                    f"  -> SKIPPING Item ID {q_id} (Dataset Index: {items_checked - 1}): Ground truth answer is missing or invalid ('{ground_truth_ans}').")
                skipped_count += 1  # Count it as skipped
                continue  # Skip to the next item in the dataset

            # If we reach here, the question AND ground truth are valid. Increment processed count.
            processed_count += 1  # Increment counter only for valid questions with valid GT
            print(f"--- Processing Valid Question {processed_count}/{max_questions or '?'} (ID: {q_id}) ---")
            print(f"  -> Extracted Question: '{raw_question_text[:100]}...'")  # Log question being processed

            # --- Call APIs for Selected Models ---
            # Initialize with default empty values
            model_responses = {
                "gemini": (None, "Not Run"),
                "gemma": (None, "Not Run"),
                "nvidia": (None, "Not Run"),
                "vertex_gemma": (None, "Not Run")
            }

            # Only call the APIs for selected models
            if "gemini" in models_to_run:
                print("--- Calling Gemini API ---")
                model_responses["gemini"] = call_gemini_api(prompt, gemini_model, task_type)

            if "gemma" in models_to_run:
                print("--- Calling OpenRouter API (Gemma) ---")
                model_responses["gemma"] = call_openrouter_gemma_api(prompt, gemma_api_key, gemma_model_id, task_type)

            if "nvidia" in models_to_run:
                print("--- Calling NVIDIA API (DeepSeek) ---")
                model_responses["nvidia"] = call_nvidia_api(prompt, nvidia_client, nvidia_model_id, task_type)

            if "vertex_gemma" in models_to_run:
                print("--- Calling Vertex AI Gemma API ---")
                model_responses["vertex_gemma"] = call_vertex_ai_gemma_api(
                    prompt, vertex_project_id, vertex_endpoint_id, vertex_location, task_type
                )

            # --- Determine Outcomes for Each Model ---
            def determine_outcome(parsed_answer, raw_response, ground_truth):
                """Determines outcome: Correct, Incorrect, API Error, Parsing Failed. Assumes GT is valid here."""
                if parsed_answer is not None:  # Parsing succeeded
                    return "Correct" if str(parsed_answer) == str(ground_truth) else "Incorrect"
                else:  # Parsing failed OR API Error occurred
                    raw_response_str = str(raw_response) if raw_response is not None else ""
                    # Check common error indicators in raw response
                    if any(err_sig in raw_response_str for err_sig in
                           ["API Error", "Retry Error", "Blocked by API", "Skipped", "Exception", "Error:",
                            "Client Error", "RequestException", "Unexpected", "Non-Retryable", "Call Exception"]):
                        return "API Error"
                    # Handle cases where API call returned None or explicitly "N/A", "Empty Response" etc.
                    elif raw_response is None or raw_response_str == "N/A" or raw_response_str.strip() == "Empty Response" or raw_response_str.strip() == "Empty Response from Parts" or raw_response_str.strip() == "Not Run":
                        return "API Error"  # Treat empty/skipped/no-init as API error for simplicity
                    else:  # API returned something, but parsing failed
                        return "Parsing Failed"

            model_outcomes = {}
            for model in models_to_run:
                parsed_ans, raw_response = model_responses[model]
                model_outcomes[model] = determine_outcome(parsed_ans, raw_response, ground_truth_ans)

                # Update counters based on outcomes
                if model_outcomes[model] == "Correct":
                    model_stats[model]["correct_count"] += 1
                elif model_outcomes[model] == "Incorrect":
                    model_stats[model]["incorrect_count"] += 1
                else:  # API Error or Parsing Failed
                    model_stats[model]["error_count"] += 1

                # Log the results
                print(f"{model.capitalize()} Parsed: {parsed_ans} (Outcome: {model_outcomes[model]})")

            # --- Save results for each selected model ---
            log_id = question_data.get(actual_id_field, q_id) if actual_id_field else q_id  # Use actual ID if possible

            # Prepare base info
            base_info = {'id': log_id, 'dataset': args.dataset, 'question': raw_question_text,
                         'ground_truth_answer': ground_truth_ans}
            if task_type == 'mcqa' and options_dict:
                base_info.update({f'option_{k.lower()}': v for k, v in options_dict.items()})
            # Get explanation using actual field name if defined
            if actual_exp_field:
                base_info['ground_truth_explanation'] = question_data.get(actual_exp_field, '')

            # Save results for each selected model
            for model in models_to_run:
                parsed_ans, raw_response = model_responses[model]
                model_info = {
                    **base_info,
                    f'{model}_model': MODEL_ID_MAPPING.get(model, "N/A"),
                    f'{model}_parsed_answer': parsed_ans if parsed_ans is not None else "N/A",
                    f'{model}_raw_response': raw_response or "N/A",
                    f'{model}_outcome': model_outcomes[model],
                }

                # Helper function to get the actual model ID for saving
                def get_model_id_for_saving(model_name):
                    if model_name == "gemini":
                        return getattr(gemini_model, 'model_name', '?') if gemini_model else "N/A"
                    elif model_name == "gemma":
                        return gemma_model_id
                    elif model_name == "nvidia":
                        return nvidia_model_id
                    elif model_name == "vertex_gemma":
                        return f"vertex-{vertex_endpoint_id}" if vertex_endpoint_id else "N/A"
                    return "N/A"

                # Update with actual model ID
                model_info[f'{model}_model'] = get_model_id_for_saving(model)

                # Save to CSV
                save_result_to_csv(model_info, output_paths[model], column_orders[model], model.capitalize())

            # Print summary every 10 questions processed
            if processed_count % 10 == 0:
                print(f"\n--- Progress Summary (After {processed_count} questions) ---")
                for model in models_to_run:
                    stats = model_stats[model]
                    print(
                        f"{model.capitalize()}: {stats['correct_count']} correct, {stats['incorrect_count']} incorrect, {stats['error_count']} errors")
                print()

        except IndexError:
            # This happens when the dataset runs out of items
            print(f"\nFinished processing all available items in the dataset ({len(dataset)} total).")
            break  # Exit the for loop

    # --- Final Summary ---
    # Calculate accuracy (counting only questions where parsing succeeded)
    def calculate_accuracy(correct, incorrect):
        total_valid = correct + incorrect
        return (correct / total_valid * 100) if total_valid > 0 else 0

    # Calculate accuracy for each model
    model_accuracy = {}
    for model in models_to_run:
        stats = model_stats[model]
        model_accuracy[model] = calculate_accuracy(stats["correct_count"], stats["incorrect_count"])

    print(f"\n--- Evaluation Summary ---")
    print(f"Dataset: '{args.dataset}'")
    print(f"Start Index: {start_index}")
    print(f"End Index: {end_index}")
    print(f"Max Questions: {max_questions or 'All'}")
    print(f"Total Items Checked in Dataset: {items_checked}")
    print(f"Skipped Due to Missing Question or Invalid Ground Truth: {skipped_count}")
    print(f"Valid Questions Processed: {processed_count}")
    print(f"\nModel Performance:")

    for model in models_to_run:
        stats = model_stats[model]
        print(
            f"  {model.capitalize()}: {stats['correct_count']} correct, {stats['incorrect_count']} incorrect, {stats['error_count']} errors (Accuracy: {model_accuracy[model]:.2f}%)")

    print(f"\nResults saved to: {results_dir}")

    # Save a summary file with the results
    summary_path = os.path.join(results_dir, f"summary_{args.dataset}.txt")
    try:
        with open(summary_path, 'w') as f:
            f.write(f"Evaluation Summary for Dataset: {args.dataset}\n")
            f.write(f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Start Index: {start_index}\n")
            f.write(f"End Index: {end_index}\n")
            f.write(f"Max Questions: {max_questions or 'All'}\n")
            f.write(f"Total Items Checked in Dataset: {items_checked}\n")
            f.write(f"Skipped Items: {skipped_count}\n")
            f.write(f"Valid Questions Processed: {processed_count}\n\n")
            f.write(f"Model Performance:\n")

            for model in models_to_run:
                stats = model_stats[model]
                model_id = "N/A"

                if model == "gemini":
                    model_id = getattr(gemini_model, 'model_name', '?') if gemini_model else "N/A"
                elif model == "gemma":
                    model_id = gemma_model_id
                elif model == "nvidia":
                    model_id = nvidia_model_id
                elif model == "vertex_gemma":
                    model_id = f"vertex-{vertex_endpoint_id}" if vertex_endpoint_id else "N/A"

                f.write(f"  {model.capitalize()} ({model_id}):\n")
                f.write(f"    Correct: {stats['correct_count']} ({model_accuracy[model]:.2f}%)\n")
                f.write(f"    Incorrect: {stats['incorrect_count']}\n")
                f.write(f"    Errors: {stats['error_count']}\n\n")

        print(f"Summary saved to: {summary_path}")
    except Exception as e:
        print(f"Error saving summary: {e}")


# Helper dictionary to map model names to their IDs
MODEL_ID_MAPPING = {
    "gemini": GEMINI_MODEL_ID,
    "gemma": GEMMA_MODEL_ID,
    "nvidia": NVIDIA_MODEL_ID,
    "vertex_gemma": f"vertex-endpoint-{VERTEX_AI_ENDPOINT_ID}"
}

# --- Main Execution (Handles Arguments) ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on medical datasets. Saves all results in timestamped directory.")
    # Add --clear_logs argument
    parser.add_argument("--clear_logs", action='store_true', help="Delete existing output CSV files first.")
    parser.add_argument("--dataset", "-d", required=True, choices=DATASET_CONFIGS.keys(),
                        help="Name of the dataset configuration to use.")
    parser.add_argument("--range", "-r", type=str, required=True,
                        help="Index range of questions to evaluate, in the format start:end (e.g. 100:500).")
    parser.add_argument("--models", "-m", nargs="+", choices=AVAILABLE_MODELS, default=AVAILABLE_MODELS,
                        help="List of models to run (default: all models)")
    args = parser.parse_args()

    # Create timestamped directory for outputs
    results_dir = create_timestamped_directory()

    # Clear logs is no longer needed since we're using timestamped directories,
    # but kept for backward compatibility
    if args.clear_logs:
        print("Note: --clear_logs not needed with timestamped directories, but noted.")

    #  extract range
    try:
        range_str = args.range
        start_str, end_str = range_str.split(':')
        start_index = int(start_str)
        end_index = int(end_str) if end_str else None
    except Exception as e:
        print(f"Wrong range {e}")

    # Initialize API clients, but only for selected models
    gemini_model_instance = None
    nvidia_client_instance = None

    # Initialize Gemini if needed
    if "gemini" in args.models:
        if GOOGLE_API_KEY and genai:  # Check if genai was imported successfully
            try:
                print(f"Configuring Google GenAI ({GEMINI_MODEL_ID})...")
                genai.configure(api_key=GOOGLE_API_KEY)
                gemini_model_instance = genai.GenerativeModel(GEMINI_MODEL_ID)
                print(" -> Google GenAI initialized successfully.")
            except Exception as e:
                print(f" -> WARNING: Error initializing Google GenAI: {e}")
        elif not GOOGLE_API_KEY:
            print(
                f"Skipping Gemini initialization (GOOGLE_API_KEY found: {bool(GOOGLE_API_KEY)}, Library loaded: {bool(genai)})")
        else:  # genai import failed
            print(
                f"Skipping Gemini initialization (google-generativeai library failed to import. Key found: {bool(GOOGLE_API_KEY)})")

    # Initialize NVIDIA if needed
    if "nvidia" in args.models:
        if NVIDIA_API_KEY and OpenAI:  # Check if OpenAI was imported successfully
            try:
                print(f"Configuring NVIDIA Client ({NVIDIA_MODEL_ID})...")
                nvidia_client_instance = OpenAI(base_url=NVIDIA_API_BASE_URL, api_key=NVIDIA_API_KEY)
                print(" -> NVIDIA API Client initialized successfully.")
            except Exception as e:
                print(f" -> WARNING: Error initializing NVIDIA API Client: {e}")
        elif not NVIDIA_API_KEY:
            print(
                f"Skipping NVIDIA initialization (NVIDIA_API_KEY found: {bool(NVIDIA_API_KEY)}, Library loaded: {bool(OpenAI)})")
        else:  # openai import failed
            print(
                f"Skipping NVIDIA initialization (openai library failed to import. Key found: {bool(NVIDIA_API_KEY)})")

    # Check OpenRouter key if needed
    if "gemma" in args.models:
        if OPENROUTER_API_KEY:
            print(f"OpenRouter Key found (Model: {GEMMA_MODEL_ID}). Will attempt calls.")
        else:
            print("Skipping OpenRouter calls (OPENROUTER_API_KEY not found).")

    # Check Vertex AI configuration if needed
    if "vertex_gemma" in args.models:
        if VERTEX_AI_PROJECT_ID and VERTEX_AI_ENDPOINT_ID and aiplatform:
            print(
                f"Vertex AI configuration found (Project: {VERTEX_AI_PROJECT_ID}, Endpoint: {VERTEX_AI_ENDPOINT_ID}).")
        else:
            print("Skipping Vertex AI Gemma calls (Missing configuration).")
            if "vertex_gemma" in args.models:
                args.models.remove("vertex_gemma")
                print(" -> Removed vertex_gemma from models to run.")

    # Check if at least one API client is available for the selected models
    valid_models = []
    if "gemini" in args.models and gemini_model_instance:
        valid_models.append("gemini")
    if "gemma" in args.models and OPENROUTER_API_KEY:
        valid_models.append("gemma")
    if "nvidia" in args.models and nvidia_client_instance:
        valid_models.append("nvidia")
    if "vertex_gemma" in args.models and VERTEX_AI_PROJECT_ID and VERTEX_AI_ENDPOINT_ID and aiplatform:
        valid_models.append("vertex_gemma")

    if not valid_models:
        print("\nError: No valid models to run. Exiting.")
        exit(1)  # Exit with error code
    else:
        print(f"\nRunning with models: {', '.join(valid_models)}")
        args.models = valid_models  # Update models list to only include valid ones

    # --- Load Data based on Args ---
    selected_dataset, selected_config = load_data(args.dataset)

    # --- Run Evaluation ---
    if selected_dataset and selected_config:
        evaluate_questions(
            dataset=selected_dataset,
            config=selected_config,
            results_dir=results_dir,  # Pass the timestamped directory
            models_to_run=args.models,  # Pass selected models
            gemini_model=gemini_model_instance,
            gemma_api_key=OPENROUTER_API_KEY,
            gemma_model_id=GEMMA_MODEL_ID,
            nvidia_client=nvidia_client_instance,
            nvidia_model_id=NVIDIA_MODEL_ID,
            vertex_project_id=VERTEX_AI_PROJECT_ID,
            vertex_endpoint_id=VERTEX_AI_ENDPOINT_ID,
            vertex_location=VERTEX_AI_LOCATION,
            start_index=start_index,
            end_index=end_index,
            max_questions=None
        )
    else:
        print(f"\nExiting due to dataset loading failure for '{args.dataset}'.")
        exit(1)  # Exit with error code

    print("\nScript finished.")