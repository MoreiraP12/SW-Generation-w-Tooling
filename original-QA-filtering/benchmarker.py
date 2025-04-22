# Install necessary libraries if you don't have them
# pip install datasets pandas google-generativeai python-dotenv requests openai argparse

import datasets
import pandas as pd
import random
import time # For delays and backoff
import os # To securely get configuration from environment variables, and check file existence
import re # For parsing the LLM response
import requests # For calling OpenRouter API
import json # For OpenRouter API payload
from dotenv import load_dotenv # To load environment variables from .env file
import argparse # For command-line arguments
from functools import wraps # For the decorator

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
    genai = None # Ensure genai is None if import failed

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError, InternalServerError
except ImportError:
    print("Error: openai library not found.")
    print("Please install using: pip install openai")
    # Set exceptions to BaseException if import fails
    APIError = RateLimitError = APIConnectionError = InternalServerError = BaseException
    OpenAI = None # Ensure OpenAI is None if import failed


# --- Load Environment Variables ---
# Load variables from a .env file if it exists
load_dotenv()
print("Attempted to load variables from .env file.")

# --- Configuration ---

# API Keys from .env
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

# Model IDs from .env (or defaults)
GEMINI_MODEL_ID = os.getenv('GEMINI_MODEL_ID', 'gemini-2.5-flash-preview-04-17')
GEMMA_MODEL_ID = os.getenv('GEMMA_MODEL_ID', 'google/gemma-3-27b-it:free')
NVIDIA_MODEL_ID = os.getenv('NVIDIA_MODEL_ID', 'deepseek-ai/deepseek-r1')

# API Endpoints / Base URLs
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Optional OpenRouter headers
OPENROUTER_REFERER = os.getenv('OPENROUTER_REFERER', '')
OPENROUTER_TITLE = os.getenv('OPENROUTER_TITLE', '')

# --- Backoff Configuration ---
MAX_RETRIES = 5           # Maximum number of retry attempts
INITIAL_DELAY = 1.0       # Initial delay in seconds
BACKOFF_FACTOR = 2.0      # Multiplier for the delay (exponential)
JITTER_FACTOR = 0.5       # Factor for randomization (0 to 1). 0.5 means +/- 50% of delay

# Define retryable exceptions for each service
# Using tuples for isinstance checks
RETRYABLE_GEMINI_EXCEPTIONS = (ResourceExhausted, ServiceUnavailable)
RETRYABLE_OPENROUTER_EXCEPTIONS = (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) # Add others if needed
RETRYABLE_NVIDIA_EXCEPTIONS = (RateLimitError, APIConnectionError, InternalServerError)

# --- Dataset Configurations ---
DATASET_CONFIGS = {
    "medmcqa": {
        "hf_path": "medmcqa",
        "hf_config": None,
        "split": "validation", # Usually evaluate on validation or test
        "task_type": "mcqa",
        "question_field": "question",
        "options_fields": ["opa", "opb", "opc", "opd"], # Specific fields (List of keys)
        "answer_field": "cop", # 1-based index
        "id_field": "id",
        "context_field": None,
        "explanation_field": "exp",
    },
    "mmlu_clinical_knowledge": {
        "hf_path": "cais/mmlu",
        "hf_config": "clinical_knowledge",
        "split": "test", # MMLU typically evaluated on test
        "task_type": "mcqa",
        "question_field": "question",
        "options_fields": "choices", # Field containing a list (String key)
        "answer_field": "answer", # 0-based index
        "id_field": None, # No specific ID field? Use index.
        "context_field": None,
        "explanation_field": None,
    },
    "pubmedqa_pqa_l": {
        "hf_path": "qiaojin/PubMedQA",
        "hf_config": "pqa_l",
        "split": "train", # Using train split as it's labeled
        "task_type": "yesno",
        "question_field": "question",
        "context_field": "CONTEXTS", # List of contexts
        "answer_field": "final_decision", # yes/no/maybe
        "id_field": "PUBMED_ID",
        "options_fields": None,
        "explanation_field": None, # Can potentially use CONTEXTS or LABELS
    },
    "pubmedqa_pqa_artificial": {
        "hf_path": "qiaojin/PubMedQA",
        "hf_config": "pqa_artificial",
        "split": "train",
        "task_type": "yesno",
        "question_field": "question",
        "context_field": "CONTEXTS", # Single context string
        "answer_field": "final_decision", # yes/no/maybe
        "id_field": "PUBMED_ID",
        "options_fields": None,
        "explanation_field": "LONG_ANSWER",
    },
}

# --- Helper Functions ---

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
            for attempt in range(MAX_RETRIES + 1): # +1 to allow the initial try
                try:
                    return func(*args, **kwargs) # Attempt the function call
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == MAX_RETRIES:
                        print(f"  -> Max retries ({MAX_RETRIES}) reached for {func.__name__}. Last error: {e}")
                        # Return the standard error format expected by the calling code
                        return None, f"Retry Error: Max retries reached. Last error: {e}"

                    # Calculate delay with jitter
                    jitter = random.uniform(-JITTER_FACTOR * delay, JITTER_FACTOR * delay)
                    wait_time = max(0, delay + jitter) # Ensure wait time is not negative

                    print(f"  -> Retryable error in {func.__name__}: {e}. Retrying in {wait_time:.2f} seconds (Attempt {attempt + 1}/{MAX_RETRIES})...")
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


def load_data(config_key: str):
    """Loads dataset based on the configuration key."""
    if config_key not in DATASET_CONFIGS:
        print(f"Error: Unknown dataset configuration key '{config_key}'")
        return None, None

    config = DATASET_CONFIGS[config_key]
    print(f"\nLoading dataset '{config_key}': Path='{config['hf_path']}', Config='{config['hf_config']}', Split='{config['split']}'...")
    try:
        # Suppress verbose logging from datasets if desired
        # datasets.logging.set_verbosity_error()
        dataset = datasets.load_dataset(
            config["hf_path"],
            name=config["hf_config"], # Use name parameter for config
            split=config["split"]
        )
        # datasets.logging.set_verbosity_warning() # Restore default
        print("Dataset loaded successfully.")
        return dataset, config # Return both dataset and its config
    except Exception as e:
        print(f"Error loading dataset '{config_key}': {e}")
        return None, None

def format_question_for_llm(question_data, config):
    """Formats the prompt based on the dataset configuration and task type."""
    task_type = config["task_type"]
    question = question_data.get(config["question_field"], "N/A")
    context = None
    options = None # Initialize options dictionary

    prompt = ""

    # Add context if available (especially for PubMedQA)
    if config["context_field"]:
        context_data = question_data.get(config["context_field"])
        if isinstance(context_data, list):
            # Simple concatenation for list context (pqa_l)
            context = "\n".join(context_data)
        elif isinstance(context_data, str):
            context = context_data # Single string context (pqa_artificial)

        if context:
             # Simple truncation if context is too long (adjust length as needed)
             max_context_len = 3000
             if len(context) > max_context_len:
                 context = context[:max_context_len] + "... (truncated)"
             prompt += f"Context:\n{context}\n\n"

    # Add question
    prompt += f"Question: {question}\n\n"

    # Add options for MCQA tasks
    if task_type == "mcqa":
        prompt += "Options:\n"
        options_fields_config = config.get("options_fields") # Use .get for safety

        # Check if the config value itself is a list (MedMCQA style)
        if isinstance(options_fields_config, list):
            # Ensure we have enough keys defined in the list
            if len(options_fields_config) >= 4:
                options = {
                    'A': question_data.get(options_fields_config[0], ""), # Use elements from the list config as keys
                    'B': question_data.get(options_fields_config[1], ""),
                    'C': question_data.get(options_fields_config[2], ""),
                    'D': question_data.get(options_fields_config[3], ""),
                }
                for key, value in options.items():
                    prompt += f"{key}. {value}\n"
            else:
                 prompt += "(Error: 'options_fields' list in config has fewer than 4 elements)\n"

        # Check if the config value is a string (MMLU style)
        # If it's a string, use it as a key to get the list of options from question_data
        elif isinstance(options_fields_config, str):
            options_data = question_data.get(options_fields_config) # Get the list using the string key
            if isinstance(options_data, list) and len(options_data) >= 4: # Check if the result is a list
                 options = {
                     'A': options_data[0],
                     'B': options_data[1],
                     'C': options_data[2],
                     'D': options_data[3],
                 }
                 for key, value in options.items():
                    prompt += f"{key}. {value}\n"
            else:
                prompt += f"(Error: Could not format options - expected a list in field '{options_fields_config}' or list has < 4 items)\n"
        # Handle case where options_fields is missing or None
        elif options_fields_config is None:
             prompt += "(Error: 'options_fields' not defined in config for MCQA task)\n"
        # Handle unexpected type for options_fields
        else:
            prompt += "(Error: Invalid 'options_fields' configuration type)\n"

    # Final instruction removed - handled by system prompts in API calls now

    return prompt, options # Return formatted prompt and options dict (if applicable)


def get_correct_answer(question_data, config):
    """Gets the correct answer string/letter based on dataset config."""
    task_type = config["task_type"]
    answer_field = config["answer_field"]
    correct_answer_raw = question_data.get(answer_field)

    if correct_answer_raw is None:
        return "N/A"

    if task_type == "mcqa":
        if config["hf_path"] == "medmcqa": # MedMCQA: 1-based index (cop)
            mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
            return mapping.get(correct_answer_raw)
        elif config["hf_path"] == "cais/mmlu": # MMLU: 0-based index (answer)
            mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            return mapping.get(correct_answer_raw)
        else: # Default MCQA unknown handling
             return str(correct_answer_raw)
    elif task_type == "yesno": # PubMedQA
        return str(correct_answer_raw).lower() # Ensure lowercase yes/no/maybe
    else:
        return str(correct_answer_raw)


def parse_llm_response(response_text: str, task_type: str):
    """
    Parses the LLM's text response based on the expected task type.
    Handles 'yes'/'no'/'maybe' and 'A'/'B'/'C'/'D'.
    """
    if not response_text:
        return None

    response_text = response_text.strip().lower() # Process in lowercase

    if task_type == "yesno":
        # Look for explicit yes/no/maybe first
        if "yes" in response_text: return "yes"
        if "no" in response_text: return "no"
        # Be careful with maybe - avoid matching parts of words if possible
        if re.search(r'\bmaybe\b', response_text): return "maybe"
        # Fallback for bracketed yes/no/maybe (from NVIDIA prompt) - already handled in call_nvidia_api specific parsing
        # if re.search(r'\[(yes|no|maybe)\]', response_text): # Redundant due to specific parser? Maybe keep?
        #      return re.search(r'\[(yes|no|maybe)\]', response_text).group(1)

    elif task_type == "mcqa":
         # Look for standalone A/B/C/D first (already handles bracket case in nvidia func)
         match = re.search(r'\b([a-d])\b', response_text)
         if match: return match.group(1).upper()

         # Fallback: Check start of string
         match = re.match(r'^\s*([a-d])[\s\.\):]*', response_text)
         if match: return match.group(1).upper()

    # If no specific format found
    print(f"  -> Fallback Warning: Could not parse valid choice ({'Yes/No/Maybe' if task_type == 'yesno' else 'A/B/C/D'}) from response: '{response_text}'")
    return None


# --- API Call Functions (Modified with Backoff Decorator) ---

@retry_with_exponential_backoff(retryable_exceptions=RETRYABLE_GEMINI_EXCEPTIONS)
def call_gemini_api(prompt: str, model: genai.GenerativeModel, task_type: str):
    """
    Calls Gemini API with retry logic handled by the decorator.
    Adapts system prompt based on task type.
    """
    raw_response_text = "N/A"
    if not model:
         print("--- Skipping Gemini API Call (Not Initialized) ---")
         return None, "Skipped: Model not initialized"
    # print("--- Calling Gemini API ---") # Logging moved to evaluate_questions for clarity
    # print(f"Model: {model.model_name}")
    # print(f"Prompt sent (first 100 chars): {prompt[:100]}...")

    # System prompt based on task type
    if task_type == "mcqa":
        system_instruction = "Please answer the following multiple-choice question. Respond with ONLY the single letter (A, B, C, or D) corresponding to the best answer."
    elif task_type == "yesno":
        system_instruction = "Based on the provided context and question, please answer with ONLY one word: Yes, No, or Maybe."
    else:
        system_instruction = "Please answer the question." # Generic fallback

    # The try/except for retryable errors is now handled by the decorator
    # We still need to catch non-retryable API issues or handle the response structure
    try:
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=15, temperature=0.1, top_p=0.95 # Slightly more tokens for Yes/No/Maybe
        )
        response = model.generate_content(
            [system_instruction, prompt],
            generation_config=generation_config
        )
        # Check for blocked content *before* trying to access .text
        if response.prompt_feedback and response.prompt_feedback.block_reason:
             block_reason = response.prompt_feedback.block_reason
             raw_response_text = f"Blocked by API: {block_reason}"
             print(f"  -> Error: {raw_response_text}")
             # print("--------------------------") # Less verbose logging
             return None, raw_response_text

        # Check if parts exist before accessing .text
        if response.parts:
            raw_response_text = response.text
            # print(f"Gemini Raw Response: '{raw_response_text}'") # Less verbose logging
            parsed_choice = parse_llm_response(raw_response_text, task_type)
            # print(f"Parsed Choice: {parsed_choice}")
            # print("--------------------------")
            return parsed_choice, raw_response_text
        else:
            # Handle cases where response exists but has no parts (unexpected)
            raw_response_text = f"API Error: Empty/unexpected response structure. Full response: {response}"
            print(f"  -> {raw_response_text}")
            # print("--------------------------")
            return None, raw_response_text

    except GoogleAPIError as e:
        # Catch specific non-retryable Google API errors if needed, otherwise decorator handles general Exception
        raw_response_text = f"Gemini API Error (Non-Retryable?): {e}"
        print(f"  -> Error calling Gemini API: {e}")
        # print("--------------------------")
        return None, raw_response_text
    except Exception as e:
        # Catch any other unexpected errors during API interaction or parsing
        raw_response_text = f"Gemini Unexpected Exception: {e}"
        print(f"  -> Error during Gemini call/processing: {e}")
        # print("--------------------------")
        return None, raw_response_text


@retry_with_exponential_backoff(retryable_exceptions=RETRYABLE_OPENROUTER_EXCEPTIONS + (requests.exceptions.HTTPError,)) # Add HTTPError for 5xx status codes
def call_openrouter_gemma_api(prompt: str, api_key: str, model_id: str, task_type: str):
    """
    Calls OpenRouter API with retry logic handled by the decorator.
    Adapts system prompt based on task type.
    """
    raw_response_text = "N/A"
    if not api_key:
        print("--- Skipping OpenRouter API Call (No API Key) ---")
        return None, "Skipped: No API Key"
    # print("--- Calling OpenRouter API (Gemma) ---")
    # print(f"Model: {model_id}")
    # print(f"Prompt sent (first 100 chars): {prompt[:100]}...")

    headers = { "Authorization": f"Bearer {api_key}", "Content-Type": "application/json" }
    if OPENROUTER_REFERER: headers["HTTP-Referer"] = OPENROUTER_REFERER
    if OPENROUTER_TITLE: headers["X-Title"] = OPENROUTER_TITLE

    # System prompt based on task type
    if task_type == "mcqa":
        system_prompt = "Please answer the following multiple-choice question. Respond with ONLY the single letter (A, B, C, or D) corresponding to the best answer."
    elif task_type == "yesno":
        system_prompt = "Based on the provided context and question, please answer with ONLY one word: Yes, No, or Maybe."
    else:
        system_prompt = "Please answer the question." # Generic fallback

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 15, "temperature": 0.1,
    }

    # The try/except for retryable errors is now handled by the decorator
    try:
        response = requests.post(
            OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=60 # Keep timeout
        )
        # Check for non-successful status codes (4xx client errors are typically not retryable)
        # 5xx server errors might be retryable, let's include HTTPError in the decorator's list
        if 400 <= response.status_code < 500:
             raw_response_text = f"API Client Error: {response.status_code} - {response.text}"
             print(f"  -> {raw_response_text}")
             return None, raw_response_text # Not retryable

        response.raise_for_status() # Raises HTTPError for 4xx/5xx. 5xx will be caught by decorator if listed.

        response_data = response.json()

        if response_data.get("choices") and len(response_data["choices"]) > 0:
            message = response_data["choices"][0].get("message", {})
            raw_response_text = message.get("content")
            if raw_response_text is not None:
                # print(f"OpenRouter Raw Response: '{raw_response_text}'")
                parsed_choice = parse_llm_response(raw_response_text, task_type)
                # print(f"Parsed Choice: {parsed_choice}")
                # print("-----------------------------")
                return parsed_choice, raw_response_text
            else:
                 raw_response_text = f"API Error: 'content' field missing. JSON: {response_data}"
                 print(f"  -> {raw_response_text}")
                 return None, raw_response_text
        else:
            raw_response_text = f"API Error: 'choices' missing or empty. JSON: {response_data}"
            print(f"  -> {raw_response_text}")
            return None, raw_response_text

    except requests.exceptions.RequestException as e:
        # This might catch non-retryable request errors if not listed in decorator
        # Should ideally be caught by decorator if listed in RETRYABLE_OPENROUTER_EXCEPTIONS
        raw_response_text = f"OpenRouter RequestException (Non-Retryable?): {e}"
        if hasattr(e, 'response') and e.response is not None:
             try: raw_response_text += f" | Response Body: {e.response.text}"
             except Exception: pass
        print(f"  -> Error calling OpenRouter API: {e}")
        return None, raw_response_text
    except Exception as e:
        # Catch JSONDecodeError or other unexpected issues
        raw_response_text = f"OpenRouter Unexpected Error: {e}"
        print(f"  -> {raw_response_text}")
        return None, raw_response_text


@retry_with_exponential_backoff(retryable_exceptions=RETRYABLE_NVIDIA_EXCEPTIONS)
def call_nvidia_api(prompt: str, client: OpenAI, model_id: str, task_type: str):
    """
    Calls NVIDIA API with retry logic handled by the decorator.
    Adapts system prompt and parsing.
    """
    raw_response_text = "N/A"
    parsed_choice = None

    if not client:
        print("--- Skipping NVIDIA API Call (Not Initialized) ---")
        return None, "Skipped: Client not initialized"
    # print("--- Calling NVIDIA API (DeepSeek) ---")
    # print(f"Model: {model_id}")
    # print(f"Prompt sent (first 100 chars): {prompt[:100]}...")

    # System prompt based on task type, requesting bracket format
    if task_type == "mcqa":
        system_prompt = "Please answer the following multiple-choice question. Analyze the options and provide your reasoning if necessary, but state your final answer clearly by putting the single letter (A, B, C, or D) inside square brackets at the end, like this: [A]."
        expected_pattern = r'\[([A-D])\]' # Regex for [A] format
    elif task_type == "yesno":
        system_prompt = "Based on the provided context and question, analyze the information and state your final answer clearly by putting the single word (Yes, No, or Maybe) inside square brackets at the end, like this: [Yes]."
        expected_pattern = r'\[(Yes|No|Maybe)\]' # Regex for [Yes] format
    else:
        system_prompt = "Please answer the question."
        expected_pattern = None # No specific format expected

    # The try/except for retryable errors is now handled by the decorator
    try:
        completion = client.chat.completions.create(
          model=model_id,
          messages=[
              {"role":"system", "content": system_prompt},
              {"role":"user","content": prompt}
          ],
          temperature=0.1, top_p=0.95,
          max_tokens=45,   # Allow more tokens for reasoning + bracket format
          stream=False
        )

        if completion.choices and len(completion.choices) > 0:
            message = completion.choices[0].message
            if message and message.content is not None:
                raw_response_text = message.content
                # print(f"NVIDIA Raw Response: '{raw_response_text}'")

                # --- Specific Parsing for [X] format ---
                if expected_pattern:
                    match = re.search(expected_pattern, raw_response_text, re.IGNORECASE)
                    if match:
                        parsed_choice = match.group(1).lower() # Use lower for yes/no/maybe consistency
                        if task_type == "mcqa": parsed_choice = parsed_choice.upper() # Uppercase for MCQA
                        # print(f"Parsed Choice (Bracket Format): {parsed_choice}")
                    else:
                        # If bracket format not found, try the general fallback parser
                        print("  -> NVIDIA: Bracket format not found, attempting fallback parsing...")
                        parsed_choice = parse_llm_response(raw_response_text, task_type)
                        # print(f"Parsed Choice (Fallback): {parsed_choice}")
                else: # No specific format expected, use general parser
                     parsed_choice = parse_llm_response(raw_response_text, task_type)
                     # print(f"Parsed Choice (General): {parsed_choice}")
                # --- End Specific Parsing ---

                # print("--------------------------")
                return parsed_choice, raw_response_text
            else:
                raw_response_text = f"API Error: 'content' missing. Completion: {completion}"
                print(f"  -> {raw_response_text}")
                return None, raw_response_text
        else:
            raw_response_text = f"API Error: 'choices' missing or empty. Completion: {completion}"
            print(f"  -> {raw_response_text}")
            return None, raw_response_text

    except APIError as e:
        # Catch specific non-retryable NVIDIA API errors if needed
        # e.g., AuthenticationError, BadRequestError, NotFoundError
        # Decorator handles RateLimitError, APIConnectionError, InternalServerError
        raw_response_text = f"NVIDIA API Error (Non-Retryable?): {e}"
        print(f"  -> Error calling NVIDIA API: {e}")
        return None, raw_response_text
    except Exception as e:
        # Catch any other unexpected errors
        raw_response_text = f"NVIDIA Unexpected Error: {e}"
        print(f"  -> {raw_response_text}")
        return None, raw_response_text


# --- NEW HELPER FUNCTION ---
def save_failure_iteratively(data_dict: dict, output_path: str, column_order: list, model_name: str):
    """
    Appends a single row of failure/error data to a CSV file.
    Creates the file and writes the header if it doesn't exist.

    Args:
        data_dict (dict): Dictionary containing the data for one failed question.
        output_path (str): Path to the CSV file.
        column_order (list): Desired order of columns in the CSV.
        model_name (str): Name of the model (e.g., "Gemini") for logging.
    """
    try:
        # Convert the single dictionary to a DataFrame row
        df_row = pd.DataFrame([data_dict])

        # Check if file exists to determine if header should be written
        file_exists = os.path.exists(output_path)

        # Ensure all columns exist in the DataFrame, add if missing (with pd.NA)
        # and reorder columns according to column_order
        for col in column_order:
            if col not in df_row.columns:
                df_row[col] = pd.NA # Use pandas' missing value indicator
        df_row = df_row[column_order] # Enforce column order

        # Append to CSV
        df_row.to_csv(output_path, mode='a', header=not file_exists, index=False, encoding='utf-8')
        status = data_dict.get(f'{model_name.lower()}_status', 'UnknownStatus') # Get the specific status
        print(f"  -> Appending {model_name} Failure/Error (Status: {status}) to {output_path}")

    except Exception as e:
        print(f"  -> Error saving failure data for {model_name} to {output_path}: {e}")


# --- Main Evaluation Logic (Modified for Iterative Saving) ---
def evaluate_questions(dataset, config,
                       gemini_output_path, gemma_output_path, nvidia_output_path,
                       gemini_model, gemma_api_key, gemma_model_id,
                       nvidia_client, nvidia_model_id,
                       max_questions=None):
    """
    Evaluates LLM performance on the loaded dataset based on its config.
    Saves failures for each model iteratively to separate CSV files.
    Includes exponential backoff for API calls.
    """
    if dataset is None or config is None:
        print("Invalid dataset or config provided.")
        return

    task_type = config["task_type"]
    # REMOVED LISTS: gemini_failed_data, gemma_failed_data, nvidia_failed_data
    processed_count = 0
    gemini_failure_count = 0 # Keep counts for summary
    gemma_failure_count = 0
    nvidia_failure_count = 0

    print(f"\nStarting evaluation for dataset '{args.dataset}' (Task Type: {task_type}).")
    print(f"Processing up to {max_questions or 'all'} questions...")
    print(f"Failures/Errors will be saved iteratively to:")
    print(f"  - Gemini: {gemini_output_path}")
    print(f"  - Gemma: {gemma_output_path}")
    print(f"  - NVIDIA: {nvidia_output_path}")
    print(f"Using Exponential Backoff: Max Retries={MAX_RETRIES}, Initial Delay={INITIAL_DELAY}s, Factor={BACKOFF_FACTOR}, Jitter={JITTER_FACTOR}")

    # --- Define base columns and model-specific column orders *before* the loop ---
    base_columns = ['id', 'dataset', 'question']
    if task_type == 'mcqa':
        base_columns.extend(['option_a', 'option_b', 'option_c', 'option_d'])
    base_columns.append('correct_answer')
    if config.get("explanation_field"): # Add explanation column only if it exists for the dataset
         base_columns.append('explanation')

    gemini_column_order = base_columns + ['gemini_model', 'gemini_answer', 'gemini_raw_response', 'gemini_status']
    gemma_column_order = base_columns + ['gemma_model', 'gemma_answer', 'gemma_raw_response', 'gemma_status']
    nvidia_column_order = base_columns + ['nvidia_model', 'nvidia_answer', 'nvidia_raw_response', 'nvidia_status']
    # --- End Column Definition ---

    for i, question_data in enumerate(dataset):
        current_max = max_questions if max_questions is not None else float('inf')
        if processed_count >= current_max:
            print(f"Reached processing limit of {max_questions} questions.")
            break

        processed_count += 1
        q_id = question_data.get(config["id_field"], f'index_{i}') if config["id_field"] else f'index_{i}'
        print(f"\nProcessing Question {processed_count} (ID: {q_id})...")

        # 1. Format prompt based on dataset config
        prompt, options_dict = format_question_for_llm(question_data, config) # options_dict only relevant for mcqa
        if "(Error:" in prompt:
            print(f"  -> Skipping question {q_id} due to prompt formatting error: {prompt.split('(Error:')[1].split(')')[0]})")
            continue # Skip to the next question

        # --- Call APIs ---
        print("--- Calling Gemini API ---")
        gemini_answer, gemini_raw_response = call_gemini_api(prompt, gemini_model, task_type)
        # time.sleep(0.1) # Optional small delay

        print("--- Calling OpenRouter API (Gemma) ---")
        gemma_answer, gemma_raw_response = call_openrouter_gemma_api(prompt, gemma_api_key, gemma_model_id, task_type)
        # time.sleep(0.1)

        print("--- Calling NVIDIA API (DeepSeek) ---")
        nvidia_answer, nvidia_raw_response = call_nvidia_api(prompt, nvidia_client, nvidia_model_id, task_type)
        # time.sleep(0.1)

        # --- Process Results ---
        correct_answer = get_correct_answer(question_data, config)
        print(f"Correct Answer: {correct_answer}")

        # Determine status for each model
        gemini_status = "API Error/Invalid Response"
        if gemini_answer is not None:
             gemini_status = "Correct" if str(gemini_answer) == str(correct_answer) else "Incorrect"

        gemma_status = "API Error/Invalid Response"
        if gemma_answer is not None:
            gemma_status = "Correct" if str(gemma_answer) == str(correct_answer) else "Incorrect"

        nvidia_status = "API Error/Invalid Response"
        if nvidia_answer is not None:
            nvidia_status = "Correct" if str(nvidia_answer) == str(correct_answer) else "Incorrect"

        print(f"Gemini Result: {gemini_answer} ({gemini_status})")
        print(f"Gemma Result: {gemma_answer} ({gemma_status})")
        print(f"NVIDIA Result: {nvidia_answer} ({nvidia_status})")

        # Determine if failed (Incorrect or Error)
        gemini_failed = gemini_status != "Correct"
        gemma_failed = gemma_status != "Correct"
        nvidia_failed = nvidia_status != "Correct"

        # --- Prepare base data dictionary for CSV logging ---
        base_info = {
            'id': q_id,
            'dataset': args.dataset, # Add dataset name
            'question': question_data.get(config["question_field"], "N/A"),
            'correct_answer': correct_answer,
        }
        if task_type == 'mcqa' and options_dict:
             base_info['option_a'] = options_dict.get('A', '')
             base_info['option_b'] = options_dict.get('B', '')
             base_info['option_c'] = options_dict.get('C', '')
             base_info['option_d'] = options_dict.get('D', '')
        if config.get("explanation_field"):
             base_info['explanation'] = question_data.get(config["explanation_field"], '')

        # --- Save failures/errors ITERATIVELY ---
        if gemini_failed:
            gemini_failure_count += 1
            gemini_info = {**base_info,
                'gemini_model': gemini_model.model_name if gemini_model else "N/A",
                'gemini_answer': gemini_answer if gemini_answer is not None else "N/A",
                'gemini_raw_response': gemini_raw_response or "N/A",
                'gemini_status': gemini_status,
            }
            # Call the helper function to save this row
            save_failure_iteratively(gemini_info, gemini_output_path, gemini_column_order, "Gemini")
            # REMOVED: gemini_failed_data.append(gemini_info)

        if gemma_failed:
            gemma_failure_count += 1
            gemma_info = {**base_info,
                'gemma_model': gemma_model_id,
                'gemma_answer': gemma_answer if gemma_answer is not None else "N/A",
                'gemma_raw_response': gemma_raw_response or "N/A",
                'gemma_status': gemma_status,
            }
            # Call the helper function to save this row
            save_failure_iteratively(gemma_info, gemma_output_path, gemma_column_order, "Gemma")
            # REMOVED: gemma_failed_data.append(gemma_info)

        if nvidia_failed:
            nvidia_failure_count += 1
            nvidia_info = {**base_info,
                'nvidia_model': nvidia_model_id,
                'nvidia_answer': nvidia_answer if nvidia_answer is not None else "N/A",
                'nvidia_raw_response': nvidia_raw_response or "N/A",
                'nvidia_status': nvidia_status,
            }
            # Call the helper function to save this row
            save_failure_iteratively(nvidia_info, nvidia_output_path, nvidia_column_order, "NVIDIA")
            # REMOVED: nvidia_failed_data.append(nvidia_info)

        if not gemini_failed and not gemma_failed and not nvidia_failed:
             print(f"  -> All Processed Models Correct.")

    # --- REMOVED FINAL SAVING BLOCK ---
    # The code that created DataFrames from lists and saved them here is gone.

    print(f"\nEvaluation finished for dataset '{args.dataset}'. Processed {processed_count} questions.")
    # Report the counts based on iterative saving
    print(f"Total Gemini failures/errors saved to '{gemini_output_path}': {gemini_failure_count}")
    print(f"Total Gemma failures/errors saved to '{gemma_output_path}': {gemma_failure_count}")
    print(f"Total NVIDIA failures/errors saved to '{nvidia_output_path}': {nvidia_failure_count}")


# --- Main Execution (Handles Arguments) ---
if __name__ == "__main__":

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Evaluate LLMs (Gemini, Gemma, NVIDIA) on medical datasets with exponential backoff.")
    parser.add_argument("--dataset", "-d", required=True, choices=DATASET_CONFIGS.keys(),
                        help="Name of the dataset configuration to use.")
    parser.add_argument("--max_questions", "-n", type=int, default=None,
                        help="Maximum number of questions to process (default: process all).")
    # Add optional arguments for output file naming? For now, derive from dataset name.
    args = parser.parse_args()

    # Determine output file names based on dataset
    dataset_key = args.dataset
    gemini_output_file = f"gemini_failed_{dataset_key}.csv"
    gemma_output_file = f"gemma_failed_{dataset_key}.csv"
    nvidia_output_file = f"nvidia_deepseek_failed_{dataset_key}.csv"

    # --- Initialize API Clients ---
    gemini_model_instance = None
    if GOOGLE_API_KEY and genai: # Check if genai was imported successfully
        try:
            print("Configuring Google Generative AI...")
            genai.configure(api_key=GOOGLE_API_KEY)
            print(f"Loading Gemini Model: {GEMINI_MODEL_ID}")
            gemini_model_instance = genai.GenerativeModel(GEMINI_MODEL_ID)
            print("Google Generative AI initialized successfully.")
        except Exception as e:
            print(f"Warning: Error initializing Google Generative AI: {e}")
    elif not GOOGLE_API_KEY:
        print("Skipping Gemini initialization (GOOGLE_API_KEY not found).")
    else: # genai import failed
         print("Skipping Gemini initialization (google-generativeai library failed to import).")

    nvidia_client_instance = None
    if NVIDIA_API_KEY and OpenAI: # Check if OpenAI was imported successfully
        try:
            print("Configuring NVIDIA API Client...")
            nvidia_client_instance = OpenAI(base_url=NVIDIA_API_BASE_URL, api_key=NVIDIA_API_KEY)
            print("NVIDIA API Client initialized successfully.")
        except Exception as e:
            print(f"Warning: Error initializing NVIDIA API Client: {e}")
    elif not NVIDIA_API_KEY:
        print("Skipping NVIDIA client initialization (NVIDIA_API_KEY not found).")
    else: # openai import failed
        print("Skipping NVIDIA client initialization (openai library failed to import).")


    # Check if at least one API key is available for models intended to run
    if not gemini_model_instance and not OPENROUTER_API_KEY and not nvidia_client_instance:
         print("Error: No API keys/clients available for any service (Gemini, OpenRouter, NVIDIA). Exiting.")
         exit()

    # --- Load Data based on Args ---
    selected_dataset, selected_config = load_data(args.dataset)

    # --- Run Evaluation ---
    if selected_dataset and selected_config:
        # Optional: Delete existing output files before starting if you want a fresh run
        # try:
        #     if os.path.exists(gemini_output_file): os.remove(gemini_output_file)
        #     if os.path.exists(gemma_output_file): os.remove(gemma_output_file)
        #     if os.path.exists(nvidia_output_file): os.remove(nvidia_output_file)
        #     print("Cleared previous output files (if any).")
        # except OSError as e:
        #     print(f"Error deleting old output files: {e}")

        evaluate_questions(
            dataset=selected_dataset,
            config=selected_config, # Pass dataset config
            gemini_output_path=gemini_output_file, # Use dynamic filenames
            gemma_output_path=gemma_output_file,
            nvidia_output_path=nvidia_output_file,
            gemini_model=gemini_model_instance,
            gemma_api_key=OPENROUTER_API_KEY,
            gemma_model_id=GEMMA_MODEL_ID,
            nvidia_client=nvidia_client_instance,
            nvidia_model_id=NVIDIA_MODEL_ID,
            max_questions=args.max_questions # Use value from args
        )
    else:
        print(f"Exiting due to dataset loading failure for '{args.dataset}'.")