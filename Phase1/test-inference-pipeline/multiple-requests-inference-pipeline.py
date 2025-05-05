import re
import time
import json
import torch
import gradio as gr
from pathlib import Path
from unsloth import FastModel, get_chat_template
from deep_research.query import process_query
from deep_research.search import search
from deep_research.crawler import WebCrawler

# --- Constants ---
MAX_STEPS = 5

class CoTResearchPipeline:
    def __init__(self, model_name="unsloth/gemma-3-4b-it", cache_dir="./cache"):
        print(f"Loading model: {model_name}")
        start_time = time.time()
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=8192,
            load_in_4bit=True,
        )
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma-3",
            mapping={"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
        )
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

        print("Initializing research crawler...")
        # Using the updated max_depth from user's provided snippet
        self.crawler = WebCrawler(
            cache_dir=cache_dir,
            respect_robots=True,
            requests_per_second=1,
            max_depth=5, # User provided max_depth=5
            max_pages_per_site=3,
            summarization_model="facebook/bart-large-cnn",
            relevance_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            qa_model="deepset/roberta-base-squad2"
        )

        # System prompt includes example of concise search query
        self.system_prompt = """
You are an AI assistant that solves problems step-by-step, using web searches when necessary.
Your goal is to reach a final answer to the user's question through a clear chain of thought.

Follow these guidelines STRICTLY:
1. Analyze the user's question and break it down into smaller, manageable steps.
2. For each step, explain your reasoning briefly.
3. If you need information you don't have, formulate a SPECIFIC search query.
4. Output ONLY ONE search query per step, enclosed in <search> tags, like: <search>your specific query</search>. The specific query search should be concise and meaningful for the deep search, for example <search>Cristiano Ronaldo's mom name </search>.
5. STOP generating after outputting the <search> tags for the current step. Do not add extra text after the tags in that step.
6. After you generate a step with <search>, the result will be appended to that step's text, marked with "[Research Result]". Check the end of your own previous turn in the history for this marker before generating the next step.
7. Use the information from "[Research Result]" to inform your reasoning for the next step.
8. If you believe you have enough information to answer the original question fully, provide the final answer enclosed in <answer> tags, like: <answer>Your final detailed answer.</answer>. Do NOT use <search> tags in the same step as the <answer> tags.
9. If you cannot find information or get stuck, explain the issue clearly within <answer> tags.
"""

    # Method with detailed terminal logging as requested
    def perform_research(self, query):
        # Log the search query attempt to console
        print(f"--- Attempting Research for Query ---")
        print(f"  Query: '{query}'")
        print(f"-------------------------------------")
        time.sleep(0.5)
        search_results_list = [] # To collect items for the crawler

        try:
            # Step 1: Get search results (list of URLs/dicts)
            results = search(query, engines=['duckduckgo'], max_results=5) # Can adjust max_results

            # --- Live Terminal Logging of URLs Found by search()---
            print(f"--- URLs Found by Search Engine (Potential Input) ---")
            urls_found_count = 0
            # Robustly extract URLs and prepare input for crawler
            if isinstance(results, dict) and 'results' in results:
                search_input_data = results.get('results', [])
                if isinstance(search_input_data, list):
                    for item in search_input_data:
                        url = None
                        title = 'N/A'
                        if isinstance(item, dict):
                            url = item.get('url')
                            title = item.get('title', 'N/A')
                        elif isinstance(item, str):
                            url = item
                        if url:
                            print(f"  [Potential URL]: {url}") # Log URL found by search
                            urls_found_count += 1
                            # Append dict format for crawler
                            search_results_list.append({'url': url, 'title': title})
                else:
                    print(f"  [WARN] 'results' key found but value is not a list: {type(search_input_data)}")
            elif isinstance(results, list):
                print(f"  [INFO] Search returned a direct list.")
                for item in results:
                    url = None
                    title = 'N/A'
                    if isinstance(item, dict):
                         url = item.get('url')
                         title = item.get('title', 'N/A')
                    elif isinstance(item, str):
                         url = item
                    if url:
                        print(f"  [Potential URL]: {url}") # Log URL found by search
                        urls_found_count += 1
                        search_results_list.append({'url': url, 'title': title})
            elif results:
                 print(f"  [WARN] Unexpected search result format: {type(results)}")

            if urls_found_count == 0:
                 print("  No URLs found by search engine.")
            print("--- End URLs Found by Search Engine ---")
            # --- End Live Logging ---

            if not search_results_list:
                 print("[INFO] No usable search results to process with crawler.")
                 return "[Research Result]: No search results found."

            # --- Log URLs being passed to crawler ---
            print(f"\n--- URLs Being Passed to Crawler for '{query}' ---")
            if search_results_list:
                for idx, item in enumerate(search_results_list):
                    url_to_crawl = item.get('url', 'N/A')
                    print(f"  [Input {idx+1}]: {url_to_crawl}")
            else:
                 print("  (No URLs to pass to crawler)")
            print(f"--- End URLs Passed to Crawler ---\n")
            # --- End Log ---

            # Step 2: Process found URLs with the crawler
            print(f"-> Starting crawler processing for '{query}'...")
            answer_data = self.crawler.generate_definitive_answer(query, search_results=search_results_list)
            print(f"<- Finished crawler processing for '{query}'.")

            # Prepare the text part *without* sources for the LLM return value
            if answer_data and answer_data.get('answer'):
                 result_text_for_llm = f"[Research Result]: {answer_data['answer']}"
            else:
                 print("[WARN] Crawler processed but returned no answer.")
                 return "[Research Result]: Could not extract a definitive answer from search results."

            # --- Console Logging of Final Answer ---
            print(f"\n--- Final Processed Result (for query: '{query}') ---")
            print(f"  Answer Text Returned to LLM Context: {answer_data['answer']}")
            # Note: We are not logging answer_data['sources'] here anymore based on the request
            print(f"--- End Final Processed Result Log ---\n")
            # --- End Console Logging ---

            # Return only the clean text result to the main loop
            return result_text_for_llm

        except Exception as e:
            print(f"[ERROR] Research/Crawl failed for '{query}': {type(e).__name__} - {e}")
            return f"[Research Result]: Error during research - {type(e).__name__}"


    def _extract_last_search_query(self, text):
        matches = list(re.finditer(r"<search>(.*?)</search>", text, re.DOTALL | re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()
        return None

    def _contains_final_answer(self, text):
        return bool(re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE))

    def generate_cot_chain_streaming(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        chat_history_for_display = [[user_input, ""]]
        current_assistant_response = ""
        debug_log = "Starting CoT Generation...\n"

        yield chat_history_for_display, debug_log

        final_answer_found = False

        for step in range(MAX_STEPS):
            step_debug_log = f"--- Step {step + 1} ---\n"
            print(f"\n--- Step {step + 1}: Generating LLM Response ---") # Terminal log

            last_assistant_message_index = -1
            if messages and messages[-1]['role'] == 'assistant':
                last_assistant_message_index = len(messages) -1

            try:
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                 print(f"[ERROR] Error applying chat template: {e}")
                 step_debug_log += f"Error applying chat template: {e}\n"
                 current_assistant_response += "[ERROR: Prompt formatting failed]"
                 chat_history_for_display[-1][1] = current_assistant_response
                 debug_log += step_debug_log
                 yield chat_history_for_display, debug_log
                 break

            step_debug_log += f"Prompt for LLM (last 700 chars):\n...{prompt_text[-700:]}\n" # Keep for Gradio debug

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                llm_output_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

            except Exception as e:
                print(f"[ERROR] Error during LLM generation: {e}")
                step_debug_log += f"Error during LLM generation: {e}\n"
                current_assistant_response += "[ERROR: LLM Generation failed]"
                chat_history_for_display[-1][1] = current_assistant_response
                debug_log += step_debug_log
                yield chat_history_for_display, debug_log
                break

            print(f"LLM Output (Step {step + 1}):\n{llm_output_text}") # Terminal log
            step_debug_log += f"LLM Output:\n{llm_output_text}\n" # Gradio debug log

            current_assistant_response += f"Assistant Step {step + 1}:\n{llm_output_text}\n\n"
            current_llm_message = {"role": "assistant", "content": llm_output_text}
            messages.append(current_llm_message)
            chat_history_for_display[-1][1] = current_assistant_response
            debug_log += step_debug_log

            yield chat_history_for_display, debug_log

            if self._contains_final_answer(llm_output_text):
                print("Final answer tag detected by script. Ending chain.") # Terminal log
                debug_log += "Final answer tag detected by script. Ending chain.\n" # Debug panel log
                final_answer_found = True
                break

            search_query = self._extract_last_search_query(llm_output_text)

            if search_query:
                # Perform research - detailed logging happens inside perform_research
                research_result_text_for_llm = self.perform_research(search_query)

                # Add result text to DEBUG PANEL log only
                step_debug_log = f"Research Result (appended to history):\n{research_result_text_for_llm}\n"
                debug_log += step_debug_log

                # Append result to the LLM message history content
                current_llm_message['content'] += f"\n\n{research_result_text_for_llm}"

                # Append the result text to the UI display string
                current_assistant_response += f"{research_result_text_for_llm}\n\n"
                # Update the display history for Gradio
                chat_history_for_display[-1][1] = current_assistant_response

                # Yield the updated state after research info added
                yield chat_history_for_display, debug_log
            else:
                print("No search query or final answer tag found in LLM output. Ending chain.") # Terminal log
                debug_log += "No search query or final answer tag found in LLM output. Ending chain.\n" # Debug panel log
                break

        # --- After the loop ---
        loop_ended_reason = ""
        if final_answer_found:
             loop_ended_reason = "Final answer provided by LLM."
        elif step == MAX_STEPS - 1:
             loop_ended_reason = f"Reached maximum steps ({MAX_STEPS})."
             current_assistant_response += f"[INFO: {loop_ended_reason}]\n\n"
        else:
             loop_ended_reason = "Stopping - no further action identified by LLM."
             current_assistant_response += f"[INFO: {loop_ended_reason}]\n\n"

        debug_log += f"Loop ended: {loop_ended_reason}\n"
        print(f"Loop ended: {loop_ended_reason}") # Terminal log

        # No final citation block added to UI

        # Update the display history with the final response
        chat_history_for_display[-1][1] = current_assistant_response

        # Yield the absolute final state for the UI
        yield chat_history_for_display, debug_log


# --- Gradio Interface Setup ---
pipeline = CoTResearchPipeline()

def bot_response_cot_streaming(user_input):
    if not user_input:
        yield [["", "Please enter a question."]], "No input provided."
        return
    yield from pipeline.generate_cot_chain_streaming(user_input)


with gr.Blocks() as demo:
    gr.Markdown("## 🔎 Streaming Multi-Step CoT Research Bot (Detailed Terminal Logs)") # Updated title
    with gr.Row():
        with gr.Column(scale=3):
            chatbox = gr.Chatbot([], label="CoT Transcript", height=600, sanitize_html=False) # sanitize can be True now
            input_box = gr.Textbox(placeholder="Ask me anything...", show_label=False)
            submit_btn = gr.Button("Generate CoT Response")
            clear_btn = gr.Button("Clear")
        with gr.Column(scale=2):
            debug_panel = gr.Textbox(label="Debug Info / Gradio Log", lines=35, interactive=False)

    submit_btn.click(
        bot_response_cot_streaming,
        [input_box],
        [chatbox, debug_panel]
    )
    clear_btn.click(lambda: ([], "", ""), None, [chatbox, input_box, debug_panel])

demo.launch(share=True)