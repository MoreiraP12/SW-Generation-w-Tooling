import re
import time
import torch
import gradio as gr
from pathlib import Path
from threading import Thread
from datetime import datetime
from unsloth import FastModel, get_chat_template
from transformers import TextIteratorStreamer
from deep_research.query import process_query
from deep_research.search import search
from deep_research.crawler import WebCrawler

# Create a timestamped log file
log_dir = Path("./logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"

def log(text):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

class StreamingResearchPipeline:
    def __init__(self, model_name="unsloth/gemma-3-4b-it", cache_dir="./cache"):
        print(f"Loading model: {model_name}")
        start = time.time()
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False
        )
        self.tokenizer = get_chat_template(self.tokenizer, chat_template="gemma-3")
        print(f"Model loaded in {time.time() - start:.2f}s")

        self.crawler = WebCrawler(
            cache_dir=cache_dir,
            respect_robots=True,
            requests_per_second=1,
            max_depth=2,
            max_pages_per_site=5,
            summarization_model="facebook/bart-large-cnn",
            relevance_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            qa_model="deepset/roberta-base-squad2"
        )

        self.system_prompt = """
            You are an AI assistant with real-time research capabilities. When you need to search for information, 
            use the <search> </search> tags. You should thoroughly research any topic 
            before providing your final answer.
            
            Follow these guidelines:
            1. Use  tags ONLY for actual web searches, not when discussing the tags themselves.
            2. Perform as many searches as needed to gather comprehensive information.
            3. Use specific, focused search queries rather than broad ones.
            4. After completing all necessary research, provide your final answer between tags <answer> </answer>.
            5. Include citations to your sources in the final answer.
            6. If answering factual questions, ensure your answer is based on the research results.
            
            Example usage:
            User: "What are the latest advancements in quantum computing?"
            
            You: "I'll research this topic for you.
            latest advancements in quantum computing 2025
            
            Let me check for more specific information about quantum error correction.
            quantum error correction breakthroughs 2025
            
            Now I'll look for information about quantum advantage demonstrations.
            quantum computational advantage demonstrations 2025
        """

    def perform_research(self, query):
        log(f"[SEARCH]: {query}")
        try:
            parsed = process_query(query)
            results = search(parsed["parsed_query"], engines=["duckduckgo"], max_results=10)
            answer = self.crawler.generate_definitive_answer(query, results)

            out = answer["answer"] + "\n\nSources:\n"
            for src in answer["sources"]:
                title = getattr(src, "title", src.get("title", "Untitled") if isinstance(src, dict) else "Untitled")
                url = getattr(src, "url", src.get("url", "") if isinstance(src, dict) else "")
                out += f"- {title}: {url}\n"
            log("[ANSWER]: " + out)
            return out
        except Exception as e:
            err = f"[SEARCH ERROR]: {str(e)}"
            log(err)
            return err

    def create_prompt(self, user_input, history):
        messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        for user, bot in history:
            messages.append({"role": "user", "content": [{"type": "text", "text": user}]})
            if bot:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": bot}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def generate_with_debug(self, prompt, chat_history=None):
        prompt = self.create_prompt(prompt, chat_history or [])
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        kwargs = dict(input_ids=input_ids, max_new_tokens=2048, streamer=streamer)
        Thread(target=self.model.generate, kwargs=kwargs).start()

        final_output = ""
        buffer = ""
        search_buffer = ""
        in_search = False

        for token in streamer:
            final_output += token
            buffer += token
            yield final_output, f"[TOKEN] {token}"

            if "<search>" in buffer:
                in_search = True
                search_buffer = ""
                buffer = ""
                yield final_output + "\n[🔍 Searching...]", "[TAG] Detected <search>"

            elif in_search and "</search>" in buffer:
                query = search_buffer.strip()
                yield final_output + f"\n[🔍 Searching for: {query}]", f"[RESEARCH] {query}"
                result = self.perform_research(query)
                final_output += "\n" + result
                yield final_output, "[RESEARCH DONE]"
                in_search = False
                buffer = ""

            elif in_search:
                search_buffer += token

            if not in_search:
                buffer = buffer[-30:]

pipeline = StreamingResearchPipeline()

def user_prompt(message, history):
    return message, history + [[message, None]]

def bot_response(history):
    user_input = history[-1][0]
    prev = history[:-1]
    for out, dbg in pipeline.generate_with_debug(user_input, prev):
        history[-1][1] = out
        yield history, dbg

with gr.Blocks() as demo:
    gr.Markdown("## 🔎 Real-time Search-Aware Chatbot (Gemma3 + Deep Research)")
    with gr.Row():
        with gr.Column(scale=3):
            chatbox = gr.Chatbot([], label="Chat")
            input_box = gr.Textbox(placeholder="Ask me anything...", show_label=False)
            clear_btn = gr.Button("Clear")
        with gr.Column(scale=2):
            debug_panel = gr.Textbox(label="Debug Info", lines=30)

    input_box.submit(user_prompt, [input_box, chatbox], [input_box, chatbox], queue=False).then(
        bot_response, [chatbox], [chatbox, debug_panel]
    )
    clear_btn.click(lambda: ([], ""), None, [chatbox, debug_panel])

demo.launch(share=True)