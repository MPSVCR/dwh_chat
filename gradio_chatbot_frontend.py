from typing import Any
import gradio as gr
from rag.chat_model import model
from rag.vector_store import load_markdown, vector_store

with open("data.md") as f:
    content = f.read()

load_markdown(content)


def random_response(message: str, history: list[dict[str, Any]]):
    closest_documents = vector_store.similarity_search(
        message
    )
    system_prompt = "Use only knowledge in the provided context, do not use your own knowledge.\n"
    system_prompt += "Context:" + "\nContext:".join(d.page_content for d in closest_documents)
    
    print("PROVIDED SYSTEM PROMPT:", system_prompt)
    
    messages_sent_to_bot = [{"role": "system", "content": system_prompt}] + [
        {"role": message["role"], "content": message["content"] }
        for message in history
    ] + [{"role": "user", "content": message}]

    stream_response = model.stream(messages_sent_to_bot)
    total_text = ""
    for chunk in stream_response:
        total_text += chunk.content
        yield total_text

gr.ChatInterface(random_response, type="messages").launch()
