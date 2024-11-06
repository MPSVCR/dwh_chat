from typing import Any
import gradio as gr
from rag.chat_model import model

def random_response(message: str, history: list[dict[str, Any]]):
    messages_sent_to_bot = [
        {"role": message["role"], "content": message["content"] }
        for message in history
    ] + [{"role": "user", "content": message}]

    stream_response = model.stream(messages_sent_to_bot)
    total_text = ""
    for chunk in stream_response:
        total_text += chunk.content
        yield total_text

gr.ChatInterface(random_response, type="messages").launch()
