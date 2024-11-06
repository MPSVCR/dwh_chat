import random
from typing import Any
import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage


model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)

def random_response(message: str, history: list[dict[str, Any]]):
    messages_sent_to_bot = [
        {"role": message["role"], "content": message["content"] }
        for message in history
    ] + [{"role": "user", "content": message}]

    stream_response = model.stream(messages_sent_to_bot)
    total_text = ""
    for chunk in stream_response:
        print("chunk:", chunk.content)
        total_text += chunk.content
        yield total_text

gr.ChatInterface(random_response, type="messages").launch()
