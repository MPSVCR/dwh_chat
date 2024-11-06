from typing import Any
import gradio as gr
from rag.chat_model import model
from rag.vector_store import vector_store


contextualize_system_message = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return as is. "
    "Reformulate it in CZECH."
)

def contextualize_user_message_with_history(message: str, history: list[dict[str, Any]]):
    """Take history and message and return a question that can be answered without the history."""
    messages = [
        {"role": "system", "content": contextualize_system_message}
    ]
    messages += [
        {"role": msg["role"], "content": msg["content"]} for msg in history
    ]
    messages += [{"role": "user", "content": message}]

    response = model.invoke(messages)
    return response.content

chatbot_system_message = (
    
)


def chatbot_response(message: str, history: list[dict[str, Any]]):
    self_contained_msg = contextualize_user_message_with_history(message, history)
    print(f"HISTORY SHORTENED TO MESSAGE: '{self_contained_msg}'")

    closest_documents = vector_store.similarity_search(self_contained_msg)
    system_prompt = "Use only knowledge in the provided context, do not use your own knowledge.\n"
    system_prompt += "Context:" + "\nContext:".join(d.page_content for d in closest_documents)
    
    
    messages_sent_to_bot = [{"role": "system", "content": system_prompt}] + [
        {"role": msg["role"], "content": msg["content"] }
        for msg in history
    ] + [{"role": "user", "content": message}]

    stream_response = model.stream(messages_sent_to_bot)
    total_text = ""
    for chunk in stream_response:
        total_text += chunk.content
        yield total_text

    total_text += f"\n\n\nREFERENCES: *{'\n'.join('-' + d.metadata['TLSource'] + ' -- ' + d.metadata['source'] for d in closest_documents)}*"
    yield {
        "role": "assistant",
        "content": total_text,
    }

gr.ChatInterface(chatbot_response, chatbot=gr.Chatbot(show_label=True, type="messages"), type="messages").launch()
