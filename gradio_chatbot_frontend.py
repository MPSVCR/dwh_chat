from typing import Any
import gradio as gr
from rag.chat_model import model
from rag.grader_model import get_relevant_db_metadata, get_relevant_wiki
from rag.contextualize import contextualize_user_message_with_history

chatbot_system_message = (
    "Use only knowledge in the provided context, do not use your own knowledge.\n"
)

def chatbot_response(message: str, history: list[dict[str, Any]]):
    self_contained_msg = contextualize_user_message_with_history(message, history)
    print(f"HISTORY SHORTENED TO MESSAGE: '{self_contained_msg}'")

    closest_wiki_documents = get_relevant_wiki(self_contained_msg)
    closest_db_metadata = get_relevant_db_metadata(self_contained_msg)

    system_prompt = str(chatbot_system_message)
    
    if closest_wiki_documents:
        system_prompt += "These documents originate from our Wikipedia\n"
        system_prompt += "Context:\n"
        system_prompt += "\n\nContext:\n".join(d.page_content for d in closest_wiki_documents)

    if closest_db_metadata:
        system_prompt += "Here are our database metadata that relate to the user's question\n"
        system_prompt += "DB metadata:\n"
        system_prompt += "\n\nDB metadata:\n".join(d.page_content for d in closest_db_metadata)

    messages_sent_to_bot = [{"role": "system", "content": system_prompt}] + [
        {"role": msg["role"], "content": msg["content"] }
        for msg in history
    ] + [{"role": "user", "content": message}]

    stream_response = model.stream(messages_sent_to_bot)
    total_text = ""
    for chunk in stream_response:
        total_text += chunk.content
        yield total_text

    all_documents = closest_wiki_documents + closest_db_metadata

    total_text += f"\n\n\n### REFERENCES:\n{'\n'.join('- *' + d.metadata['TLSource'] + ' -- ' + d.metadata['source'] + '*' for d in all_documents)}"
    yield {
        "role": "assistant",
        "content": total_text,
    }

gr.ChatInterface(
    chatbot_response,
    chatbot=gr.Chatbot(
        elem_id="chatbot",
        show_label=True,
        type="messages",
        height=800
    ),
    type="messages",
).launch()
