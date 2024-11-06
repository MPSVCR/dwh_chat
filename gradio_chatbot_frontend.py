from typing import Any
import gradio as gr
from rag.chat_model import model
from rag.grader_model import get_relevant_db_metadata, get_relevant_wiki
from rag.contextualize import contextualize_user_message_with_history

chatbot_system_message = (
    "You are a chatbot for the Ministry of Social Affairs, acting as a helpful assistant.\n"
    "Only use information provided in the given context, and do not rely on any external or prior knowledge.\n"
    "\n"
    "Guidelines:\n"
    "1. If the information needed to answer a question is not in the provided context, respond with:\n"
    "   'I do not have enough relevant information to answer the question.'\n"
    "2. Always maintain a concise and direct style in your responses. Aim to keep answers short and to the point.\n"
    "3. Your users are analysts, developers, and business analysts, so respond with clarity and precision, using professional language.\n"
    "4. Avoid adding any assumptions or interpretations; stay strictly within the information provided.\n"
    "5. If the question is unclear or incomplete, ask for clarification instead of attempting an answer.\n"
    "\n"
    "Remember: Context is your only source of knowledge for each response. Do not use personal insight or general knowledge.\n"
    "All responses should be provided in Czech.\n"
)


def chatbot_response(message: str, history: list[dict[str, Any]]):
    self_contained_msg = contextualize_user_message_with_history(message, history)
    print(f"HISTORY SHORTENED TO MESSAGE: '{self_contained_msg}'")

    closest_wiki_documents = get_relevant_wiki(self_contained_msg)
    closest_db_metadata = get_relevant_db_metadata(self_contained_msg)

    system_prompt = ""
    
    if closest_wiki_documents:
        system_prompt += "These documents originate from our Wikipedia\n"
        system_prompt += "Context:\n"
        system_prompt += "\n\nContext:\n".join(d.page_content for d in closest_wiki_documents)

    if closest_db_metadata:
        system_prompt += "Here are our database metadata that relate to the user's question\n"
        system_prompt += "DB metadata:\n"
        system_prompt += "\n\nDB metadata:\n".join(d.page_content for d in closest_db_metadata)

    if not system_prompt:
        system_prompt = chatbot_system_message + "Please respond that you have no relevant information for the user's query."

    system_prompt = chatbot_system_message + system_prompt

    messages_sent_to_bot = [
        {"role": "system", "content": system_prompt}
    ] + [
        {"role": msg["role"], "content": msg["content"] }
        for msg in history
    ] + [
        {"role": "user", "content": message}
    ]

    stream_response = model.stream(messages_sent_to_bot)
    total_text = ""
    for chunk in stream_response:
        total_text += chunk.content
        yield total_text

    all_documents = closest_wiki_documents + closest_db_metadata

    total_text += f"\n\n\n### REFERENCE:\n{'\n'.join('- *' + d.metadata['TLSource'] + ' -- ' + d.metadata['source'] + '*' for d in all_documents)}"
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
