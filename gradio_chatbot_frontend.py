from typing import Any
import gradio as gr
from rag.chat_model import model
from rag.grader_model import evaluate_individual_contexts
from rag.vector_store import vector_store
from rag.contextualize import contextualize_user_message_with_history

chatbot_system_message = (
    "Use only knowledge in the provided context, do not use your own knowledge.\n"
)

def chatbot_response(message: str, history: list[dict[str, Any]]):
    self_contained_msg = contextualize_user_message_with_history(message, history)
    print(f"HISTORY SHORTENED TO MESSAGE: '{self_contained_msg}'")

    closest_documents = vector_store.similarity_search(self_contained_msg)
    closest_documents = evaluate_individual_contexts(self_contained_msg, closest_documents)

    system_prompt = str(chatbot_system_message)
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

    total_text += f"\n\n\n### REFERENCES:\n{'\n'.join('- *' + d.metadata['TLSource'] + ' -- ' + d.metadata['source'] + '*' for d in closest_documents)}"
    yield {
        "role": "assistant",
        "content": total_text,
    }


CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""

gr.ChatInterface(chatbot_response, css=CSS, chatbot=gr.Chatbot(elem_id="chatbot", show_label=True, type="messages", height="80%"), type="messages").launch()
