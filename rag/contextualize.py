from rag.chat_model import model
from typing import Any

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