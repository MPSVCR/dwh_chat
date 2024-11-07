from typing import Any
import gradio as gr
from rag.chat_model import model
from rag.grader_model import get_relevant_db_metadata, get_relevant_meeting_data, get_relevant_wiki
from rag.contextualize import contextualize_user_message_with_history
import re

chatbot_system_message = (
    "You are a chatbot for the Ministry of Social Affairs, acting as a helpful assistant.\n"
    "Only use information provided in the given context, and do not rely on any external or prior knowledge.\n"
    "\n"
    "Guidelines:\n"
    "1. If the information needed to answer a question is not in the provided context, respond with:\n"
    "   'I do not have enough relevant information to answer the question.'\n"
    "2. Always maintain a concise and direct style in your responses. Aim to keep answers short and to the point.\n"
    "3. Avoid adding any assumptions or interpretations; stay strictly within the information provided.\n"
    "4. If any table is referenced, then please use its full name, the full names of the tables are very relevant.\n"
    "\n"
    "Remember: Context is your only source of knowledge for each response. Do not use personal insight or general knowledge.\n"
    "If you have any context, use it to answer the question.\n"
    "All responses should be provided in Czech.\n"
)


def chatbot_response(message: str, history: list[dict[str, Any]]):
    self_contained_msg = contextualize_user_message_with_history(message, history)
    print(f"HISTORY SHORTENED TO MESSAGE: '{self_contained_msg}'")

    closest_wiki_documents = get_relevant_wiki(self_contained_msg)
    closest_db_metadata = get_relevant_db_metadata(self_contained_msg)
    closest_meeting_metadata = get_relevant_meeting_data(self_contained_msg)

    system_prompt = ""
    
    if closest_wiki_documents:
        print("USING WIKI DOCUMENTS")
        system_prompt += "\nSTART OF PROVIDED CONTEXT:\n"
        system_prompt += "\nThese documents originate from our Wikipedia\n"
        system_prompt += "Wiki Context:\n"
        system_prompt += ("-" * 5 + "\n\nWiki Context:\n").join(d.page_content for d in closest_wiki_documents)

    if closest_db_metadata:
        print("USING DB METADATA")
        if system_prompt:
            system_prompt += "\n" + ("-" * 5)
        else:
            system_prompt += "\nSTART OF PROVIDED CONTEXT:\n"
        system_prompt += "\nHere are our database metadata that relate to the user's question\n"
        system_prompt += "DB metadata:\n"
        system_prompt += ("-" * 5 + "\n\nDB metadata:\n").join(d.page_content for d in closest_db_metadata)

    if closest_meeting_metadata:
        print("USING MEETING METADATA")
        if system_prompt:
            system_prompt += "\n" + ("-" * 5)
        else:
            system_prompt += "\nSTART OF PROVIDED CONTEXT:\n"
        system_prompt += (
            "\nHere are excerpts from meetings that relate to the user's question\n"
            "If you have the information about who said that information, then instead of just referencing the information,\n"
            "inform the user about the person or instance that originated that information. For example, if the information"
            " was told in a meeting by \"Vojtěch Šíp\" at time 00:00:00, tell that the information holds according to Vojtěch Šíp at time 00:00:00."
        )
        system_prompt += "\nMeeting excerpt:\n"
        system_prompt += ("-" * 5 + "\n\nMeeting excerpt:\n").join(d.page_content for d in closest_meeting_metadata)

    if not system_prompt:
        system_prompt = "Please respond that you have no relevant information for the user's query."
    else:
        system_prompt += "\nEnd of provided context\n\n"

    system_prompt = chatbot_system_message + system_prompt

    messages_sent_to_bot = [
        {"role": "system", "content": system_prompt}
    ] + [
        {"role": msg["role"], "content": re.sub(r"### REFERENCE.*", "", msg["content"]) }
        for msg in history
    ] + [
        {"role": "user", "content": message}
    ]

    stream_response = model.stream(messages_sent_to_bot)
    total_text = ""
    for chunk in stream_response:
        total_text += chunk.content
        yield total_text

    all_documents = closest_wiki_documents + closest_db_metadata + closest_meeting_metadata

    reference_names: list[str] = []
    for d in all_documents:

        name = ""
        if d.metadata['TLSource'] == "wiki" :    
            file_path = d.metadata['source'].removeprefix("/wiki")
            url = "https://dev.azure.com/mpsvcrtest/DWH/_git/wiki?path=" + file_path
            name = f"**WIKI**: [{d.metadata['source'].removeprefix("/wiki")}]({url})"
        elif d.metadata['TLSource'] == "meeting_transcript":
            name = f"**Meeting transcript**: source: [`{d.metadata['source']}`]({d.metadata['source_link']}) time: `{d.metadata['Timestamp']}`"
        elif d.metadata['TLSource'] == "db_metadata":
            name = f"**database**: `{d.metadata['source']}`"
        
        if name:
            reference_names.append(name)

    links = ""
    for name in reference_names:
        links += f"\n- {name}"
    total_text += f"\n\n\n### REFERENCE:\n{links}"
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
        height=300
    ),
    type="messages",
).launch(share=True, server_port=5000)
