import json
from rag.chat_model import model
from rag.vector_store import vector_store
from langchain_core.documents import Document

def _get_relevant_data(self_contained_msg: str, template: str, chunk_name: str, tl_source: str):
    closest_documents = vector_store.similarity_search(
        self_contained_msg,
        k=10,
        filter={"TLSource": tl_source}
    )
    closest_documents = evaluate_individual_contexts(self_contained_msg, closest_documents, template, chunk_name)
    return closest_documents

def get_relevant_db_metadata(self_contained_msg: str):
    template = """You are a grader assessing relevance of retrieved database metadata to a user question. We have a list
of database metadata files, which may contain the answer or which may relate to the posed question. Please, check the
comments and the table names in order to assess whether the table may contain relevant data.

Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Provide a JSON of array of dicts with a single key 'score' for each db metadata and no preamble or explanation.
Here is the user question: '{user_question}'"""
    return _get_relevant_data(
        self_contained_msg,
        template,
        chunk_name="START OF DB METADATA",
        tl_source="db_metadata"
    )

def get_relevant_wiki(self_contained_msg: str):
    template = """You are a grader assessing relevance of retrieved documents to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Provide a JSON of array of dicts with a single key 'score' for each document and no preamble or explanation.
Here is the user question: '{user_question}'"""
    return _get_relevant_data(
        self_contained_msg,
        template,
        chunk_name="WIKI DOCUMENT",
        tl_source="wiki"
    )


def evaluate_individual_contexts(
    self_contained_msg: str,
    closest_documents: list[Document],
    template: str,
    chunk_name: str
):
    template=template.replace("{user_question}", self_contained_msg)
    
    for i, document in enumerate(closest_documents):
        template += f"\n{chunk_name} {i + 1}:\n" + document.page_content + "\n"

    result = model.invoke([{"role": "user", "content": template}])

    try:
        relevances = json.loads(result.content)
    except:
        print(f"GRADING FAILED ('{chunk_name}'), using all the documents - grading: {result.content}")
        return closest_documents

    if not isinstance(relevances, list):
        print(f"GRADING FAILED ('{chunk_name}'), using all the documents - grading: {result.content}")
        return

    relevant_documents: list[Document] = []
    for i, relevance in enumerate(relevances):
        if "score" not in relevance or relevance["score"] == "yes":
            relevant_documents.append(closest_documents[i])


    return relevant_documents
    