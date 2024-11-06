import json
from rag.chat_model import model
from langchain_core.documents import Document

def evaluate_individual_contexts(self_contained_msg: str, closest_documents: list[Document]):
    template=f"""You are a grader assessing relevance of retrieved documents to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Provide a JSON of array of dicts with a single key 'score' for each document and no preamble or explanation.
Here is the user question: {self_contained_msg}"""
    
    for i, document in enumerate(closest_documents):
        template += f"\nDOCUMENT {i + 1}:\n" + document.page_content + "\n"

    result = model.invoke([{"role": "user", "content": template}])

    try:
        relevances = json.loads(result.content)

        relevant_documents: list[Document] = []
        for i, relevance in enumerate(relevances):
            if relevance["score"] == "yes":
                relevant_documents.append(closest_documents[i])

        return relevant_documents
    except:
        print("GRADING FAILED, using all the documents")
        return closest_documents
    