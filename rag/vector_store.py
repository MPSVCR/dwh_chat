import os
from typing import Any
from langchain_postgres.vectorstores import PGVector
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

embeddings_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDER_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)

vector_store = PGVector(
    embeddings=embeddings_model,
    collection_name=os.environ["PGVECTOR_COLLECTION"],
    connection=os.environ["POSTGRES_CONNECTION"],
    use_jsonb=True,
    create_extension=False
)

_headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(_headers_to_split_on)

def load_markdown(markdown_str: str, metadata: dict[str, Any]):
    documents = markdown_splitter.split_text(markdown_str)
    for document in documents:
        document.metadata.update(metadata)
    vector_store.add_documents(documents)

