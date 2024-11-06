import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

embeddings_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDER_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)

vector_store = InMemoryVectorStore(embedding=embeddings_model)

_headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(_headers_to_split_on)

def load_markdown(markdown_str: str):
    documents = markdown_splitter.split_text(markdown_str)
    vector_store.add_documents(documents)

print(vector_store.similarity_search("Pear", k=1))
