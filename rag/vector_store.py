import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document

embeddings_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDER_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)

documents = [
    Document(t) for t in [
        "Apple. Iphone is great",
        "Pear. Microsoft is great",
        "Banana. Amazon is great"
    ]
]

vector_store = InMemoryVectorStore(embedding=embeddings_model)
vector_store.add_documents(documents=documents)

print(vector_store.similarity_search("Pear", k=1))
