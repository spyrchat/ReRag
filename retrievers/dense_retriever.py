from langchain.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings


def get_qdrant_retriever(client, collection_name: str, embedding: Embeddings, k: int = 10):
    vectordb = Qdrant(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever
