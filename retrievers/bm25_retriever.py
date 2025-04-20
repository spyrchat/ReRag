from langchain.retrievers import BM25Retriever
from typing import List
from langchain.schema import Document


def get_bm25_retriever(docs: List[Document], k: int = 10):
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25
