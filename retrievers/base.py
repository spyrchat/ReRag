from abc import ABC, abstractmethod
from langchain.schema import Document
from typing import List


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        pass
