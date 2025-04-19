from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document


class BaseSplitter(ABC):
    @abstractmethod
    def split(self, docs: List[Document]) -> List[Document]:
        pass
