from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass
