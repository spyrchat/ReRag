from abc import ABC, abstractmethod


class BaseVectorDB(ABC):
    @abstractmethod
    def init_collection(self, vector_size: int):
        pass

    @abstractmethod
    def get_client(self):
        pass

    @abstractmethod
    def get_collection_name(self):
        pass
