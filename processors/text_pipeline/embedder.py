import os
from embedding.factory import get_embedder
from dotenv import load_dotenv

load_dotenv(override=True)

class Embedder:
    def __init__(self):
        self.dense_model = get_embedder(os.getenv("DENSE_EMBEDDER"))
        self.sparse_model = get_embedder(os.getenv("SPARSE_EMBEDDER"))
