from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class HuggingFaceEmbedder(HuggingFaceEmbeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        super().__init__(
            model_name=model_name,
            model_kwargs={"device": device},
        )
