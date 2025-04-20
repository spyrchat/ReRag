# playground/test_bm25_rag.py

from embedding.bedrock_embeddings import TitanEmbedder
from utils.embedding_wrapper import TitanLangchainWrapper
from langchain.vectorstores import Qdrant
from dotenv import load_dotenv
import os
import logging

from langchain.retrievers import BM25Retriever
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
# Or your storage for raw docs
from database.qdrant_controller import QdrantVectorDB

logging.basicConfig(level=logging.INFO)
load_dotenv()

# === 1. Load all documents ===
db = QdrantVectorDB()
client = db.get_client()

# Fetch all stored docs (replace this with however you stored them)

titan = TitanEmbedder()
embedding = TitanLangchainWrapper(titan)

vectordb = Qdrant(
    client=client,
    collection_name=db.get_collection_name(),
    embedding_function=embedding
)

all_docs = vectordb.similarity_search("a", k=10_000)  # Fetch a large number

# === 2. Build BM25 retriever ===
bm25 = BM25Retriever.from_documents(all_docs)
bm25.k = 10

# === 3. Use Gemini LLM ===
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.4,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# === 4. Create QA chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=bm25,
    return_source_documents=True
)

# === 5. Run test query ===
query = "What can we learn about electric vehicle charging patterns?"
result = qa_chain(query)

print("\n=== FINAL ANSWER ===")
print(result["result"])

print("\n=== SOURCES ===")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source", "unknown"))
