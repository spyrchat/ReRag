from mongodb_utils import connect_to_mongodb, MongoAtlasRetriever
from beir.datasets.data_loader import GenericDataLoader
from embeddings import TitanEmbeddingWrapper
import os

# Define the dataset path
corpus_path = "trec-covid"

# Ensure the dataset exists
if not os.path.exists(corpus_path):
    raise FileNotFoundError(
        f"Dataset not found at {corpus_path}. Please download the dataset.")

# Load the queries from the BEIR dataset
_, queries, _ = GenericDataLoader(corpus_path).load(split="test")
first_qid, first_query = next(iter(queries.items()))

# Connect to MongoDB
client = connect_to_mongodb()
collection = client["aws_gen_ai"]["TrecCovid"]

# Initialize the embedding wrapper
embedding_wrapper = TitanEmbeddingWrapper(model="amazon.titan-embed-text-v2:0")

# Initialize the MongoAtlasRetriever with the embedding wrapper
retriever = MongoAtlasRetriever(
    collection, embedding_wrapper, index_name="vector_search", top_k=5)

# Retrieve results for the first query
results = retriever.retrieve(first_query)

# Display the results
print(f"Query: {first_query}")
print("Top results:")
for i, doc in enumerate(results, 1):
    print(f"Rank {i}:")
    print(f"  doc_id: {doc.get('doc_id')}")
    print(f"  text: {doc.get('text')[:300]}...")  # truncate for preview
    print(f"  score: {doc.get('score')}")
