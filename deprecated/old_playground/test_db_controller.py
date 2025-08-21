from database import QdrantVectorDB

if __name__ == "__main__":
    db = QdrantVectorDB()
    db.init_collection(vector_size=384)
