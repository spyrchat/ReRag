import re
from typing import List, Dict
from rank_bm25 import BM25Okapi


def simple_tokenizer(text: str) -> List[str]:
    """Tokenizes by splitting on word boundaries."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25SparseEmbedder:
    """
    BM25-based sparse embedder for use with Qdrant hybrid search.
    Returns sparse vectors in the format Qdrant expects.
    """

    def __init__(self, corpus: List[str], tokenizer=simple_tokenizer):
        self.tokenizer = tokenizer
        self.tokenized_corpus = [self.tokenizer(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Map tokens to indices in the BM25 vocabulary
        vocab_set = set()
        for doc in self.tokenized_corpus:
            vocab_set.update(doc)
        self.vocab = {token: idx for idx,
                      token in enumerate(sorted(vocab_set))}

    def get_vector(self, text: str) -> Dict[str, List[float]]:
        """
        Generate a Qdrant-compatible sparse vector for a single text input.
        """
        tokens = self.tokenizer(text)
        doc_scores = self.bm25.get_scores(tokens)

        indices = []
        values = []

        for i, score in enumerate(doc_scores):
            if score > 0:
                indices.append(i)
                values.append(score)

        return {"indices": indices, "values": values}

    def get_vectors(self, texts: List[str]) -> List[Dict[str, List[float]]]:
        return [self.get_vector(text) for text in texts]
