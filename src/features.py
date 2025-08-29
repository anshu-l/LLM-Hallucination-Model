import numpy as np
from sentence_transformers import SentenceTransformer
from .preprocess import simple_tokens

class SimilarityLens:
    def __init__(self, model_name: str):
        self.embedder = SentenceTransformer(model_name)

    def cosine_similarity(self, a: str, b: str) -> float:
        va = self.embedder.encode(a, normalize_embeddings=True)
        vb = self.embedder.encode(b, normalize_embeddings=True)
        return float(np.dot(va, vb))

    def lexical_overlap(self, a: str, b: str) -> float:
        A, B = set(simple_tokens(a)), set(simple_tokens(b))
        return len(A & B) / max(1, len(A | B))

    def length_ratio(self, a: str, b: str) -> float:
        ta, tb = simple_tokens(a), simple_tokens(b)
        return max(0.0, min(5.0, (len(tb)+1e-9)/(len(ta)+1e-9)))
