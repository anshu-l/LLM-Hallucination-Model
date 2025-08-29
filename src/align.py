from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from .preprocess import simple_tokens

class Aligner:
    def __init__(self, model_name: str, token_sim_threshold: float):
        self.emb = SentenceTransformer(model_name)
        self.tau = token_sim_threshold  # e.g., 0.5

    def suspicious_tokens(self, reference: str, hypothesis: str) -> List[str]:
        ref_t = simple_tokens(reference)
        hyp_t = simple_tokens(hypothesis)
        if not ref_t or not hyp_t:
            return []
        ref_vecs = self.emb.encode(ref_t, normalize_embeddings=True)
        hyp_vecs = self.emb.encode(hyp_t, normalize_embeddings=True)

        sus = []
        for i, hv in enumerate(hyp_vecs):
            sims = np.dot(ref_vecs, hv)       # similarity to every ref token
            if float(np.max(sims)) < self.tau:  # no good match -> suspicious
                sus.append(hyp_t[i])
        return sus
