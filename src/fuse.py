class Fusion:
    def __init__(self, cfg):
        self.cfg = cfg

    def weighted_prob(self, cos_sim, overlap, p_entail, p_contra):
        """
        Combine:
        - similarity (want high)
        - lexical overlap (want high)
        - NLI contradiction vs entailment (want low)
        into a single hallucination probability in [0,1].
        """
        w = self.cfg["weights"]
        raw = (
            w["w_cosine_inverse"] * (1.0 - cos_sim) +
            w["w_overlap_inverse"] * (1.0 - overlap) +
            w["w_nli_contra_delta"] * max(0.0, p_contra - p_entail)
        )
        return max(0.0, min(1.0, raw))
