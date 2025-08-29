from typing import Tuple, Dict
import numpy as np
from sentence_transformers import CrossEncoder

class NLILens:
    def __init__(self, model_name: str):
        # loads local model (cached after first time)
        self.model = CrossEncoder(model_name)
        # try to read the model's id2label mapping safely
        try:
            self.id2label: Dict[int, str] = self.model.model.config.id2label
            # normalize label strings to lowercase for safety
            self.id2label = {i: lbl.lower() for i, lbl in self.id2label.items()}
        except Exception:
            # fallback guess if config is missing (rare)
            self.id2label = {0: "entailment", 1: "contradiction", 2: "neutral"}

    def _probs_one_order(self, a: str, b: str) -> Dict[str, float]:
        """
        Runs NLI on the pair (a, b) and returns a dict:
        {'entailment': pe, 'contradiction': pc, 'neutral': pn}
        using the model's actual label mapping.
        """
        scores = self.model.predict([[a, b]], apply_softmax=True)[0]
        # ensure numpy array
        scores = np.asarray(scores, dtype=float)
        # map by id2label
        out = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for idx, p in enumerate(scores):
            lbl = self.id2label.get(idx, str(idx)).lower()
            if "entail" in lbl:
                out["entailment"] = float(p)
            elif "contra" in lbl:
                out["contradiction"] = float(p)
            elif "neutral" in lbl:
                out["neutral"] = float(p)
        return out

    def probs(self, reference: str, hypothesis: str) -> Tuple[float, float, float]:
        """
        Robust NLI:
        - evaluate both orders: (ref, hyp) and (hyp, ref)
        - take the max contradiction and max entailment across orders (neutral = min)
        This handles models with different label orders AND directionality.
        """
        fwd = self._probs_one_order(reference, hypothesis)   # ref -> hyp
        rev = self._probs_one_order(hypothesis, reference)   # hyp -> ref

        p_entail = max(fwd["entailment"], rev["entailment"])
        p_contra = max(fwd["contradiction"], rev["contradiction"])
        p_neutral = min(fwd["neutral"], rev["neutral"])  # conservative

        return float(p_entail), float(p_contra), float(p_neutral)
