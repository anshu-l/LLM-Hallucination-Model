from .config import load_config
from .features import SimilarityLens
from .nli import NLILens
from .fuse import Fusion
from .data_io import load_shroom_dev

def main():
    cfg = load_config()
    sim = SimilarityLens(cfg["paths"]["sentence_encoder"])
    nli = NLILens(cfg["paths"]["nli_model"])
    fusion = Fusion(cfg)

    dev_data = load_shroom_dev("data/dev")
    correct, total = 0, 0

    for ex in dev_data[:200]:  # test on first 200 for speed
        cos = sim.cosine_similarity(ex["reference"], ex["hypothesis"])
        ov  = sim.lexical_overlap(ex["reference"], ex["hypothesis"])
        pe, pc, pn = nli.probs(ex["reference"], ex["hypothesis"])
        prob = fusion.weighted_prob(cos, ov, pe, pc)
        pred = 1 if prob >= cfg["thresholds"]["final_decision"] else 0

        if ex["label"] in [0,1]:
            total += 1
            if pred == ex["label"]:
                correct += 1

    print(f"Accuracy on first {total} dev examples: {correct/total:.2%}")

if __name__ == "__main__":
    main()
