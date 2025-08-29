import gradio as gr
from .config import load_config
from .features import SimilarityLens
from .nli import NLILens
from .fuse import Fusion
from .align import Aligner

cfg = load_config()
sim = SimilarityLens(cfg["paths"]["sentence_encoder"])
nli = NLILens(cfg["paths"]["nli_model"])
fusion = Fusion(cfg)
aligner = Aligner(cfg["paths"]["sentence_encoder"], cfg["thresholds"]["align_token_sim"])

def check(task, reference, hypothesis):
    # Lens A
    cos = sim.cosine_similarity(reference, hypothesis)
    ov  = sim.lexical_overlap(reference, hypothesis)
    # Lens B
    pe, pc, pn = nli.probs(reference, hypothesis)
    # Fuse
    prob = fusion.weighted_prob(cos, ov, pe, pc)
    label = "Hallucination" if prob >= cfg["thresholds"]["final_decision"] else "Not Hallucination"

    # Alignment-based highlights
    sus = set(aligner.suspicious_tokens(reference, hypothesis))
    marked_words = []
    for w in hypothesis.split():
        key = "".join(ch for ch in w.lower() if ch.isalnum())  # normalize
        if key in sus:
            marked_words.append(f"<span style='background:#ffcccc'>{w}</span>")
        else:
            marked_words.append(w)
    marked = " ".join(marked_words)

    diag = (f"Similarity={cos:.2f} | Overlap={ov:.2f} | "
            f"NLI: entail={pe:.2f}, contra={pc:.2f}, neutral={pn:.2f}")
    reason = f"Suspicious tokens (low semantic match): {', '.join(list(sus)[:10]) or '(none)'}"
    return label, float(prob), diag, marked, reason

def ui():
    with gr.Blocks(title="Two-Lens Detector — Similarity + NLI + Alignment") as demo:
        gr.Markdown("### Two-Lens: Similarity + Logic (NLI) + Alignment Highlights")
        task = gr.Dropdown(["MT","Paraphrase","Definition","(Other)"], value="(Other)", label="Task")
        reference = gr.Textbox(label="Reference (truth)", lines=4, value="Guido van Rossum created the Python programming language.")
        hypothesis = gr.Textbox(label="Answer to check", lines=4, value="James Gosling created the Python programming language.")
        btn = gr.Button("Check")
        label = gr.Label(label="Label")
        prob = gr.Slider(0,1,step=0.01,interactive=False,label="Probability of Hallucination")
        diag = gr.Textbox(label="Numbers used", interactive=False)
        marked = gr.HTML(label="Answer with highlights")
        reason = gr.Textbox(label="Why", interactive=False)
        btn.click(check, [task, reference, hypothesis], [label, prob, diag, marked, reason])
    return demo

if __name__ == "__main__":
    ui().launch()
    
    
    #heelo
