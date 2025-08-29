import json
import os
from typing import Iterator, Dict

def read_jsonl(path: str) -> Iterator[Dict]:
    """Reads a JSONL file line by line"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_shroom_dev(dev_folder: str):
    """
    Loads SHROOM dev data from folder (expects .jsonl files).
    Returns a list of dicts with keys: task, reference, hypothesis, label.
    """
    data = []
    for fname in os.listdir(dev_folder):
        if fname.endswith(".jsonl"):
            for ex in read_jsonl(os.path.join(dev_folder, fname)):
                # dataset format: src, tgt, hyp, label, task
                ref = ex.get("src") or ex.get("tgt") or ""
                hyp = ex.get("hyp") or ""
                label = int(ex.get("label", -1))  # -1 if unlabeled
                task = ex.get("task", "(unknown)")
                data.append({
                    "task": task,
                    "reference": ref,
                    "hypothesis": hyp,
                    "label": label
                })
    return data
