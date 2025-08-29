import re
STOP = set("a an the is am are was were be been being to of in on at by for with from as and or if than then this that these those it its their his her you your me my we our they them he she i".split())

def simple_tokens(text: str):
    text = text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    return [w for w in words if w not in STOP and len(w) > 1]
