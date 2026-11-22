#!/usr/bin/env python3
"""Round 5b: try intfloat/multilingual-e5-large (with 'passage: ' prefixes)."""
import json
import numpy as np
from fastembed import TextEmbedding

gold = json.load(open("gold.json", encoding="utf-8"))
extracts = json.load(open("extracts.json", encoding="utf-8"))
anchors = {int(k): v for k, v in json.load(open("anchors_v2.json", encoding="utf-8")).items()}

ITEMS = []
for fname in ["responses.json", "responses2.json"]:
    d = json.load(open(fname, encoding="utf-8"))
    for r in d["responses"]:
        key = f"{r['persona_id']}@{r['price']}"
        ITEMS.append({"file": fname, "pid": r["persona_id"], "price": r["price"],
                      "text": r["text"], "gold": gold[fname][key], "extract": extracts[fname][key]})

model = TextEmbedding(model_name="intfloat/multilingual-e5-large")
_cache = {}
def embed(texts):
    todo = [t for t in texts if t not in _cache]
    if todo:
        for t, v in zip(todo, model.embed(["passage: " + t for t in todo])):
            e = np.array(v, dtype=np.float64)
            _cache[t] = e / np.linalg.norm(e)
    return np.array([_cache[t] for t in texts])

def score(text):
    a_texts, a_levels = [], []
    for lvl in range(1, 6):
        for t in anchors[lvl]:
            a_texts.append(t); a_levels.append(lvl)
    A = embed(a_texts); x = embed([text])[0]
    sims = A @ x
    ls = [float(np.mean(sims[[j for j, l in enumerate(a_levels) if l == lvl]])) for lvl in range(1, 6)]
    w = np.power(np.clip(np.array(ls), 0.01, None), 10)
    p = w / w.sum()
    return float(np.dot(p, np.arange(1, 6)))

def ranks(a):
    o = np.argsort(a); r = np.empty(len(a)); r[o] = np.arange(len(a)); return r

for mode, get_text in [("whole", lambda it: it["text"]), ("extract", lambda it: it["extract"])]:
    scores = [score(get_text(it)) for it in ITEMS]
    golds = np.array([it["gold"] for it in ITEMS], float)
    mae = float(np.mean(np.abs(np.array(scores) - golds)))
    sp = float(np.corrcoef(ranks(scores), ranks(golds))[0, 1])
    by = {(it["file"], it["pid"], it["price"]): s for s, it in zip(scores, ITEMS)}
    inv = sum(1 for f in ["responses.json", "responses2.json"] for pid in range(1, 13) if by[(f, pid, 800)] > by[(f, pid, 300)])
    print(f"e5-large_v2_{mode}: MAE {mae:.3f} Spearman {sp:.3f} inversions {inv}/24")
    json.dump(scores, open(f"e5_{mode}_scores.json", "w"))
