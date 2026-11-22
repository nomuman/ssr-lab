#!/usr/bin/env python3
"""Round 9: direct Likert elicitation vs SSR, same personas/concepts/gold.

Direct ratings were generated WITHOUT seeing the free-text responses or
gold labels (persona + concept + price only, forced to one number).
SSR pmfs are recomputed here for the paper-style distribution comparison:
KS similarity between the gold label distribution and (a) the empirical
direct-rating distribution, (b) the pmf-averaged SSR distribution.
"""
import glob
import json
import re

import numpy as np
from fastembed import TextEmbedding

anchors = {int(k): v for k, v in json.load(open("anchors_v2.json", encoding="utf-8")).items()}
direct = json.load(open("validation/direct_ratings.json", encoding="utf-8"))

items = []
for fname in sorted(glob.glob("data/*.json")):
    if ".gold." in fname or ".extracts." in fname:
        continue
    slug = fname.split("/")[-1].replace(".json", "")
    data = json.load(open(fname, encoding="utf-8"))
    gold = json.load(open(fname.replace(".json", ".gold.json"), encoding="utf-8"))
    extracts = json.load(open(fname.replace(".json", ".extracts.json"), encoding="utf-8"))
    for r in data["responses"]:
        key = f"{r['persona_id']}@{r['price']}"
        items.append({"slug": slug, "key": key, "text": r["text"], "extract": extracts[key],
                      "gold": gold[key], "direct": direct[slug][key]})
print(f"{len(items)} items")

model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
_cache = {}
def embed(texts):
    todo = [t for t in texts if t not in _cache]
    for i in range(0, len(todo), 32):
        chunk = todo[i:i + 32]
        for t, v in zip(chunk, model.embed(chunk)):
            e = np.array(v, dtype=np.float64)
            _cache[t] = e / np.linalg.norm(e)
    return np.array([_cache[t] for t in texts])

def pmf(text):
    a_texts, a_levels = [], []
    for lvl in range(1, 6):
        for t in anchors[lvl]:
            a_texts.append(t); a_levels.append(lvl)
    A = embed(a_texts)
    x = embed([text])[0]
    sims = A @ x
    ls = [float(np.mean(sims[[j for j, l in enumerate(a_levels) if l == lvl]])) for lvl in range(1, 6)]
    w = np.power(np.clip(np.array(ls), 0.01, None), 10)
    return w / w.sum()

def exp_of(p):
    return float(np.dot(p, np.arange(1, 6)))

def split_sents(text):
    return [s.strip() for s in re.split(r"[。！？]", text) if len(s.strip()) >= 4]

for it in items:
    pw = pmf(it["text"])
    whole = exp_of(pw)
    ext = exp_of(pmf(it["extract"]))
    sents = split_sents(it["text"]) or [it["text"]]
    exps, confs = [], []
    for s in sents:
        p = pmf(s)
        exps.append(exp_of(p)); confs.append(float(np.max(p)))
    it["pmf"] = pw
    it["whole"] = whole
    it["ensemble"] = float(np.mean([whole, ext, float(np.average(exps, weights=confs))]))

golds = np.array([it["gold"] for it in items], float)
directs = np.array([it["direct"] for it in items], float)
wholes = np.array([it["whole"] for it in items])
ens = np.array([it["ensemble"] for it in items])

def dist_of(values):
    return np.array([np.mean(np.array(values) == k) for k in range(1, 6)])

def ks_sim(dist_a, dist_b):
    return round(1 - float(np.max(np.abs(np.cumsum(dist_a) - np.cumsum(dist_b)))), 3)

gold_dist = dist_of(golds)
direct_dist = dist_of(directs)
ssr_dist = np.mean([it["pmf"] for it in items], axis=0)

res = {
    "n": len(items),
    "direct": {"mae": round(float(np.mean(np.abs(directs - golds))), 3),
               "ks_similarity": ks_sim(gold_dist, direct_dist),
               "share_3": round(float(np.mean(directs == 3)), 3),
               "std": round(float(np.std(directs)), 3)},
    "gold": {"share_3": round(float(np.mean(golds == 3)), 3), "std": round(float(np.std(golds)), 3)},
    "ssr_whole": {"mae": round(float(np.mean(np.abs(wholes - golds))), 3)},
    "ssr_ensemble": {"mae": round(float(np.mean(np.abs(ens - golds))), 3)},
    "ssr_pmf_distribution_ks": ks_sim(gold_dist, ssr_dist),
    "distributions": {"gold": [round(float(x), 3) for x in gold_dist],
                      "direct": [round(float(x), 3) for x in direct_dist],
                      "ssr_pmf": [round(float(x), 3) for x in ssr_dist]},
}
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open("validation/direct_vs_ssr_results.json", "w"), ensure_ascii=False, indent=1)
