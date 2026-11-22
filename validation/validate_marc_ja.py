#!/usr/bin/env python3
"""Validate the SSR mapping on REAL human text: MARC-ja (Multilingual Amazon
Reviews Corpus, Japanese test split via the SetFit mirror on Hugging Face).

Each review has a true 1-5 star rating. We score the review text with SSR
(satisfaction anchors, MPNet, T=10) and compare:
  - per-item MAE / Spearman of expected score vs. stars
  - distribution-level KS similarity (paper-style): CDF distance between the
    true star distribution and the pmf-averaged predicted distribution
  - a single-word-anchor baseline for contrast

Note: this validates text->scale mapping on human writing. The construct is
post-purchase satisfaction (stars), not pre-purchase intent.

Data: curl -sL -o validation/marc_ja_test.jsonl \
  https://huggingface.co/datasets/SetFit/amazon_reviews_multi_ja/resolve/main/test.jsonl
(Raw reviews are NOT committed to this repo; only ids + metrics are.)
"""
import json
import random

import numpy as np
from fastembed import TextEmbedding

SEED = 42
PER_STAR = 200
MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
T = 10

rows = [json.loads(l) for l in open("validation/marc_ja_test.jsonl", encoding="utf-8")]
rng = random.Random(SEED)
sample = []
for star in range(5):
    pool = [r for r in rows if r["label"] == star]
    sample += rng.sample(pool, PER_STAR)
print(f"sample: {len(sample)} reviews (balanced {PER_STAR}/star)")

model = TextEmbedding(model_name=MODEL)
_cache = {}
BATCH = 32
MAXCHARS = 1000  # reviews run to 2000 chars; SSR targets short texts, cap for memory
def embed(texts):
    texts = [t[:MAXCHARS] for t in texts]
    todo = [t for t in texts if t not in _cache]
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        for t, v in zip(chunk, model.embed(chunk)):
            e = np.array(v, dtype=np.float64)
            _cache[t] = e / np.linalg.norm(e)
    return np.array([_cache[t] for t in texts])

def run(anchors, label):
    a_texts, a_levels = [], []
    for lvl in range(1, 6):
        for t in anchors[lvl]:
            a_texts.append(t); a_levels.append(lvl)
    A = embed(a_texts)
    X = embed([r["text"] for r in sample])
    sims = X @ A.T
    pmfs = []
    for i in range(len(sample)):
        ls = [float(np.mean(sims[i][[j for j, l in enumerate(a_levels) if l == lvl]])) for lvl in range(1, 6)]
        w = np.power(np.clip(np.array(ls), 0.01, None), T)
        pmfs.append(w / w.sum())
    pmfs = np.array(pmfs)
    exp = pmfs @ np.arange(1, 6)
    stars = np.array([r["label"] + 1 for r in sample], float)
    mae = float(np.mean(np.abs(exp - stars)))
    order = np.argsort(exp); rk = np.empty(len(exp)); rk[order] = np.arange(len(exp))
    order2 = np.argsort(stars); rk2 = np.empty(len(stars)); rk2[order2] = np.arange(len(stars))
    spear = float(np.corrcoef(rk, rk2)[0, 1])
    true_dist = np.array([np.mean(stars == k) for k in range(1, 6)])
    pred_dist = pmfs.mean(axis=0)
    ks = float(np.max(np.abs(np.cumsum(true_dist) - np.cumsum(pred_dist))))
    per_star = {int(k): round(float(np.mean(np.abs(exp[stars == k] - k))), 3) for k in range(1, 6)}
    res = {"mae": round(mae, 3), "spearman": round(spear, 3), "ks_similarity": round(1 - ks, 3),
           "per_star_mae": per_star, "pred_distribution": [round(float(x), 3) for x in pred_dist]}
    print(label, json.dumps(res, ensure_ascii=False))
    return res

sat = {int(k): v for k, v in json.load(open("validation/anchors_satisfaction.json", encoding="utf-8")).items()}
words = {1: ["最低"], 2: ["いまいち"], 3: ["普通"], 4: ["良い"], 5: ["最高"]}

out = {"dataset": "MARC-ja test split (SetFit/amazon_reviews_multi_ja), 1000 reviews balanced by star",
       "satisfaction_anchors": run(sat, "satisfaction anchors (6/level):"),
       "single_word_anchors": run(words, "single-word anchors:"),
       "sample_ids": [r["id"] for r in sample]}
json.dump(out, open("validation/marc_ja_results.json", "w"), ensure_ascii=False, indent=1)
print("saved validation/marc_ja_results.json")
