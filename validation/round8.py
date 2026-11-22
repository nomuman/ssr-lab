#!/usr/bin/env python3
"""Round 8: (a) improve SSR on real reviews, (b) check generality.

Datasets (real human text, true 1-5 stars):
  marc_ja : MARC-ja test, same 1,000-review sample as round 7
  marc_en : MARC-en test (SetFit mirror), 1,000 reviews balanced by star
  yelp    : Yelp Review Full test, 1,000 reviews balanced by star

Methods (MPNet unless noted):
  whole            : whole-text SSR
  sentwmean        : sentence-level SSR, confidence-weighted mean
  anchors3         : whole-text with only 3 anchors per level (ablation)
  whole_cal        : whole + split-half linear calibration (fit A->test B, B->A)
  sentwmean_cal    : sentwmean + split-half calibration
  e5_whole(_cal)   : multilingual-e5-large, marc_ja only

Metrics: MAE, Spearman, KS similarity (pmf distribution vs true distribution).
"""
import json
import random
import re

import numpy as np
from fastembed import TextEmbedding

SEED = 42
PER_STAR = 200
T = 10
BATCH = 32
MAXCHARS = 1000

# ---------- data ----------
def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def balanced(rows, per=PER_STAR):
    rng = random.Random(SEED)
    out = []
    for star in range(5):
        pool = [r for r in rows if r["label"] == star]
        out += rng.sample(pool, min(per, len(pool)))
    return out

ja_rows = load_jsonl("validation/marc_ja_test.jsonl")
prev = json.load(open("validation/marc_ja_results.json"))
by_id = {r["id"]: r for r in ja_rows}
ja_sample = [by_id[i] for i in prev["sample_ids"]]
en_sample = balanced(load_jsonl("validation/marc_en_test.jsonl"))

import pyarrow.parquet as pq
yelp_table = pq.read_table("validation/yelp_test.parquet")
yelp_rows = [{"text": t, "label": l} for t, l in zip(yelp_table.column("text").to_pylist(),
                                                      yelp_table.column("label").to_pylist())]
yelp_sample = balanced(yelp_rows)

# ---------- embedding infra ----------
class Embedder:
    def __init__(self, model_name, prefix=""):
        self.model = TextEmbedding(model_name=model_name)
        self.prefix = prefix
        self.cache = {}
    def embed(self, texts):
        texts = [t[:MAXCHARS] for t in texts]
        todo = [t for t in texts if t not in self.cache]
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            for t, v in zip(chunk, self.model.embed([self.prefix + c for c in chunk])):
                e = np.array(v, dtype=np.float64)
                self.cache[t] = e / np.linalg.norm(e)
        return np.array([self.cache[t] for t in texts])

mpnet = Embedder("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

def pmfs_for(texts, anchors, emb):
    a_texts, a_levels = [], []
    for lvl in range(1, 6):
        for t in anchors[lvl]:
            a_texts.append(t); a_levels.append(lvl)
    A = emb.embed(a_texts)
    X = emb.embed(texts)
    sims = X @ A.T
    out = []
    for i in range(len(texts)):
        ls = [float(np.mean(sims[i][[j for j, l in enumerate(a_levels) if l == lvl]])) for lvl in range(1, 6)]
        w = np.power(np.clip(np.array(ls), 0.01, None), T)
        out.append(w / w.sum())
    return np.array(out)

def split_sents(text, lang):
    pat = r"[。！？]" if lang == "ja" else r"[.!?]"
    return [s.strip() for s in re.split(pat, text) if len(s.strip()) >= 4][:40]

def ranks(a):
    a = np.asarray(a, float)
    o = np.argsort(a); r = np.empty(len(a)); r[o] = np.arange(len(a)); return r

def metrics(exp, pmf_mean, stars):
    mae = float(np.mean(np.abs(exp - stars)))
    sp = float(np.corrcoef(ranks(exp), ranks(stars))[0, 1])
    true_dist = np.array([np.mean(stars == k) for k in range(1, 6)])
    ks = float(np.max(np.abs(np.cumsum(true_dist) - np.cumsum(pmf_mean))))
    return {"mae": round(mae, 3), "spearman": round(sp, 3), "ks_similarity": round(1 - ks, 3)}

def split_half_cal_mae(exp, stars):
    idx = np.arange(len(exp))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    half = len(idx) // 2
    maes = []
    for fit_i, test_i in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
        a, b = np.polyfit(exp[fit_i], stars[fit_i], 1)
        pred = np.clip(a * exp[test_i] + b, 1, 5)
        maes.append(float(np.mean(np.abs(pred - stars[test_i]))))
    return round(float(np.mean(maes)), 3)

def evaluate(sample, anchors, lang, emb, with_e5=False):
    stars = np.array([r["label"] + 1 for r in sample], float)
    texts = [r["text"] for r in sample]
    res = {}
    pmf = pmfs_for(texts, anchors, emb)
    exp = pmf @ np.arange(1, 6)
    res["whole"] = metrics(exp, pmf.mean(axis=0), stars)
    res["whole_cal"] = {"mae": split_half_cal_mae(exp, stars)}
    a3 = {k: v[:3] for k, v in anchors.items()}
    pmf3 = pmfs_for(texts, a3, emb)
    res["anchors3"] = metrics(pmf3 @ np.arange(1, 6), pmf3.mean(axis=0), stars)
    # sentence-weighted
    all_sents, spans = [], []
    for t in texts:
        ss = split_sents(t, lang) or [t]
        spans.append((len(all_sents), len(all_sents) + len(ss)))
        all_sents += ss
    spmf = pmfs_for(all_sents, anchors, emb)
    sexp = spmf @ np.arange(1, 6)
    sconf = spmf.max(axis=1)
    wexp, wpmf = [], []
    for a, b in spans:
        w = sconf[a:b]
        wexp.append(float(np.average(sexp[a:b], weights=w)))
        wpmf.append(np.average(spmf[a:b], axis=0, weights=w))
    wexp = np.array(wexp)
    res["sentwmean"] = metrics(wexp, np.array(wpmf).mean(axis=0), stars)
    res["sentwmean_cal"] = {"mae": split_half_cal_mae(wexp, stars)}
    if with_e5:
        e5 = Embedder("intfloat/multilingual-e5-large", prefix="passage: ")
        ep = pmfs_for(texts, anchors, e5)
        eexp = ep @ np.arange(1, 6)
        res["e5_whole"] = metrics(eexp, ep.mean(axis=0), stars)
        res["e5_whole_cal"] = {"mae": split_half_cal_mae(eexp, stars)}
    return res

ja_anchors = {int(k): v for k, v in json.load(open("validation/anchors_satisfaction.json", encoding="utf-8")).items()}
en_anchors = {int(k): v for k, v in json.load(open("validation/anchors_satisfaction_en.json", encoding="utf-8")).items()}

import os
out = {}
def save_partial():
    json.dump(out, open("validation/round8_results.json", "w"), ensure_ascii=False, indent=1)
print("== marc_ja =="); out["marc_ja"] = evaluate(ja_sample, ja_anchors, "ja", mpnet, with_e5=(os.environ.get("E5") == "1")); print(json.dumps(out["marc_ja"], ensure_ascii=False)); save_partial()
mpnet.cache.clear()
print("== marc_en =="); out["marc_en"] = evaluate(en_sample, en_anchors, "en", mpnet); print(json.dumps(out["marc_en"], ensure_ascii=False)); save_partial()
mpnet.cache.clear()
print("== yelp =="); out["yelp"] = evaluate(yelp_sample, en_anchors, "en", mpnet); print(json.dumps(out["yelp"], ensure_ascii=False))
json.dump(out, open("validation/round8_results.json", "w"), ensure_ascii=False, indent=1)
print("saved validation/round8_results.json")
