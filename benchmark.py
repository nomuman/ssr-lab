#!/usr/bin/env python3
"""benchmark.py — run the SSR method comparison across every concept in data/.

Methods (all: paraphrase-multilingual-mpnet-base-v2 + anchors_v2, T=10):
  whole     : score the full free-text response
  extract   : score the LLM-extracted one-sentence intent
  ensemble  : mean of whole + extract + sentence-weighted

Also tests calibration transfer: fit the linear stretch on one group of
concepts, apply it to the other group, and see whether MAE still improves.

  uv run --with fastembed --with numpy python benchmark.py
Writes benchmark_results.json and prints a summary table.
"""
import glob
import json
import re

import numpy as np
from fastembed import TextEmbedding

MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
T = 10

anchors_raw = json.load(open("anchors_v2.json", encoding="utf-8"))
ANCHORS = {int(k): v for k, v in anchors_raw.items()}

_model = TextEmbedding(model_name=MODEL)
_cache = {}

def embed(texts):
    todo = [t for t in texts if t not in _cache]
    if todo:
        for t, v in zip(todo, _model.embed(todo)):
            e = np.array(v, dtype=np.float64)
            _cache[t] = e / np.linalg.norm(e)
    return np.array([_cache[t] for t in texts])

def pmf(text):
    a_texts, a_levels = [], []
    for lvl in range(1, 6):
        for t in ANCHORS[lvl]:
            a_texts.append(t); a_levels.append(lvl)
    A = embed(a_texts)
    x = embed([text])[0]
    sims = A @ x
    ls = [float(np.mean(sims[[j for j, l in enumerate(a_levels) if l == lvl]])) for lvl in range(1, 6)]
    w = np.power(np.clip(np.array(ls), 0.01, None), T)
    return w / w.sum()

def expected(p):
    return float(np.dot(p, np.arange(1, 6)))

def split_sentences(text):
    return [s.strip() for s in re.split(r"[。！？]", text) if len(s.strip()) >= 4]

def score_whole(text):
    return expected(pmf(text))

def score_extract(text):
    return expected(pmf(text))

def score_ensemble(text, extract_text):
    whole = score_whole(text)
    ext = score_extract(extract_text)
    sents = split_sentences(text) or [text]
    exps, confs = [], []
    for s in sents:
        p = pmf(s)
        exps.append(expected(p)); confs.append(float(np.max(p)))
    return float(np.mean([whole, ext, float(np.average(exps, weights=confs))]))

def ranks(a):
    a = np.asarray(a, float)
    order = np.argsort(a)
    r = np.empty(len(a)); r[order] = np.arange(len(a))
    return r

def metrics(items, scores):
    golds = np.array([it["gold"] for it in items], float)
    s = np.array(scores, float)
    mae = float(np.mean(np.abs(s - golds)))
    sp = float(np.corrcoef(ranks(s), ranks(golds))[0, 1])
    by = {(it["concept"], it["pid"], it["price"]): v for it, v in zip(items, scores)}
    inv = pairs = 0
    for c in {it["concept"] for it in items}:
        citems = [it for it in items if it["concept"] == c]
        lo, hi = citems[0]["lo"], citems[0]["hi"]
        for pid in {it["pid"] for it in citems}:
            pairs += 1
            if by[(c, pid, hi)] > by[(c, pid, lo)]:
                inv += 1
    return {"mae": round(mae, 3), "spearman": round(sp, 3), "inversions": inv, "pairs": pairs}

def loo_cal_mae(items, scores):
    golds = np.array([it["gold"] for it in items], float)
    s = np.array(scores, float)
    preds = np.zeros_like(s)
    for i in range(len(s)):
        m = np.ones(len(s), bool); m[i] = False
        a, b = np.polyfit(s[m], golds[m], 1)
        preds[i] = np.clip(a * s[i] + b, 1, 5)
    return round(float(np.mean(np.abs(preds - golds))), 3)

def transfer_cal_mae(fit_items, fit_scores, test_items, test_scores):
    fg = np.array([it["gold"] for it in fit_items], float)
    fs = np.array(fit_scores, float)
    a, b = np.polyfit(fs, fg, 1)
    tg = np.array([it["gold"] for it in test_items], float)
    ts = np.clip(a * np.array(test_scores, float) + b, 1, 5)
    return round(float(np.mean(np.abs(ts - tg))), 3)

def main():
    items = []
    for fname in sorted(glob.glob("data/*.json")):
        if ".gold." in fname or ".extracts." in fname:
            continue
        slug = fname.split("/")[-1].replace(".json", "")
        data = json.load(open(fname, encoding="utf-8"))
        gold = json.load(open(fname.replace(".json", ".gold.json"), encoding="utf-8"))
        extracts = json.load(open(fname.replace(".json", ".extracts.json"), encoding="utf-8"))
        lo, hi = data["prices"]
        for r in data["responses"]:
            key = f"{r['persona_id']}@{r['price']}"
            items.append({"concept": slug, "pid": r["persona_id"], "price": r["price"],
                          "lo": lo, "hi": hi, "text": r["text"],
                          "gold": gold[key], "extract": extracts[key]})
    print(f"{len(items)} items across {len({it['concept'] for it in items})} concepts")

    scores = {
        "mpnet_v2_whole": [score_whole(it["text"]) for it in items],
        "mpnet_v2_extract": [score_extract(it["extract"]) for it in items],
        "mpnet_ensemble": [score_ensemble(it["text"], it["extract"]) for it in items],
    }

    out = {"n_items": len(items), "methods": {}, "per_concept": {}, "calibration": {}}
    for name, sc in scores.items():
        out["methods"][name] = metrics(items, sc)
        out["methods"][name]["loo_cal_mae"] = loo_cal_mae(items, sc)
        print(f"{name}: {out['methods'][name]}")
    for c in sorted({it["concept"] for it in items}):
        idx = [i for i, it in enumerate(items) if it["concept"] == c]
        sub = [items[i] for i in idx]
        out["per_concept"][c] = {name: metrics(sub, [sc[i] for i in idx]) for name, sc in scores.items()}
        print(f"  {c}: whole MAE {out['per_concept'][c]['mpnet_v2_whole']['mae']}")

    techo = [i for i, it in enumerate(items) if it["concept"].startswith("techo-")]
    other = [i for i, it in enumerate(items) if not it["concept"].startswith("techo-")]
    if techo and other:
        for name, sc in scores.items():
            t2o = transfer_cal_mae([items[i] for i in techo], [sc[i] for i in techo],
                                   [items[i] for i in other], [sc[i] for i in other])
            o2t = transfer_cal_mae([items[i] for i in other], [sc[i] for i in other],
                                   [items[i] for i in techo], [sc[i] for i in techo])
            out["calibration"][name] = {"fit_techo_test_others": t2o, "fit_others_test_techo": o2t}
        print("calibration transfer:", json.dumps(out["calibration"], ensure_ascii=False))

    json.dump({"items": [{k: it[k] for k in ("concept", "pid", "price", "gold")} for it in items],
               "scores": scores, **out}, open("benchmark_results.json", "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
