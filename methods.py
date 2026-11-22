#!/usr/bin/env python3
"""SSR method comparison harness.
Methods vary: embedding model x anchors x input granularity (whole/sentence/extract) x temperature.
Evaluation vs gold labels: MAE, Spearman, price-inversion count, flagged-case MAE.
"""
import json
import re
import numpy as np
from fastembed import TextEmbedding

MINI = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

gold = json.load(open("gold.json", encoding="utf-8"))
extracts = json.load(open("extracts.json", encoding="utf-8"))
anchors_v1 = {int(k): v for k, v in json.load(open("anchors_v1.json", encoding="utf-8")).items()}
anchors_v2 = {int(k): v for k, v in json.load(open("anchors_v2.json", encoding="utf-8")).items()}

ITEMS = []  # (file, pid, price, text, gold, extract)
for fname in ["responses.json", "responses2.json"]:
    d = json.load(open(fname, encoding="utf-8"))
    for r in d["responses"]:
        key = f"{r['persona_id']}@{r['price']}"
        ITEMS.append({
            "file": fname, "pid": r["persona_id"], "price": r["price"],
            "text": r["text"], "gold": gold[fname][key],
            "extract": extracts[fname][key],
        })

_caches = {}
def embedder(model_name):
    if model_name not in _caches:
        model = TextEmbedding(model_name=model_name)
        cache = {}
        def embed(texts):
            todo = [t for t in texts if t not in cache]
            if todo:
                vecs = list(model.embed(todo))
                for t, v in zip(todo, vecs):
                    e = np.array(v, dtype=np.float64)
                    cache[t] = e / np.linalg.norm(e)
            return np.array([cache[t] for t in texts])
        _caches[model_name] = embed
    return _caches[model_name]

def pmf_for(texts, anchors, embed, T=10):
    a_texts, a_levels = [], []
    for lvl in range(1, 6):
        for t in anchors[lvl]:
            a_texts.append(t); a_levels.append(lvl)
    A = embed(a_texts)
    X = embed(texts)
    sims = X @ A.T
    out = []
    for i in range(len(texts)):
        ls = [float(np.mean(sims[i, [j for j, l in enumerate(a_levels) if l == lvl]])) for lvl in range(1, 6)]
        w = np.power(np.clip(np.array(ls), 0.01, None), T)
        out.append(w / w.sum())
    return out

def expected(pmf):
    return float(np.dot(pmf, np.arange(1, 6)))

def split_sentences(text):
    return [s.strip() for s in re.split(r"[。！？]", text) if len(s.strip()) >= 4]

def spearman(x, y):
    def ranks(a):
        order = np.argsort(a); r = np.empty(len(a), dtype=float)
        r[order] = np.arange(len(a))
        return r
    rx, ry = ranks(np.array(x)), ranks(np.array(y))
    return float(np.corrcoef(rx, ry)[0, 1])

FLAGGED = {("responses.json", 10, 300), ("responses.json", 3, 300), ("responses.json", 1, 800),
           ("responses2.json", 10, 800), ("responses2.json", 2, 800), ("responses2.json", 12, 800),
           ("responses2.json", 11, 300), ("responses2.json", 6, 300)}

def evaluate(name, scores):
    errs = [abs(s - it["gold"]) for s, it in zip(scores, ITEMS)]
    inv = 0
    by_key = {(it["file"], it["pid"], it["price"]): s for s, it in zip(scores, ITEMS)}
    for fname in ["responses.json", "responses2.json"]:
        for pid in range(1, 13):
            if by_key[(fname, pid, 800)] > by_key[(fname, pid, 300)]:
                inv += 1
    ferr = [abs(by_key[(f, p, pr)] - gold[f][f"{p}@{pr}"]) for (f, p, pr) in FLAGGED]
    return {"method": name, "MAE": round(float(np.mean(errs)), 3),
            "spearman": round(spearman(scores, [it["gold"] for it in ITEMS]), 3),
            "inversions": inv, "flagged_MAE": round(float(np.mean(ferr)), 3),
            "scores": [round(float(s), 2) for s in scores]}

def run():
    results = []
    texts = [it["text"] for it in ITEMS]
    exts = [it["extract"] for it in ITEMS]

    for label, model_name in [("mini", MINI), ("mpnet", MPNET)]:
        embed = embedder(model_name)
        for aname, anchors in [("v1", anchors_v1), ("v2", anchors_v2)]:
            pmfs = pmf_for(texts, anchors, embed)
            results.append(evaluate(f"{label}_{aname}_whole", [expected(p) for p in pmfs]))
            pmfs_e = pmf_for(exts, anchors, embed)
            results.append(evaluate(f"{label}_{aname}_extract", [expected(p) for p in pmfs_e]))
        # sentence-level with v2 anchors
        anchors = anchors_v2
        conf_scores, wmean_scores = [], []
        for it in ITEMS:
            sents = split_sentences(it["text"]) or [it["text"]]
            pmfs_s = pmf_for(sents, anchors, embed)
            exps = [expected(p) for p in pmfs_s]
            confs = [float(np.max(p)) for p in pmfs_s]
            conf_scores.append(exps[int(np.argmax(confs))])
            wmean_scores.append(float(np.average(exps, weights=confs)))
        results.append(evaluate(f"{label}_v2_sentconf", conf_scores))
        results.append(evaluate(f"{label}_v2_sentwmean", wmean_scores))

    # temperature variants on mini v2 whole
    embed = embedder(MINI)
    for T in [5, 20]:
        pmfs = pmf_for(texts, anchors_v2, embed, T=T)
        results.append(evaluate(f"mini_v2_whole_T{T}", [expected(p) for p in pmfs]))

    results.sort(key=lambda r: r["MAE"])
    for r in results:
        print(f"{r['method']:24s} MAE {r['MAE']:.3f}  Spearman {r['spearman']:.3f}  inversions {r['inversions']:2d}/24  flaggedMAE {r['flagged_MAE']:.3f}")
    json.dump({"items": [{k: it[k] for k in ("file", "pid", "price", "gold")} for it in ITEMS],
               "results": results}, open("methods_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    run()
