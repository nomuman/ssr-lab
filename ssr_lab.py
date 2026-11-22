#!/usr/bin/env python3
"""ssr_lab.py — SSR scorer with the best configuration found in this lab.

Config (round 4 verdict):
  model     : sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  anchors   : anchors_v2.json (6 statements per Likert point)
  input     : whole free-text response (default)
  ensemble  : with --extracts, averages whole-text + extracted-intent +
              sentence-weighted scores (best Spearman in round 4)
  T         : 10  (p(k) proportional to sim^T)

Usage:
  uv run --with fastembed --with numpy python ssr_lab.py score data/techo-ai-letter.json
  uv run --with fastembed --with numpy python ssr_lab.py score data/techo-ai-letter.json --extracts data/techo-ai-letter.extracts.json
  uv run --with fastembed --with numpy python ssr_lab.py evaluate   # all concepts in data/
"""
import argparse
import json
import re
import sys

import numpy as np
from fastembed import TextEmbedding

MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TEMPERATURE = 10

# Optional linear calibration (round 5): SSR scores compress toward the
# middle, so a stretch fitted on labelled data helps WITHIN that domain
# (LOO-validated on the techo concepts: MAE 0.71->0.62 whole, 0.65->0.51
# ensemble). Round 6 showed calibration does NOT transfer across domains —
# refit per domain, never import coefficients from another concept group.
CALIBRATION = {"whole": (1.51, -1.92), "ensemble": (1.62, -2.36)}
USE_CALIBRATION = False

def calibrate(score, mode):
    if not USE_CALIBRATION:
        return score
    a, b = CALIBRATION[mode]
    return float(np.clip(a * score + b, 1, 5))

_model = None
_cache = {}

def embed(texts):
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL)
    todo = [t for t in texts if t not in _cache]
    if todo:
        for t, v in zip(todo, _model.embed(todo)):
            e = np.array(v, dtype=np.float64)
            _cache[t] = e / np.linalg.norm(e)
    return np.array([_cache[t] for t in texts])

def load_anchors(path="anchors_v2.json"):
    raw = json.load(open(path, encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}

def pmf(text, anchors):
    a_texts, a_levels = [], []
    for lvl in range(1, 6):
        for t in anchors[lvl]:
            a_texts.append(t); a_levels.append(lvl)
    A = embed(a_texts)
    x = embed([text])[0]
    sims = A @ x
    ls = [float(np.mean(sims[[j for j, l in enumerate(a_levels) if l == lvl]])) for lvl in range(1, 6)]
    w = np.power(np.clip(np.array(ls), 0.01, None), TEMPERATURE)
    return w / w.sum()

def expected(p):
    return float(np.dot(p, np.arange(1, 6)))

def split_sentences(text):
    return [s.strip() for s in re.split(r"[。！？]", text) if len(s.strip()) >= 4]

def score_response(text, anchors, extract_text=None):
    whole = expected(pmf(text, anchors))
    if extract_text is None:
        return calibrate(whole, "whole")
    ext = expected(pmf(extract_text, anchors))
    sents = split_sentences(text) or [text]
    exps, confs = [], []
    for s in sents:
        p = pmf(s, anchors)
        exps.append(expected(p)); confs.append(float(np.max(p)))
    wmean = float(np.average(exps, weights=confs))
    return calibrate(float(np.mean([whole, ext, wmean])), "ensemble")

def load_responses(path):
    return json.load(open(path, encoding="utf-8"))

def cmd_score(args):
    anchors = load_anchors()
    data = load_responses(args.file)
    extracts = None
    if args.extracts:
        extracts = json.load(open(args.extracts, encoding="utf-8"))
    personas = {p["id"]: p for p in data["personas"]}
    rows = []
    for r in data["responses"]:
        ext = None
        if extracts:
            ext = extracts.get(f"{r['persona_id']}@{r['price']}")
        s = score_response(r["text"], anchors, ext)
        rows.append((r, personas[r["persona_id"]]["segment"], s))
    print(f"Concept: {data['concept']}")
    for price in data["prices"]:
        vals = [s for r, _, s in rows if r["price"] == price]
        print(f"  @{price}: mean {np.mean(vals):.2f}  (n={len(vals)})")
    for seg in sorted({sg for _, sg, _ in rows}):
        line = []
        for price in data["prices"]:
            vals = [s for r, sg, s in rows if r["price"] == price and sg == seg]
            if vals:
                line.append(f"@{price} {np.mean(vals):.2f}")
        print(f"    {seg}: {' / '.join(line)}")

def concept_files():
    import glob
    return sorted(f for f in glob.glob("data/*.json")
                  if ".gold." not in f and ".extracts." not in f)

def cmd_evaluate(args):
    anchors = load_anchors()
    for mode in ["whole", "ensemble"]:
        errs, inv, pairs = [], 0, 0
        for fname in concept_files():
            gold = json.load(open(fname.replace(".json", ".gold.json"), encoding="utf-8"))
            extracts = json.load(open(fname.replace(".json", ".extracts.json"), encoding="utf-8"))
            data = load_responses(fname)
            got = {}
            for r in data["responses"]:
                key = f"{r['persona_id']}@{r['price']}"
                ext = extracts[key] if mode == "ensemble" else None
                s = score_response(r["text"], anchors, ext)
                got[r["persona_id"], r["price"]] = s
                errs.append(abs(s - gold[key]))
            lo, hi = data["prices"]
            for pid in {p["id"] for p in data["personas"]}:
                pairs += 1
                if got[pid, hi] > got[pid, lo]:
                    inv += 1
        print(f"{mode}: MAE {np.mean(errs):.3f}  inversions {inv}/{pairs}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("score")
    sp.add_argument("file")
    sp.add_argument("--extracts", default=None)
    sp.set_defaults(fn=cmd_score)
    ep = sub.add_parser("evaluate")
    ep.set_defaults(fn=cmd_evaluate)
    ap.add_argument("--calibrate", action="store_true",
                    help="apply the round-5 linear calibration fitted on gold.json")
    args = ap.parse_args()
    globals()["USE_CALIBRATION"] = args.calibrate
    args.fn(args)

if __name__ == "__main__":
    main()
