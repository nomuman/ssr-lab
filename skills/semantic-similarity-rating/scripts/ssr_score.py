#!/usr/bin/env python3
"""ssr_score.py — standalone Semantic Similarity Rating (SSR) scorer.

Scores free-text responses on a 5-point Likert scale by embedding similarity
to anchor statements. No API key needed: embeddings run locally via fastembed.

Run with uv (dependencies are declared inline via the command, not the file):
  uv run --with fastembed --with numpy python ssr_score.py responses.json
  uv run --with fastembed --with numpy python ssr_score.py responses.json \
      --anchors satisfaction_en --mode ensemble --out scores.json

Input formats:
  A) lab format: {"concept": str, "prices": [lo, hi],
                  "personas": [{"id": 1, "segment": str, "profile": str}],
                  "responses": [{"persona_id": 1, "price": 300,
                                 "text": str, "extract": optional str}]}
  B) simple list: [{"id": "r1", "text": str,
                    "extract": optional str, "group": optional str}]

Modes:
  whole    : score the full response text (no preparation needed)
  extract  : score the one-sentence intent extract (write extracts first;
             items without an extract fall back to whole-text)
  ensemble : mean of whole + extract + sentence-weighted (best balance of
             accuracy and price-inversion resistance in the ssr-lab benchmark)
"""
import argparse
import json
import os
import re
import sys

import numpy as np

PRESETS = {
    "purchase_ja": "anchors_purchase_intent_ja.json",
    "purchase_en": "anchors_purchase_intent_en.json",
    "satisfaction_ja": "anchors_satisfaction_ja.json",
    "satisfaction_en": "anchors_satisfaction_en.json",
}
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

_model = None
_cache = {}

def embed(texts, model_name):
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=model_name)
    todo = [t for t in texts if t not in _cache]
    for i in range(0, len(todo), 32):
        chunk = todo[i:i + 32]
        for t, v in zip(chunk, _model.embed(chunk)):
            e = np.array(v, dtype=np.float64)
            _cache[t] = e / np.linalg.norm(e)
    return np.array([_cache[t] for t in texts])

def load_anchors(spec):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "assets", PRESETS[spec]) if spec in PRESETS else spec
    raw = json.load(open(path, encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}

def make_pmf_fn(anchors, temperature, model_name):
    a_texts, a_levels = [], []
    for lvl in range(1, 6):
        for t in anchors[lvl]:
            a_texts.append(t); a_levels.append(lvl)
    A = embed(a_texts, model_name)
    def pmf(text):
        x = embed([text], model_name)[0]
        sims = A @ x
        ls = [float(np.mean(sims[[j for j, l in enumerate(a_levels) if l == lvl]])) for lvl in range(1, 6)]
        w = np.power(np.clip(np.array(ls), 0.01, None), temperature)
        return w / w.sum()
    return pmf

def expected(p):
    return float(np.dot(p, np.arange(1, 6)))

def split_sentences(text):
    return [s.strip() for s in re.split(r"[。！？.!?]", text) if len(s.strip()) >= 4]

def score_item(item, pmf, mode):
    whole_p = pmf(item["text"])
    whole = expected(whole_p)
    if mode == "whole":
        return whole, whole_p, "whole"
    ext_text = item.get("extract")
    if not ext_text:
        return whole, whole_p, "whole (no extract)"
    ext = expected(pmf(ext_text))
    if mode == "extract":
        return ext, pmf(ext_text), "extract"
    sents = split_sentences(item["text"]) or [item["text"]]
    exps, confs = [], []
    for s in sents:
        p = pmf(s)
        exps.append(expected(p)); confs.append(float(np.max(p)))
    wmean = float(np.average(exps, weights=confs))
    return float(np.mean([whole, ext, wmean])), whole_p, "ensemble"

def normalize(data):
    """Return (items, group_key_fn). Items: {key, text, extract?, group}."""
    if isinstance(data, list):
        return ([{"key": str(it.get("id", i)), "text": it["text"],
                  "extract": it.get("extract"), "group": it.get("group", "all")}
                 for i, it in enumerate(data)], True)
    segments = {p["id"]: p.get("segment", "") for p in data.get("personas", [])}
    items = []
    for r in data["responses"]:
        items.append({"key": f"{r['persona_id']}@{r['price']}", "text": r["text"],
                      "extract": r.get("extract"),
                      "group": f"price:{r['price']}",
                      "segment": segments.get(r["persona_id"], "")})
    return items, False

def main():
    ap = argparse.ArgumentParser(description="SSR scorer (see module docstring)")
    ap.add_argument("input")
    ap.add_argument("--anchors", default="purchase_ja",
                    help="preset (purchase_ja, purchase_en, satisfaction_ja, satisfaction_en) or path to anchors JSON")
    ap.add_argument("--mode", choices=["whole", "extract", "ensemble"], default="whole")
    ap.add_argument("--temperature", type=float, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    anchors = load_anchors(args.anchors)
    pmf = make_pmf_fn(anchors, args.temperature, args.model)
    data = json.load(open(args.input, encoding="utf-8"))
    items, simple = normalize(data)

    results = []
    for it in items:
        s, p, used = score_item(it, pmf, args.mode)
        results.append({"key": it["key"], "group": it["group"], "score": round(s, 3),
                        "pmf": [round(float(x), 3) for x in p], "mode_used": used,
                        **({"segment": it["segment"]} if "segment" in it else {})})

    groups = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r["score"])
    summary = {g: {"mean": round(float(np.mean(v)), 3), "n": len(v)} for g, v in sorted(groups.items())}
    out = {"summary": summary, "items": results}
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
    print(json.dumps(summary, ensure_ascii=False))
    print(f"scored {len(results)} items (mode={args.mode}, anchors={args.anchors})", file=sys.stderr)

if __name__ == "__main__":
    main()
