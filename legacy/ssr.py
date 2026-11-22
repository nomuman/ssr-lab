#!/usr/bin/env python3
"""SSR (Semantic Similarity Rating) mini trial.
LLM free-text responses -> embeddings -> cosine similarity to Likert anchor
statements -> probability distribution over the 5-point purchase-intent scale.
Method follows Maier et al. 2025 (arXiv:2510.08338), simplified.
"""
import json
import sys
import numpy as np
from fastembed import TextEmbedding

IN_FILE = sys.argv[1] if len(sys.argv) > 1 else "responses.json"
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "results.json"
ANCHOR_FILE = sys.argv[3] if len(sys.argv) > 3 else None
DATA = json.load(open(IN_FILE, encoding="utf-8"))

ANCHORS = {
    5: ["ぜひ使いたいです。お金を払ってでもすぐに始めたいです。",
        "これは自分にぴったりだと思います。必ず購入します。",
        "とても魅力的です。迷わず申し込みます。"],
    4: ["いいと思います。たぶん購入するでしょう。",
        "気に入りました。買う可能性は高いです。",
        "前向きに検討します。価格も妥当だと思います。"],
    3: ["どちらとも言えません。少し興味はありますが迷います。",
        "まあまあです。様子を見て考えます。",
        "内容次第です。まず試してみないと判断できません。"],
    2: ["あまり魅力を感じません。たぶん買いません。",
        "自分には合わない気がします。購入は見送ると思います。",
        "価格が高いと感じます。今のところは要りません。"],
    1: ["絶対に買いません。全く興味がありません。",
        "お金を払う価値はないと思います。",
        "論外です。利用するつもりは一切ありません。"],
}

if ANCHOR_FILE:
    _raw = json.load(open(ANCHOR_FILE, encoding="utf-8"))
    ANCHORS = {int(k): v for k, v in _raw.items()}

TEMPERATURE = 10  # paper: p(k) proportional to sim^T

def main():
    model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    anchor_texts, anchor_levels = [], []
    for lvl in range(1, 6):
        for t in ANCHORS[lvl]:
            anchor_texts.append(t)
            anchor_levels.append(lvl)
    anchor_emb = np.array(list(model.embed(anchor_texts)))
    anchor_emb /= np.linalg.norm(anchor_emb, axis=1, keepdims=True)

    resp_texts = [r["text"] for r in DATA["responses"]]
    resp_emb = np.array(list(model.embed(resp_texts)))
    resp_emb /= np.linalg.norm(resp_emb, axis=1, keepdims=True)

    sims = resp_emb @ anchor_emb.T  # (n_resp, n_anchor)

    personas = {p["id"]: p for p in DATA["personas"]}
    results = []
    for i, r in enumerate(DATA["responses"]):
        level_sim = []
        for lvl in range(1, 6):
            cols = [j for j, l in enumerate(anchor_levels) if l == lvl]
            level_sim.append(float(np.mean(sims[i, cols])))
        level_sim = np.array(level_sim)
        w = np.power(np.clip(level_sim, 0.01, None), TEMPERATURE)
        pmf = w / w.sum()
        expected = float(np.dot(pmf, np.arange(1, 6)))
        results.append({
            "persona_id": r["persona_id"],
            "segment": personas[r["persona_id"]]["segment"],
            "price": r["price"],
            "level_sim": [round(x, 3) for x in level_sim],
            "pmf": [round(float(x), 3) for x in pmf],
            "expected_score": round(expected, 2),
            "text": r["text"],
        })

    def agg(rows):
        pmf = np.mean([r["pmf"] for r in rows], axis=0)
        return {"n": len(rows),
                "mean_score": round(float(np.mean([r["expected_score"] for r in rows])), 2),
                "dist_pct": [round(float(x) * 100, 1) for x in pmf]}

    summary = {}
    for price in DATA["prices"]:
        summary[f"price_{price}"] = agg([r for r in results if r["price"] == price])
    segments = sorted({r["segment"] for r in results})
    for seg in segments:
        for price in DATA["prices"]:
            rows = [r for r in results if r["segment"] == seg and r["price"] == price]
            summary[f"{seg}@{price}"] = agg(rows)

    out = {"concept": DATA["concept"], "summary": summary, "results": results}
    json.dump(out, open(OUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
