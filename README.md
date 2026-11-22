# ssr-lab

[![CI](https://github.com/nomuman/ssr-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/nomuman/ssr-lab/actions/workflows/ci.yml)

English | [日本語版 README](README.ja.md)

Experiments with **Semantic Similarity Rating (SSR)** — a method for turning
LLM-generated free-text survey responses into Likert-scale purchase-intent
scores, following Maier et al., *"LLMs Reproduce Human Purchase Intent via
Semantic Similarity Elicitation of Likert Ratings"* (arXiv:2510.08338,
PyMC Labs × Colgate-Palmolive).

## What is SSR?

Asking an LLM "rate this 1–5" produces flat, middle-of-the-road answers.
SSR instead:

1. Gives the LLM a persona (age, habits, budget sensitivity) and a product
   concept, and lets it answer **in free text**.
2. Embeds that text and measures cosine similarity against a set of
   **anchor statements** written for each Likert point.
3. Converts similarities into a probability distribution over the 5-point
   scale (`p(k) ∝ sim^T`), yielding an expected purchase-intent score.

## Dataset

128 labelled responses across **6 concepts in 5 domains** (Japanese),
each with 10–12 personas answering at 2 price points:

| Concept | Domain | Prices (JPY) | Items |
|---|---|---|---|
| `techo-ai-letter` | Diary app feature: AI monthly recap letter | 300 / 800 per month | 24 |
| `techo-handwriting` | Diary app feature: handwriting input | 300 / 800 per month | 24 |
| `protein-subscription` | Protein delivery subscription | 3,500 / 5,800 per month | 20 |
| `smart-humidifier` | Smart humidifier (one-time) | 12,800 / 19,800 | 20 |
| `language-app-premium` | Language-learning app premium plan | 980 / 1,980 per month | 20 |
| `frozen-bento-subscription` | Frozen bento delivery | 6,980 / 9,980 per month | 20 |
| `kids-phone-ai-qa` | Kids phone option: child asks AI questions by voice | 500 / 980 per month | 24 |

Every response ships with a **gold label** (1–5, rubric in CONTRIBUTING.md)
and a one-sentence **intent extract**. The responses deliberately include
realistic noise: polite negations, topic drift, trial-conditional answers.
Gold labels are lab-authored readings — *not* a real consumer panel
(see Limitations).

## Benchmark results (round 6, all 128 items)

MPNet-multilingual embeddings, v2 anchors, T=10:

| Method | MAE | Spearman | Price inversions |
|---|---|---|---|
| Extracted intent | **0.593** | **0.734** | 13/64 |
| Ensemble (whole + extract + sentence-weighted) | 0.646 | 0.730 | **9/64** |
| Whole text | 0.717 | 0.609 | 13/64 |

Findings so far:

- **The embedding model matters most.** Swapping MiniLM for
  paraphrase-multilingual-MPNet was the single biggest win in early rounds.
- **Across diverse domains, extraction beats whole-text.** On the original
  two concepts whole-text looked safest (0 inversions); with 6 concepts it
  degrades to 13/64 inversions, while extraction holds the best MAE.
  The ensemble is the balanced pick: near-best MAE, fewest inversions.
- **Calibration does not transfer across domains.** A linear stretch fitted
  on one concept group mostly *hurt* MAE on another (e.g. whole-text:
  0.724 raw → 0.774 transferred). Fit calibration per domain, with local
  labelled data — never import coefficients.
- Picking the single most confident sentence is a failure mode — discard it.
- Scores compress at both ends (a flat "absolutely not" still lands ≈2.9,
  a rave ≈3.8). Use SSR for **relative** comparisons (price A/B, concept
  ranking), not absolute prediction.
- **Direct Likert is not always worse.** In a control run (round 9),
  direct ratings elicited without seeing the responses scored MAE 0.454
  vs SSR's 0.672 — inside an all-LLM loop, direct elicitation shares the
  labeller's judgment. SSR's documented advantage is against *human*
  distributions; it also returns a per-item probability distribution
  instead of one forced number.

- Bigger isn't better by itself: multilingual-e5-large scored worse raw
  (MAE 0.983) but reached the top tier after calibration (LOO 0.501) —
  its ranking is strong, its raw scale is just heavily compressed.

Per-concept numbers and full history (including failures) are in
`REPORT.md` (Japanese) and `benchmark_results.json`.

## Use it as an agent skill (any LLM)

This repo ships an installable agent skill — the method, the scorer, and
the interpretation rules in one package, for Claude Code, Codex, Cursor,
and other skills-compatible agents:

```bash
npx skills add nomuman/ssr-lab
```

or copy `skills/semantic-similarity-rating/` into your agent's skills
directory manually. Scoring runs locally (fastembed + multilingual MPNet):
no API key, and response text never leaves your machine.

## Quick start

```bash
# Score one concept (whole-text, no extra steps)
uv run --with fastembed --with numpy python ssr_lab.py score data/protein-subscription.json

# Evaluate all concepts against gold labels
uv run --with fastembed --with numpy python ssr_lab.py evaluate

# Full benchmark incl. calibration-transfer test
uv run --with fastembed --with numpy python benchmark.py

# Validate the dataset (no model needed)
python -m unittest discover -s tests
```

Note: in proxied environments, Hugging Face downloads may need
`HF_HUB_DISABLE_XET=1` and a sanitized `NO_PROXY` (see REPORT.md round 1).

## Validation on real human text

The paper's own dataset (9,300 consumers) is not public, so we validated
the text→scale mapping on **MARC-ja** — 1,000 real Japanese Amazon reviews
(balanced 200 per star, test split) scored against their true star ratings
with satisfaction anchors:

| Anchors | MAE | Spearman | KS similarity |
|---|---|---|---|
| Satisfaction sentences (6/level) | 0.797 | 0.701 | 0.856 |
| Single words only | 1.024 | 0.592 | 0.840 |

Round 8 extended this to three real datasets (1,000 reviews each):
whole-text SSR held up across languages and sources — MARC-en
0.760 / 0.728 / 0.874, Yelp 0.793 / 0.714 / 0.847. Improvement attempts
were mostly inconclusive on real text: sentence-level scoring did not
beat whole-text on MAE, in-domain calibration gained little, and
halving the anchors slightly hurt MAE while *improving* KS similarity.

The mapping works on real human writing, with the same end-compression
seen in the synthetic lab (5-star reviews are hardest, MAE 1.167).
Note: this validates the mechanism on post-purchase *satisfaction*;
purchase *intent* itself remains validated only by the original paper.
Reproduce with `validation/validate_marc_ja.py` (data fetch instructions
in the script header; raw reviews are not redistributed here).

## Repository layout

```
data/        concept datasets (responses + gold + extracts per concept)
ssr_lab.py   packaged scorer (best-practice defaults, optional calibration)
benchmark.py generalized benchmark across all concepts
tests/       dataset validation (schema, coverage, label ranges)
legacy/      round 1-5 harnesses and raw outputs, kept for reproducibility
REPORT.md    full experiment log in Japanese, failures included
```

## Limitations

- Gold labels are lab-authored judgments, not measured consumer data.
  The original paper validated against 9,300 real survey responses; this
  repo validates *method mechanics* only.
- Personas and answers were generated by an LLM; no real users were surveyed.
- All concepts are hypothetical products.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a concept dataset.

## License

MIT
