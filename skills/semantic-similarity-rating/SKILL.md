---
name: semantic-similarity-rating
description: Estimate 5-point Likert ratings (purchase intent, satisfaction) from free-text responses using Semantic Similarity Rating (SSR) — never ask an LLM for a number directly. Use when the user wants to run a synthetic consumer survey, test a product concept or price point, compare concepts, score open-ended survey/review text on a 1-5 scale, or reproduce the SSR method from Maier et al. (arXiv:2510.08338). Works with any LLM as the respondent; scoring runs locally with a bundled script (no API key).
---

# Semantic Similarity Rating (SSR)

Direct Likert elicitation fails: asked "rate this 1–5", LLMs cluster on
safe middle answers (the paper measured distribution similarity of only
0.26). SSR fixes this by separating *expression* from *measurement*:

1. The respondent (an LLM persona, or a real human) answers in **free text**.
2. The text is embedded and compared by cosine similarity to **anchor
   statements** written for each scale point.
3. Similarities become a probability distribution over the scale
   (`p(k) ∝ sim^10`); the expected value is the score.

Validated in the paper on 9,300 real consumers (KS similarity > 0.85,
~90% of human test–retest reliability) and in the ssr-lab benchmark
(6 concepts, 128 labelled responses; plus 1,000 real Japanese Amazon
reviews, KS similarity 0.856).

## Workflow

### 1. Define the study
- One concept, stated concretely (what it is, what it costs).
- 2 price points if price sensitivity matters.
- The scale: **purchase intent** (default) or **satisfaction** (for
  post-purchase text such as reviews).

### 2. Build the panel
- 10–12 personas per concept, genuinely diverse: age, segment, budget
  sensitivity, category involvement.
- Each persona answers **in character, in free text**, at each price.
  Prompt them for an honest reaction in their own words — never for a
  number. Expect realistic messiness: conditions, polite negations,
  topic drift. That messiness is what SSR is for.
- Real human free-text responses can be scored the same way (skip to 4).

### 3. Extract intents (recommended)
For each response, write **one sentence** stating only the purchase
intent, stripped of reasons and drift. Extraction was the best single
method in the ssr-lab benchmark across diverse domains (MAE 0.593 vs
0.717 for whole text). Keep the original text too — the ensemble uses both.

### 4. Score
Bundled script (local embeddings via fastembed, multilingual MPNet —
no API key, nothing leaves the machine except the one-time model download):

```bash
uv run --with fastembed --with numpy python scripts/ssr_score.py responses.json \
    --anchors purchase_ja --mode ensemble --out scores.json
```

- `--anchors`: `purchase_ja` / `purchase_en` / `satisfaction_ja` /
  `satisfaction_en`, or a path to a custom anchors JSON
  (`{"1": [...], ..., "5": [...]}`, 3–6 statements per point).
- `--mode`: `whole` (no preparation), `extract`, or `ensemble`
  (recommended when extracts exist: best accuracy/inversion balance).
- Input formats: the lab format (`concept` / `prices` / `personas` /
  `responses`) or a simple list of `{"id", "text", "extract"?, "group"?}`.
  See `assets/example_responses.json`.

### 5. Report
- Means by price and by segment; the score **distribution**, not just means.
- The main objections and delights **in the respondents' own words** —
  the qualitative text is at least as valuable as the numbers.
- Apply the interpretation rules below. Always.

## Interpretation rules (from the ssr-lab benchmark — do not skip)

- **Relative comparisons only.** SSR scores compress at both ends
  (a flat "absolutely not" lands ≈2.9; a rave ≈3.8). Use SSR to rank
  concepts and compare prices, never as an absolute demand forecast.
- **Calibration does not transfer across domains.** A linear correction
  fitted on one product category mostly *hurt* accuracy on another.
  Only calibrate with labelled data from the same domain, and say so.
- **Extraction caveat:** extraction occasionally flips a single item
  badly. For high-stakes reads, use `ensemble`, not `extract` alone.
- **Sentence picking is a trap.** Scoring only the "most confident
  sentence" of a response was the worst method tested. Never do it.
- **Bigger embedding models are not automatically better.** The bundled
  MPNet default beat a much larger model on raw scores; the larger model
  only caught up after calibration.
- **Synthetic ≠ real.** Persona answers are LLM-generated. Present
  results as a fast first-pass screen before real research — never as
  measured consumer data, and never invent human validation that
  did not happen.

## Files

- `scripts/ssr_score.py` — standalone scorer (whole / extract / ensemble)
- `assets/anchors_purchase_intent_ja.json`, `assets/anchors_purchase_intent_en.json`
- `assets/anchors_satisfaction_ja.json`, `assets/anchors_satisfaction_en.json`
- `assets/example_responses.json` — worked example (Japanese, 3 personas × 2 prices)

Full benchmark data, method comparisons, and failure log:
https://github.com/nomuman/ssr-lab
