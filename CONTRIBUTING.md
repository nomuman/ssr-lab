# Contributing

## Adding a new concept dataset

Concepts live in `data/`. Each concept is three files:

| File | Contents |
|---|---|
| `data/<slug>.json` | `concept`, `prices` (exactly 2, ascending), `personas`, `responses` |
| `data/<slug>.gold.json` | `{"<persona_id>@<price>": <1-5>}` for every response |
| `data/<slug>.extracts.json` | `{"<persona_id>@<price>": "<intent in one sentence>"}` for every response |

Guidelines:

- **Personas**: 10-12 per concept, genuinely diverse (age, segment, budget
  sensitivity, category involvement). Each persona answers at both prices.
- **Responses**: natural consumer free text, 1-4 sentences. Aim for a mix —
  roughly 30% positive, 30% negative, 40% conditional/mixed. Include
  realistic noise: polite negations, topic drift, conditions ("if the trial
  is free…"). The messiness is the point; a dataset of clean answers tests
  nothing.
- **Gold labels**: judge the text as written, with this rubric —
  1 = absolutely not / out of the question, 2 = probably won't, 3 = unsure /
  would try first / conditional, 4 = probably will, 5 = definitely / right away.
  Label what the text says, not what the persona "should" feel.
- **Extracts**: one sentence stating the purchase intent only, stripped of
  reasons and topic drift.

Validate before submitting:

```bash
python -m unittest discover -s tests
```

## Running the benchmark

```bash
uv run --with fastembed --with numpy python benchmark.py
```

This scores every concept with the main methods and tests whether a
calibration fitted on one group of concepts transfers to another.
Results land in `benchmark_results.json`.

## Honesty rules

- Gold labels in this repo are lab-authored readings, not a real consumer
  panel. Never describe them as measured human data.
- Report failures in `REPORT.md` with the same prominence as wins.
- Calibration coefficients are dataset-specific until transfer is shown.
