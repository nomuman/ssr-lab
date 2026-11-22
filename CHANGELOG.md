# Changelog

## 0.3.0 — 2026-11-22
- Installable agent skill: `skills/semantic-similarity-rating/` — SKILL.md
  workflow + standalone scorer + purchase-intent/satisfaction anchors
  (JA/EN) + worked example. Install: `npx skills add nomuman/ssr-lab`
- Round 7 validation on real human text (MARC-ja, 1,000 reviews)

## 0.2.0 — 2026-11-22
- Dataset expanded from 2 to 6 concepts across domains (diary-app features,
  protein subscription, smart humidifier, language-app premium, frozen
  bento delivery): 128 labelled responses total
- New generalized `benchmark.py` with per-concept metrics and a
  calibration-transfer test
- Repository restructured: concepts under `data/`, round 1-5 harnesses
  under `legacy/`, dataset validation tests + CI, CONTRIBUTING guide

## 0.1.0 — 2026-11-22
- Rounds 1-5: SSR replication on two diary-app feature concepts
  (48 responses), 14-method comparison, packaged `ssr_lab.py` scorer,
  linear calibration (LOO-validated), e5-large evaluation
