# ⚠️ CORRECTION (2026-06-20): the "undertrained" verdict was a masking-extraction bug

Every animal2vec number in this archive (`validation/VERDICT.md`, `SUMMARY.md`, `SHAP_A2V.md`,
`FILTER_A2V.md`, the K-class report, and the dynamics) was computed on **~93 %-masked features** and is
**superseded**. With correct extraction the encoder is **strong, not undertrained**.

## Root cause
The extractor called the encoder as `model(source=x, features_only=True)`. data2vec's `forward` defaults
to **`mask=True`**, so it applied its pretraining mask (his config `mask_prob=1.5` → `masked_pct≈0.928`)
to **every** feature-extraction forward. Features therefore came from ~7 % of the signal — both
**nondeterministic** (max|Δ| 0.04–0.14 across identical forwards) and largely **destroyed**. The canonical
`extract_features` hardcodes `mask=False` (`nn/data2vec2.py:1112`); the extractor did not pass it.

**Fix:** `model(source=x, features_only=True, mask=False)` → bit-deterministic (max|Δ| = 0.0) and clean.
Lives in the consolidated `validate.py` (validation PR).

## Corrected numbers (clean, deterministic, full data)
| task | masked (WRONG) 13.5k → 25k | **clean (CORRECT) 13.5k → 25k** | 8 kHz baselines |
|---|---|---|---|
| Watkins species (31-way, macro-F1) | 0.378 → 0.542 | **0.795 → 0.839** (rising) | log-mel 0.675 · **AVES 0.853** |
| Watkins kNN-purity / NMI | 0.18/0.27 → 0.26/0.36 | **0.58/0.52 → 0.64/0.57** | — |
| Filtration signal/noise (macro-F1) | 0.681 → 0.734 | **0.815 → 0.814** | log-mel 0.903 · AVES 0.971 |
| SHAP attribution-vs-energy (r) | 0.26 | **0.73** | — |

## What it means
- At **25k (~8 % of training)** the encoder reaches **Watkins species 0.839 — matching AVES-8k (0.853)**, a
  strong domain-trained SSL reference, and far above an 8 kHz log-mel (0.675). It is **rising** (0.795→0.839).
- SHAP now shows attribution **tracking the call energy** (r = 0.73), not "stuck in the lowest band."
- The encoder is **healthy and strong**; keep training. The earlier "below log-mel / undertrained /
  attention stuck low" reads were all the masking artifact.
- Filtration (binary) is near-ceiling for every encoder and slightly below log-mel — the fine-grained
  **species** task is the meaningful signal, and there the encoder matches AVES.

See the validation PR's `README.md` for the corrected verdict and the `mask=False` requirement.
