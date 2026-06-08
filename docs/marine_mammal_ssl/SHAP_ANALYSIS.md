# SHAP / Attribution Analysis — dataset & model (iaisheipak task)

*Consolidates `ssl_shap_v1/2/3` (frozen-encoder frequency-band attribution) and adds a new tie-in to the 8 kHz finding. captum ShapleyValueSampling / KernelSHAP over 16 log-frequency bands (0–8 kHz), probe-logit targets, AVES / wav2vec2 / Whisper.*

## 1. What the encoders rely on (existing ssl_shap_v2/v3)
- **AVES is the most faithful & domain-stable interpreter.** Frozen probe: lab macro-F1 **0.879**, field **0.677**. **Lab↔field band-profile correlation = 0.853** — the bands AVES uses on clean lab clips are the same it uses on noisy field events ⇒ interpretability survives the domain gap (more stable than wav2vec2/Whisper).
- **Revelatory (v3) vs confirmatory (v2).** Marginal SHAP just recovers where the call *energy* sits (attr–energy Pearson 0.62). The **contrastive target** (probe-logit margin of each class vs its data-driven confusable partner) drops that to 0.51 and surfaces the **discriminative** bands. **Faithfulness deletion-AUC passed** — ablating high-|attr| bands collapses the margin faster than ablating by energy.
- The call-fundamental (0.5–3.5 kHz) carries most mass, but several call-types reach well above 4 kHz (§2).

## 2. NEW — SHAP explains WHY 8 kHz hurts (ties §0.5-B to interpretability)
For each call-type, the fraction of AVES SHAP attribution sitting in the **>4 kHz bands** (exactly what the 8 kHz pipeline's 4 kHz Nyquist destroys) vs that call-type's spectral energy >4 kHz (Exp2):

> **Pearson(energy >4 kHz, SHAP attribution >4 kHz) = 0.867** across 11 call-types.

| call | energy >4 kHz | SHAP attr >4 kHz |
|---|---:|---:|
| **K21** | **52.0%** | **77.1%** |
| K13 | 28.5% | 43.6% |
| K1 | 22.6% | 23.8% |
| K14 | 24.8% | 23.6% |
| K7 | 28.5% | 18.7% |
| K5 | 16.4% | 18.4% |
| K27 | 23.0% | 18.2% |
| K4 | 22.9% | 16.4% |
| K12 | 26.8% | 12.8% |
| K17 | 22.4% | 11.9% |
| K10 | 26.0% | 9.0% |

**Mechanistic conclusion:** the 8 kHz pipeline doesn't merely lose energy — it **destroys the exact frequency bands the encoder relies on to classify**, worst for **K21** (52% of energy and 77% of attribution live >4 kHz). This is the mechanistic *why* behind the **+10.2 pt 8→16 kHz gain** (§0.5-B): moving to 16 kHz restores the discriminative bands SHAP shows the model actually uses. (`shap_bandwidth_tiein.{json,png}`)

## 3. For the paper
Three independent analyses converge into one interpretability story: AVES's attributions are **faithful** (deletion test), **domain-stable** (r=0.85 lab↔field), and **mechanistically predictive of the sample-rate effect** (r=0.87 attribution↔energy). That triangulation — benchmark + bandwidth + SHAP all pointing the same way — is a strong, defensible figure.
