# animal2vec — SHAP / frequency-band attribution (ckpt 25k, 8 kHz)

**TL;DR:** the encoder's (weak) discrimination leans almost entirely on the **lowest band (0–0.5 kHz)**;
it barely uses — and above 3.5 kHz actively *ignores* — the 1–4 kHz region where the orca-call
fundamentals/harmonics live. Attribution tracks the actual call energy only weakly (**r = 0.29**).
This is the **interpretability fingerprint of undertraining**: at ~8 % of the schedule the model has
learned coarse low-frequency energy cues, not fine spectral call structure. Consistent with the probe
verdict (K-class F1 0.25, Watkins species 0.54, below 8 kHz log-mel).

## Method
Occlusion-SHAP on the frozen linear probe (best layer **L3**, the strongest K-class layer). For each of
12 K-call classes × 10 clips: bandstop-remove each 0.5 kHz band (0–4 kHz, 8 bands), re-embed, measure the
**drop in the true-class probability**. Larger drop ⇒ band more important. 8 kHz ⇒ Nyquist 4 kHz, so the
analysis covers the full available spectrum. Memory-safe (1 s clips, one model, GPU, watchdog).

## Result (mean over classes)
| band | 0–0.5k | 0.5–1k | 1–1.5k | 1.5–2k | 2–2.5k | 2.5–3k | 3–3.5k | 3.5–4k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **attribution** | **+0.099** | +0.011 | −0.001 | −0.000 | −0.016 | +0.008 | −0.005 | **−0.039** |
| call energy | 0.292 | 0.085 | 0.099 | 0.153 | 0.142 | 0.081 | 0.090 | 0.057 |

- **Attribution collapses to the 0–0.5 kHz band.** Energy is spread across 0–2.5 kHz, but the encoder
  weights only the lowest band — it is **not** attending where the call structure is (Pearson r = 0.29).
- The **3.5–4 kHz band is slightly negative**: removing it *helps* the probe → the model treats the
  top of its 8 kHz range as noise rather than signal.

## Per-class (see `a2v_shap.png`)
- **noise**: identified by low-band presence (+0.19) **and absence of mid structure** (1–1.5 kHz = −0.31,
  2–2.5 kHz = −0.19) → the encoder separates noise mostly by gross spectral shape (relevant to filtration).
- **K21 / K27**: strongest low-band reliance (+0.29). K21's discriminative energy is >4 kHz — destroyed at
  8 kHz — so the model can only grab a low-band proxy. Matches the verdict's bandwidth point.
- **K5**: the exception — genuine **mid-band** use (1–1.5 kHz +0.28, 1.5–2 kHz +0.18). **K1**: broad, healthy
  decay from low→high. These few classes show the encoder *can* form mid-band features; most have not yet.

## What this adds to the verdict
1. Confirms **undertraining at the feature level**, not just the probe number: a well-trained call encoder
   should put attribution mass on the 1–4 kHz fundamentals; this one sits at 0–0.5 kHz.
2. **Cheap health-check going forward:** re-run on later checkpoints — as training continues (and at 16 kHz)
   attribution mass should migrate *up* into the call bands. That shift is the thing to watch.
3. **Filtration angle:** the noise class is linearly separable and the SHAP shows *why* (low-band-present +
   mid-band-absent). A dedicated binary signal/noise probe is the natural next check.

## Files
`a2v_shap.py` (turnkey, modern GPU) · `a2v_shap.json` · `a2v_shap.png`
