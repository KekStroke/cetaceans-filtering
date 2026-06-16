# animal2vec checkpoint validation (turnkey)

Validates an animal2vec (`data2vec_multi`) pretraining checkpoint on **modern torch 2.9 + GPU** (the
legacy torch-1.13 fairseq stack is CPU-only on the RTX 5090). Loads losslessly by weights:
sanitizes Anvar-fork config keys (`multi_corpus_keys`, …), `skip_ema`, and patches version-mismatched
fairseq fns. Input = 10s @ 8kHz (80000) or native.

## Scripts
- `a2v_extract.py` — checkpoint + audio → mean-pooled encoder embeddings (`.npz`). GPU.
- `a2v_layer_sweep.py` — per-transformer-layer frozen-probe sweep (final layer is usually worst).
- `a2v_validate.py` — embeddings → frozen LogReg probe (recording-disjoint) + clustering (k-NN purity,
  silhouette, KMeans NMI/ARI, t-SNE) vs bars.

## First result — `checkpoint_2_25000` on Olga K-class (12-way)
- **best layer (L4) macro-F1 = 0.236** (all 16 layers 0.20–0.24; final 0.215). Above chance (~0.08) but
  far below the 8kHz ceiling (~0.58), log-mel (0.654), AVES (0.874).
- **Confounds (do NOT over-conclude):** only 8% of training (25k/320k); 8kHz vs the 16kHz bars; lab-clean
  clips vs field-passive pretraining domain; K-call-types are fine-grained.
- **Next:** Watkins species (coarser) + sound/noise filtration (closest to pretraining) to disambiguate
  "undertrained" vs "learned-coarse-not-fine"; re-probe a later checkpoint.
