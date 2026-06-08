# Training Speed — animal2vec (and the BEST-RQ alternative)

*Iaroslav's "посмотрю как ускорить обучение" deliverable. Grounded in the data2vec-2.0 paper + the cetaceans-filtering pipeline diagnosis + our measured BEST-RQ run.*

## Why animal2vec is slow — root causes

animal2vec = **data2vec 2.0** applied to raw 8 kHz audio ([repo: livingingroups/animal2vec](https://github.com/livingingroups/animal2vec), [paper arXiv 2406.01253](https://arxiv.org/abs/2406.01253)). data2vec-2.0's speed is *entirely* about amortizing the EMA-teacher cost — if the config doesn't amortize it, you pay roughly 2×.

1. **EMA-teacher forward every step.** data2vec needs a full **teacher** forward (the EMA copy of the student) to build the regression target, *on top of* the student forward. Naïvely that ~doubles per-step compute.
2. **The fix is MULTIMASK (M / `clone_batch`).** data2vec-2.0's headline efficiency — **16.4× vs MAE (vision), 10.6× less time than wav2vec 2.0 (speech) at equal accuracy** ([Meta](https://ai.meta.com/blog/ai-self-supervised-learning-data2vec/), [arXiv 2212.07525](https://arxiv.org/abs/2212.07525)) — comes from creating **M masked versions of each sample and reusing ONE teacher representation across all M**. As M grows, the teacher cost becomes negligible. **If Anvar's multimask `clone_batch`/M is small (or 1), he's running at ~data2vec-1.0 speed and leaving the entire 2.0 speedup on the table.** ← almost certainly the #1 lever.
3. **Don't-encode-masked-tokens + fast conv decoder** — both are data2vec-2.0 efficiency features; confirm they're enabled (the upstream config has them).
4. **CPU-bound data pipeline.** The cetaceans-filtering loader writes one tiny WAV per 10 s chunk and resamples per read (verified) — over millions of files this starves the GPU (the canonical fairseq bottleneck).
5. **Raw-waveform @ 8 kHz** → long sequences through the conv frontend + O(T²) attention.
6. fp16 instability / small batch / no `torch.compile`.

## How to speed it up — ranked, with expected gain

| # | lever | expected | how |
|---|-------|---------:|-----|
| 1 | **Raise multimask M** | up to ~M× on the teacher-bound part | set `clone_batch`/M to 8–16 — amortizes the EMA-teacher forward (the core data2vec-2.0 win) |
| 2 | **Fix the data pipeline** | ~1.5–2.5× | pre-shard to WebDataset/int16, `num_workers`/`persistent_workers`, no per-step resample (already 8 kHz) |
| 3 | **bf16 autocast (NOT fp16)** | ~1.3–1.7× + stability | also kills the loss-spike/NaN risk |
| 4 | **torch.compile + SDPA/flash attn** | ~1.2–1.5× | |
| 5 | **larger batch + grad accumulation** | raises MFU/GPU-util | |
| 6 | **switch workhorse to BEST-RQ** | **~2.4× and no teacher at all** | fixed random-projection quantizer target = zero target-compute, no EMA; interpretable CE loss |

Gains multiply *within* the pipeline (2) and compute (3–5) tiers but not across; realistic stacked single-GPU gain ≈ **2.5–4×** on top of the multimask fix.

## BEST-RQ vs animal2vec — why we built BEST-RQ (`ml-intern-runs/bestrq-marine/path-2/`)

| | animal2vec (data2vec-2.0) | BEST-RQ (ours) |
|---|---|---|
| SSL target | EMA-teacher contextualized repr (needs teacher fwd) | **fixed random-projection quantizer — zero compute, frozen** |
| per-step cost | student + (amortized) teacher | **student only** |
| loss | latent MSE — **uninterpretable** (the "loss is huge" panic) | **cross-entropy from ln(8192)=9.0 — interpretable** |
| "did it train?" | needs a probe to even tell | readable from the loss + masked-acc directly |
| measured throughput (RTX 5090, 14 M params) | — | **~1455 clips/s** (1 s audio each ≈ 1455× realtime), bf16, batch 64 |

**Measured demo** (`ml-intern-runs/bestrq-marine/path-2/`, 750 steps before a reboot cut it short): masked-CE loss fell **9.19 → 3.88** (interpretable, from ln(8192)=9.01), masked-acc **0 → 0.31**, target-perplexity stayed **~133** (no collapse), and the frozen K-class probe rose **0.642 → 0.695** — note the random-init encoder already ≈ log-mel (0.654) because the front-end is a fixed mel, so the honest SSL gain is **+0.04 over log-mel in 750 steps**. The harness + interpretable loss + health metric all work; this is Anvar's drop-in (`bestrq.py` + `bestrq_train.py`, swap the audio loader for his corpus). Throughput ~1455 clips/s on a *laptop* GPU shows the method is cheap.

→ For the team's **twin pains (slow + uninterpretable loss), BEST-RQ fixes both at once.** Keep animal2vec as the "uninterpretable-loss motivation" baseline; make BEST-RQ the workhorse.

## Immediate checklist for Anvar
- [ ] Print the effective **multimask `clone_batch`/M** in his config — if small, raise it (biggest free win).
- [ ] Confirm **bf16** (not fp16) autocast.
- [ ] Profile one step (`nvidia-smi dmon` + `torch.profiler`): if GPU-util < 80% → data-pipeline-bound → shard the corpus.
- [ ] Try the **BEST-RQ trainer** as the fast, interpretable alternative (drop his corpus in place of the Olga audio loader).

**Sources:** [data2vec 2.0 (arXiv 2212.07525)](https://arxiv.org/abs/2212.07525) · [Meta data2vec 2.0 blog](https://ai.meta.com/blog/ai-self-supervised-learning-data2vec/) · [animal2vec (arXiv 2406.01253)](https://arxiv.org/abs/2406.01253) · [animal2vec repo](https://github.com/livingingroups/animal2vec)
