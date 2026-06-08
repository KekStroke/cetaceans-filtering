# Marine Mammal SSL — Plan & Analysis (June 2026)

> Program lead's master action plan. Integrates 6 expert analyses (Data, SSL-Loss, Speed, Benchmark, PR, Roadmap/Method) **and a full adversarial review pass**. Grounded in the two decks (`plan_analysis/deck_SSL.txt`, `deck_ROADMAP.txt`), the cloned repo (`/tmp/cetaceans-filtering`, on `main`), and verified local facts. Web citations preserved inline. Contradictions between experts are resolved in **§9**; corrections applied during review are listed in **§10**.

---

## ⚠️ Confidence & unverified numbers (read first)

This plan is decision-ready for **week-1 diagnostics**, but several load-bearing numbers are **estimates pending a P0 manifest/header audit**. Nothing downstream of these should be quoted as fact until the audit lands.

| Figure | Status | Why |
|---|---|---|
| **All hour counts** (3,089 h, 2,055/296/432/576, the 0.14% universe share, the 5k/10k gaps) | **ESTIMATED, UNAUDITED (±)** | Derived from `assume_sample_rate_hz` + `assume_minutes_per_file=5.0` in the repo — **no file's true duration is measured.** See §1.0. |
| **3,089 h vs 3,359 h** | **DISCREPANCY — unresolved until audit** | The §1.1 table **sums to ~3,359 h**, but the load-bearing number used everywhere is 3,089 h. One source is double-counted or the per-source estimates disagree. **Reconciled, not hidden** — §1.0. |
| **Disk footprints** | Stated for **BOTH 8 kHz and 16 kHz** | The headline is 16 kHz; 8 kHz figures **halve** the real footprint. §4.1. |
| **animal2vec mask params** (`p=0.15/span M=2`, "~96% masked") | **UNVERIFIED** (same caveat as the EMA-tau schedule) | The "~96%" is arithmetically impossible from those values; flagged, not asserted. §3.3. |
| **BEST-RQ speedup** | **~2.4× on the cited speech setup — to be re-measured on our config** | Single speech data point (262 vs 109 GPU-h, 8×V100, 83 M, 200k steps). Not a guarantee for a ~20-30 M marine model. §4. |
| **$ cost / clips-per-sec / epoch time** | **Projection — gated on wk1-2 profiling** | Stacks unmeasured bf16+compile+SDPA+BEST-RQ gains on sm_120. Pessimistic case included. §4.5, §9 #5. |
| **Watkins ~32 species** | **UNVERIFIED** against `confit/wmms-parquet` | Class count drives the whole task-2 design; confirm from the parquet, not the deck. §6.1. |
| **Filter→retrain "v1 > v0" delta** | **Currently n=1 per arm** | The SSL *pretraining* is single-run; needs seeds/subsets, a negative control, a threshold sweep, and a leakage-disjoint eval source before it can headline. §6.2.x, §6.4. |

**Rule for the team:** every hour figure below carries an implicit "estimated, unaudited (±)". The **manifest+header audit (§1.0) is P0 and runs before any number is quoted in the paper.**

---

## 0. TL;DR

- **Where we are:** Phase 1 of the 4-phase ROADMAP — the *Foundation SSL Encoder* inner loop (pretrain SSL → ablate arch×size×SR → eval transfer Detection/Classification/Discovery → train sound-filter → filter corpus → **retrain on filtered data → compare**). **The whole paper = Phase 1.** Phases 2-4 (tokenizer+LLM, translation, "Doctor Dolittle") are one future-work paragraph.
- **Is the data enough?** **Borderline-yes for a *small* encoder, no for the 315M default — and the hour count is itself unaudited.** We hold **~3,089 h (estimated, unaudited; the inventory table actually sums to ~3,359 h — see §1.0)** of unlabeled passive audio ≈ **~0.14% of the ~2.17M-h universe**. That sits *above* animal2vec (1,068 h) and *below* AVES-all / AudioSet (~5,000-5,800 h). But raw passive hours overstate usable signal: AVES hit topline on **360 h that were warm-started from 960 h of speech and curated to event-bearing audio**, whereas our corpus is mostly continuous ocean noise (§1.4). **Sufficiency must be judged on EFFECTIVE event-bearing hours after filtering, not raw passive hours.** Provisional target: **5,000-10,000 h** for the headline run, revisited after the audit.
- **Quality is the real blocker, not scale:** everything is force-resampled to **8 kHz (Nyquist 4 kHz)** in `audio_saver.py` — this strips orca **echolocation clicks (20-60 kHz)** and the **6-12 kHz peak of whistles** ([SeaWorld/JASA](https://seaworld.org/animals/all-about/killer-whale/communication/)). The SSL masked-prediction target becomes mostly low-freq ambient/ship noise. **Move to 16 kHz** (matches AVES/wav2vec2/WavLM/AudioMAE/BEATS) and keep a native **≥32 kHz high-band subset** (Pacific 256 kHz tier + ONC 64 kHz FLAC). **We will quantify the actual information loss above 4 kHz (§1.7) to justify the costly 16 kHz move with numbers, not assertions.**
- **Is the "VERY BIG loss" broken? Almost certainly NOT.** The default is **animal2vec / data2vec-2.0** ([arXiv 2406.01253](https://arxiv.org/abs/2406.01253)): a **latent-regression** of a masked student onto an EMA-teacher target. That MSE is **scale-dependent and uninterpretable** — its magnitude is an artifact of the normalized teacher-target scale, not a quality score. **The number being "big" tells you nothing.** Broken-vs-fine is decided by **trend + target-variance (collapse check) + grad-norm + a frozen linear probe**.
- **Single most important next action:** **Wire a frozen linear probe + a 3-signal logger (loss-trend, grad-norm pre-clip, teacher target-variance) onto the existing run THIS WEEK.** Use **two disjoint probes** — `new_training_data/` (lab K-class) **and** Watkins species — and require **both** to move (§3.2). If both probes climb and target-variance stays > 0, the model is training and the panic ends — at near-zero cost, before buying any compute.
- **Fastest path to a paper:** (1) prove the loss is fine via the probes; (2) switch the *workhorse* SSL method to **BEST-RQ** (interpretable cross-entropy loss starting ≈ `ln(8192)=9.0`, ~2.4× faster *on the cited speech setup*, no EMA), keep AVES/HuBERT as baseline + data2vec2 as a single "uninterpretable-loss" motivation run — **with an explicit fallback rule if BEST-RQ underperforms AVES on marine transfer (§5.1)**; (3) fix the **data pipeline** (pack to WebDataset/int16 shards → ~2.5-4× speedup); (4) build the **frozen-probe BEANS-style benchmark** over the obtainable tasks; (5) run the **filter→retrain A/B with leakage discipline, a negative control, and a threshold sweep** — *that delta is the headline*.
- **Minimal-viable headline (commit to this):** tasks **{1 Filter, 2 Watkins, 4 Orcasound, 5 Olga}** — all in hand or with a working downloader. **Dominica sperm-whale clicks and Dolph2Vec are explicit STRETCH/appendix** (login-walled / no open endpoint); the paper must stand without them (§6.1, §9 #8).
- **Speed:** the "loss is big" and "training is slow" pains **share a root cause and a shared fix.** Profile first (`nvidia-smi dmon` + one `torch.profiler` trace): the repo writing **one tiny WAV per 10 s chunk** + per-step `librosa.resample` makes it **pipeline/CPU-bound** (the canonical fairseq failure). Re-shard once → ~1.5-2.5× alone; stack bf16 + `torch.compile` + SDPA + fused AdamW → **~2.5-4× single-GPU**; BEST-RQ adds a **method-level win (~2.4× on the cited speech setup, to be re-measured)**.
- **PR status:** the proposed change — **enable 4 NOAA SSL families (`dclde/ pifsc/ nefsc/ afsc/`)** in `data_loading.yaml` — is **already applied to the working tree on `main` (uncommitted)** but the branch `data/enable-noaa-ssl-prefixes` does **not exist yet**. Config-only, `+16/−4`, reuses tested code, verified safe against the live `noaa-passive-bioacoustic` bucket. Needs: branch it, commit, open PR.
- **Benchmark thesis:** *at equal frozen linear probe, our marine-pretrained SSL encoder matches/beats AVES & Perch 2.0 on marine tasks; and **filtering the corpus then retraining the SSL encoder measurably improves it — demonstrated on an eval source DISJOINT from the filter-training corpus, against a same-size random-subset control**.* We are the **only marine-pretrained SSL encoder** in the grid — that is the gap the paper fills.

---

## 0.5 ✅ MEASURED on the local RTX 5090 (this session)

Two decision-relevant experiments run locally on Olga's K-call clips (native 48 kHz, **recording-disjoint** split by tape key — the anti-leakage discipline from §6.2.1). Scripts: `exp1_frozen_probe.py`, `exp2_bandwidth.py`; raw results + figure in `experiments_5090/`.

**A. BenchSuite v0 — frozen-probe benchmark, the bar the marine SSL encoder must beat (§6).** 4 encoders × 2 local tasks, frozen embeddings → LogReg, **GroupKFold-5 by recording** (leakage-safe). Pooled macro-F1 (± fold std):

| Encoder | Olga K-class (12-way) | Olga field (11-way) |
|---|---:|---:|
| **AVES** (animal SSL) | **0.874 ± 0.01** | **0.717 ± 0.06** |
| whisper-tiny (ASR) | 0.843 ± 0.01 | 0.508 ± 0.06 |
| log-mel + LR | 0.654 ± 0.01 | 0.550 ± 0.11 |
| wav2vec2-base (speech SSL) | 0.647 ± 0.02 | 0.423 ± 0.05 |

→ **AVES (animal-domain SSL) wins both tasks** — the bar to beat at equal probe (K-class 0.874, field 0.717). On *last-layer* probing wav2vec2 looks weak (K-class 0.64) — **but A.1 below shows that's a last-layer artifact** (its best layer hits 0.875, near AVES), so the "domain ≫ SSL" read was overstated. Nuance worth a sentence in the paper: **whisper-tiny is a strong *clean*-clip probe (0.843) but collapses on field (0.508)** — ASR mel-features capture clean call structure but not in-situ/noisy detection. The **field task is far harder for every encoder** (best 0.72 vs 0.87) — the real-world gap the paper must close. (`bench_suite.py`; BEATs/Perch2 not locally loadable → added later. Watkins T2 obtainable: `DBD-research-group/beans_watkins`.)

*Probe-protocol robustness (autoresearch, `autoresearch-runs/probe-olga-kclass/`):* an 8-config sweep of the probe head (MLP / SVM-RBF / PCA / C / L2-normalize) **found nothing beats LogReg + StandardScaler beyond fold noise** — the head is not the lever, the encoder is; the **StandardScaler is the one load-bearing component** (removing it → −0.052). This de-risks the frozen-probe protocol and pre-empts "did you tune the probe?". **But the *layer* IS a lever (gen-2):** probing AVES **layer 5–7 instead of the last layer lifts K-class macro-F1 0.873 → 0.894 (+2.1 pts, beyond fold-noise)** — the classic wav2vec2 "mid-layers transfer best" effect (`layer_sweep.py`). ⇒ the **fair AVES bar is 0.894**, and the benchmark must use **per-encoder best-layer selection** (apply to wav2vec2/whisper too), not last-layer, to be fair.

**A.1 — Fair per-encoder best-layer (the rigorous Table 1; corrects A).** Each encoder probed at its *best* layer, not last (Olga K-class, recording-disjoint):

| encoder | last-layer | best-layer | Δ |
|---|---:|---:|---:|
| AVES (animal SSL) | 0.873 | **0.894** (L5) | +0.021 |
| wav2vec2-base (speech SSL) | 0.644 | **0.875** (L2) | **+0.231** |
| whisper-tiny (ASR) | 0.840 | **0.875** (L1) | +0.035 |
| log-mel | 0.654 | 0.654 | — |

**⚠️ This overturns the v0 (last-layer) read.** wav2vec2 isn't weak — at its best layer it **ties whisper and nearly matches AVES** (0.875 vs 0.894). The "generic speech-SSL ≈ log-mel, domain matters" conclusion was a **last-layer artifact**. Honest claim for the paper: *at fair best-layer probing, all SSL encoders cluster at 0.875–0.894 and clearly beat log-mel (0.654); animal-domain AVES has only a ~0.02 edge on clean orca calls.* The real domain advantage must be demonstrated on the **harder field/marine tasks** (where v0 already shows a wider spread), not clean K-class. **Methodological takeaway worth a paragraph: last-layer probing can flip benchmark conclusions — always select the probe layer per encoder.**

**B. 8 vs 16 vs 32 kHz information-loss (§1.7) — resolves the re-download decision.** Identical CNN, 14,400 clips, bandwidth varied by masking the 48 kHz log-mel above each Nyquist:

| Effective SR (Nyquist) | acc | macro-F1 |
|---|---:|---:|
| 8 kHz (4 kHz) — current pipeline | 64.3% | 0.578 |
| **16 kHz (8 kHz)** | **74.5%** | **0.740** |
| 32 kHz (16 kHz) | 77.1% | 0.750 |
| 48 kHz native (24 kHz) | 69.3% | 0.682 |

→ **8 → 16 kHz buys +10.2 pts accuracy (+0.16 macro-F1).** Mean **25.8%** of spectral energy sits above the 8 kHz-Nyquist cut (**K21 = 52%**). Returns diminish fast above 16 kHz (16→32 only +2.6; native 48 *drops* — for orca *calls* almost all energy is <16 kHz, so extra bandwidth is just noise to a fixed-capacity model). **Verdict: 16 kHz is the empirically-justified headline SR for the orca tasks; 32 kHz only earns its keep on the click/whistle stretch tasks.** Caveat: masking ablation, single CNN seed — directional, not a final number. **SHAP corroborates the mechanism:** per-call-type AVES attribution in >4 kHz bands correlates **r=0.87** with energy >4 kHz (K21: 52% energy, 77% attribution) — 8 kHz destroys the exact bands the encoder uses to classify, so this isn't just lost energy but lost *discriminative* signal (`plan_analysis/SHAP_ANALYSIS.md`).

---

## 1. Data Inventory & Sufficiency

### 1.0 ⚠️ P0 — the hour counts are estimates, and the table does not foot

**Before any number in this section is used in the paper, run the manifest+header audit.** Two problems must be resolved first:

1. **All hour counts are `assume_*` estimates, not measurements.** The repo derives durations from `assume_sample_rate_hz` and `assume_minutes_per_file=5.0` — **no file's true duration is read.** Every "~N h" below is therefore **estimated, unaudited (±)**. (Note `assume_minutes_per_file=5.0` while clips are stored as 10 s segments — itself a sign the estimate basis is loose.)
2. **The inventory sums to ~3,359 h, but we use ~3,089 h everywhere.** SanctSound 2,055 + ONC 296 + Pacific 432 + Orcasound 576 = **3,359 h**, yet the load-bearing figure (the 0.14% share, the animal2vec/AVES positioning, the "+1,900 to 5k / +6,900 to 10k" gaps) is **3,089 h**. The gap math is internally consistent with 3,089, so either a source is **double-counted** (most likely Orcasound, which appears in both pretraining and labeled-eval rows) or the per-source estimates disagree. **This ~270 h (~9%) discrepancy is flagged here rather than hidden; it is resolved by the audit, not by picking a number.**

**P0 audit procedure:**
- For **every file**, read the **real decoded header** via `soundfile.info(path)` — true sample rate, frames, channels — and compute true duration = `frames / samplerate`. **Do not** trust `assume_sample_rate_hz`.
- Sum true hours **per source and per recording/deployment ID**; produce a `manifest_audit.csv` (one row/file: path, source, deployment_id, true_sr, true_seconds).
- **Reconcile 3,359 vs 3,089:** identify any double-counted recordings (cross-check Orcasound pretrain vs labeled rows by recording ID).
- **Replace every "3,089 h" / per-source estimate in the paper with the measured total.** Until then they read "estimated, unaudited (±)".

This audit also feeds the leakage manifest (§6.2.1), the sub-rate decision (§1.5 #3), and the effective-hours estimate (§1.4).

### 1.1 UNLABELED — already downloaded (~3,089 h *estimated, unaudited*; table foots to ~3,359 h — see §1.0)
*Verified: `audio_saver.py` resamples to `raw_sample_rate` via `librosa.resample`; `_should_skip_sample_rate` discards sub-rate files (but the SR it tests may be the assumed constant, not the real header — see §1.5 #3); one WAV per 10 s chunk via `write_wav`.*

| Source | Hours (est., unaudited) | Native SR | Stored | Notes |
|---|---:|---|---|---|
| NOAA SanctSound (19 deployments) | ~2,055 | ~48 kHz (assumed) | 8 kHz / 10 s | `assume_sample_rate_hz=48000` (assumed, not read); 20 active per-site prefixes |
| ONC (6 sites) | ~296 | ~64 kHz | 8 kHz / 10 s | FLAC; 6 `location_code` targets |
| Pacific Sound | ~432 | **16 kHz tier** (native 256 kHz) | 8 kHz / 10 s | `tier: 16khz` → already decimated, then 8 kHz again (**double-decimation**) |
| Orcasound | ~576 | ~48 kHz (assumed) | 8 kHz / 10 s | AWS Open Data |
| **TABLE SUM** | **~3,359** | mixed → 8 kHz | — | **does NOT match the 3,089 used downstream — §1.0** |
| **Load-bearing figure used downstream** | **~3,089** | — | — | **~0.14% of universe; pending audit** |

### 1.2 UNLABELED — available universe (NOT downloaded; SSL deck slide 2)

| Source | Hours avail. | SR | Purpose |
|---|---:|---|---|
| Orcasound (labeled) | ~1,900 | 44.1 kHz | validation / SSL |
| Orcasound (archive) | ~14,000 | 48 kHz | SSL pretrain |
| NOAA | ~773,000 | 2-500 kHz | SSL pretrain |
| Pacific Sound | ~85,825 | 256 kHz | SSL pretrain |
| ONC | ~1,300,000 | 64 kHz mostly | SSL pretrain |
| **Universe** | **~2.17M h** | — | 3,089 h ≈ **0.14%** (estimated) |

### 1.3 LABELED — downstream / validation

| Set | Size | SR | Status | Use |
|---|---|---|---|---|
| Olga K-calls raw (`training_data/`) | 21,077 call + 11,974 noise | clip | **local** | supervised train/eval |
| → per-class | K1=2650, K5=2012, K7=1180, K4=940, K3=650, K12=428, K10=368, K17=315, K21=211, K13=177, **K14=94, K27=78** | — | local | **34:1 imbalance** |
| Olga balanced (`new_training_data/`) | **45,429** WAVs (~3k/class ×11 + ~12k noise; labels in **filename prefix**) | clip | **local** | ConvNeXt V2 Pico **97.99%** |
| Olga **field** (`OLGA/new_testing_files_2016/`) | **17 files ≈ 1.22 h** | 48 kHz | local | in-situ eval (field-FT ensemble **F1=53%**) |
| Watkins MMSDB | ~5 h, 600 Hz-166 kHz | varies | **not downloaded** | species classification (`download_watkins.py`, HF `confit/wmms-parquet`) — **species count UNVERIFIED, confirm from parquet** |
| Voices in the Sea | 0.47 h | 5-48 kHz | **not downloaded** | validation |
| Orcasound (labeled) | ~1,900 h | 44.1 kHz | **not downloaded** | orca sound ID |
| Dominica sperm-whale clicks | ~17.8 GB, 192 kHz | high | **not downloaded** (IEEE DataPort, login-walled) | click detection — **STRETCH/appendix only** |
| Dolph2Vec dolphin calls | ~180k whistles | — | **not downloaded** (no open bulk endpoint) | dolphin classification — **STRETCH/appendix only** |
| Manual Sound/Noise filter set (deck slide 4) | ~15 h | — | planned | binary sound detector (powers filter→retrain) |

### 1.4 Sufficiency verdict — judge on EFFECTIVE event-bearing hours, not raw passive hours
**Raw scale is *just* workable for a small encoder; the 8 kHz resampling is the dominant problem — and the comparison to AVES must not flatter the corpus.** Web-verified SSL pretraining budgets:

| Model | Pretrain hrs | From scratch vs warm-started | SR | Source |
|---|---:|---|---|---|
| wav2vec 2.0 | 960 (speech) | **from scratch on speech** | 16 kHz | [AVES paper](https://arxiv.org/pdf/2210.14493) |
| **animal2vec** (MeerKAT) | **1,068** (184 labeled) | from scratch on MeerKAT | 8 kHz | [arXiv 2406.01253](https://arxiv.org/abs/2406.01253) |
| AVES core/bio/**all** | 153 / 360 / **5,054** | **CONTINUED-pretraining: warm-started from HuBERT (960 h speech), then curated event-bearing animal audio** | 16 kHz | [earthspecies/aves](https://github.com/earthspecies/aves) |
| BEATs / AudioMAE (AudioSet-2M) | **~5,833** | from scratch on AudioSet | 16 kHz | [arXiv 2212.09058](https://arxiv.org/abs/2212.09058) |
| Perch 2.0 (**supervised**, not SSL) | ~1.5M rec / ~14,597 species | supervised | 32 kHz | [arXiv 2508.04665](https://arxiv.org/abs/2508.04665) |

**Two corrections to the naive "360 h ⇒ 3,089 h is ample" inference:**
1. **From-scratch ≠ continued-pretraining.** AVES's 360 h is **not 360 h from scratch** — it initializes from HuBERT (960 h speech) and *continues* on curated animal audio. If we pretrain **from scratch** (BEST-RQ has no speech warm-start), the relevant budget is closer to the from-scratch rows. **We may warm-start from AVES/wav2vec2 ourselves** — if we do, state it; if not, the 360 h precedent does not apply.
2. **Effective hours ≪ raw hours.** AVES's 360 h are curated *event-bearing* clips. Our ~3,089 h are mostly continuous ocean audio, **overwhelmingly noise/silence after the 8 kHz Nyquist strip** (the plan's own framing: "the masked-prediction target becomes mostly low-freq ambient/ship noise"). **Sufficiency is judged on EFFECTIVE event-bearing hours after VAD/sound-filtering, not raw passive hours.** Estimate effective hours during the §1.0 audit (run the filter at a candidate threshold over a corpus sample; report the event-bearing fraction). This also makes the filter→retrain motivation honest: filtering raises the *signal density* the SSL target sees.

→ With those caveats, **3,089 h (est.) sits between animal2vec (1,068 h) and AVES-all (5,054 h)** and is enough to *attempt and validate a small encoder*. **Provisional target 5,000-10,000 h** (gap ≈ 1,900 h to 5k, ≈ 6,900 h to 10k — both relative to the unaudited 3,089). Revisit once effective hours are known.

### 1.5 Data quality problems (the real blockers)
1. **8 kHz Nyquist ceiling (4 kHz)** destroys odontocete content: orca clicks 20-60 kHz (peaks to ~108 kHz) → 100% gone; whistle peak 6-12 kHz → gone; K-calls keep ~0.5-4 kHz fundamental, lose all harmonics. **Quantified in §1.7.**
2. **Pacific Sound double-decimation** (`tier: 16khz` → 8 kHz): the 256 kHz marquee resolution is unused — contradicts the roadmap's "ablate sample rate" step. **(Repo-verified.)**
3. **Silent data loss + assumed-SR interaction:** `raw_skip_below_sample_rate: true` *skips* natively sub-rate files — **but the skip may test the assumed `assume_sample_rate_hz` (e.g. 48000 for NOAA/Orcasound), not the real header SR.** If so, (a) flipping the flag changes nothing for those sources, and (b) genuinely sub-8 kHz NOAA files (the universe spans 2-500 kHz) could be **passed through and then up-sampled to 8 kHz, injecting empty high-band garbage into the SSL target.** **Before flipping the flag (§1.6), verify in `audio_saver.py` whether `_should_skip_sample_rate` sees the decoded SR or the constant.** True retained hours are unknown until the §1.0 header audit. **Decide explicitly: keep sub-rate sources at native SR (preferred) — do NOT silently upsample them into the 8/16 kHz target.**
4. **SR heterogeneity collapsed** (44.1/48/64/256 kHz → 8 kHz) hides domain shift; mismatches eval targets (Olga 48 kHz, Watkins to 166 kHz).
5. **K-class imbalance 34:1** (K1=2,650 vs K27=78, K14=94). The 97.99% rests on the *balanced* set; treat K14/K27 as low-confidence.
6. **Validation tininess/absence:** field 1.22 h, Watkins ~5 h, Voices 0.47 h — and Watkins/Dominica/Dolph2Vec/Orcasound-labeled **not yet downloaded.** (Drives the Watkins power problem, §6.1.)

### 1.6 Data actions

| Pri | Action | Target |
|---|---|---|
| **P0** | **Manifest+header audit (§1.0):** read real `soundfile.info` per file; replace estimated hours with measured; reconcile 3,359-vs-3,089; emit recording-ID manifest (feeds §6.2.1 leakage audit). | measured hours |
| **P0** | **Train SSL at 16 kHz** (`raw_sample_rate=16000`) — lifts Nyquist 4→8 kHz, recovers whistle/call peak. **Gated on the §1.7 information-loss quantification justifying the re-download cost.** | 16 kHz |
| **P0** | **Verify the skip uses REAL header SR, then decide** `raw_skip_below_sample_rate`. Keep-native vs upsample is an explicit choice, not a silent default (§1.5 #3). | — |
| **P0** | **Diagnose the loss** by re-running a short SSL job on 16 kHz + a 32 kHz high-band subset; compare loss curves *before* buying compute. | — |
| **P0** | **Quantify 8 kHz information loss (§1.7)** — spectral energy >4 kHz on a corpus sample + on K-call discriminative content. | a number |
| **P1** | **Scale toward 5-10k h:** finish SanctSound + remaining ONC; + ~2,000 h Orcasound archive (orca-relevant, 48 kHz, **Glacier-restore flow**); + ~1,000 h NOAA dclde/pifsc/nefsc/afsc. Targets are vs the **unaudited** 3,089. | +1,900→5k; +6,900→10k |
| **P1** | **Native high-band subset** for odontocetes: Pacific 256 kHz tier + ONC native 64 kHz, kept ≥32 kHz (SR ablation + click detection). | few hundred h ≥32 kHz |
| **P1** | **Assemble labeled eval suite NOW:** Watkins ~5 h (verify species count), the ~15 h Sound/Noise set, Olga 1.22 h field; expand tiny sets or add CV. Dominica/Dolph2Vec = stretch. | multi-task |
| **P2** | **Freeze a manifest/file-structure contract** (re: Anvar) — fixed `{sample_rate, segment_duration, mono, dtype}` schema — re-verify before scaling. | — |
| **P2** | **Report per-class** (not just accuracy) for any probe; K14/K27 low-confidence. | — |

### 1.7 ✅ 8 kHz information-loss QUANTIFICATION (justifies the costly 16 kHz move) — FIRST PASS DONE, see §0.5-B (8→16 kHz = +10.2 pts; mean 25.8% energy >4 kHz, K21=52%)
The plan asserts clicks/whistles are lost at 4 kHz Nyquist but never measures it. Turn the qualitative claim into a number **before** committing to the 16 kHz re-download:

1. **Corpus-level spectral audit (sample).** On a stratified random sample (≥1,000 native-SR files across all four sources, before the 8 kHz decimation), compute the **fraction of total spectral energy above 4 kHz** (and the 4-8 kHz band specifically). This bounds what 8 kHz discards corpus-wide.
2. **K-call discriminative-content audit.** On Olga's labeled K-class clips at native 48 kHz, measure (a) per-class energy fraction >4 kHz, and (b) **probe/classifier accuracy delta** when the same clips are low-passed to 4 kHz (8 kHz Nyquist) vs 8 kHz (16 kHz Nyquist) vs native. If K-class separability drops materially when band-limited to 4 kHz, that *quantitatively* justifies 16 kHz for the headline; if not, 8 kHz may suffice for the orca tasks and 16 kHz is justified only by the odontocete-stretch tasks.
3. **Report** the >4 kHz energy fractions and the band-limited accuracy deltas in the paper as the empirical basis for the SR decision (feeds §5 SR ablation and §9 #1).

---

## 2. Where We Are: the 4-Phase Roadmap

The "SSL" and "ROADMAP" decks are **the same program at two zoom levels** — not competing plans. ROADMAP = north star; SSL deck = the concrete data/model/eval spec for **the first box of Phase 1**.

| Phase | Goal | Status | In paper? |
|---|---|---|---|
| **1. Foundation SSL Encoder** | massive data → train SSL variants → **ablate arch×size×SR** → eval transfer (Detection / Classification / **Discovery**) → build sound filter → filter data → **retrain SSL on filtered → compare (with leakage discipline + negative control)** | **CURRENT FOCUS** | **YES — the whole paper** |
| 2. Bioacoustic Tokenizer + LLM | neural codec w/ semantic distillation aligned to Phase-1 encoder; acoustic LLM predicting future tokens | Future | 1 paragraph |
| 3. Unpaired Bioacoustic Translation | denoising seq2seq A↔H, latent alignment + back-translation | Future | No |
| 4. "Doctor Dolittle" | prompt → ControlNet-style generative animal speech + RL loop | Future | No |

**Every later phase consumes the Phase-1 encoder → Phase-1 quality is the program bottleneck.** SSL-deck slides map onto Phase 1: **slide 2** = passive datasets to pretrain on; **slide 5** = model table to ablate; **slides 4 & 7** = the filter's training data + the downstream benchmark.

**What the paper claims (scoped to ~2-3 months) = the Phase-1 inner loop, nothing more:**
1. A **marine-mammal SSL encoder** on ~3,089 h (est., unaudited) real passive audio (→ 5-10k h target).
2. A **reproducible frozen-probe transfer benchmark** (BEANS protocol, [arXiv 2210.12300](https://arxiv.org/abs/2210.12300)) over the **obtainable** slide-7 tasks {1 Filter, 2 Watkins, 4 Orcasound, 5 Olga}; Dominica/Dolph2Vec appendix.
3. **Headline:** a sound/noise **filter-then-retrain** improves downstream transfer vs raw-pretraining — **shown on an eval source DISJOINT from the filter-training corpus, against a same-size random-subset control, with the filter threshold pre-registered** (= the roadmap's "retrain on filtered → compare", done leak-free).
4. A **focused ablation:** architecture × model-size × sample-rate (8-vs-16 guaranteed; 32 kHz preliminary unless the high-band subset lands in time).

---

## 3. The SSL Loss Problem

### 3.1 Framing — why a "big loss" is (probably) NOT a problem
The default is **animal2vec / data2vec-2.0 self-distillation** ([arXiv 2406.01253](https://arxiv.org/abs/2406.01253)): one CNN feature extractor + an **EMA teacher** (full input) + a **student** (masked input). The teacher emits a contextualized latent target; the student **regresses** onto it with **MSE/L2** (data2vec v1 used smooth-L1/Huber, [arXiv 2202.03555](https://arxiv.org/abs/2202.03555)). Targets are **instance-normalized and averaged over top-K teacher layers**; the published guidance is explicit that *"normalizing the targets prevents collapse."* **The MSE is measured against a moving, normalized, data-dependent target → its magnitude is an artifact of that target's scale, not a quality score. "Big" is meaningless.**

> Contrast — objectives where the loss **IS** interpretable:
> - **BEST-RQ** ([arXiv 2202.01855](https://arxiv.org/abs/2202.01855)): frozen random-projection matrix + frozen codebook → discrete pseudo-labels → **cross-entropy**. Loss **starts ≈ `ln(codebook_size)`** (e.g. `ln(8192)≈9.0` nats), should fall and stay **bounded**. Flat-at-`ln(K)` = dead; finite-decreasing = learning. Also **~2.4× faster** than wav2vec2 *on the cited speech setup* (to be re-measured on our config).
> - **Contrastive/InfoNCE** (wav2vec2): chance ≈ `−log(1/(K+1))` → interpretable.

### 3.2 Broken vs fine-but-ugly

| Signal | Broken | Fine-but-ugly |
|---|---|---|
| **Loss TREND** | flat from step 0 / rising / NaN-spikes | decreasing → plateau (plateau can be "big") |
| **Target variance** (Var over feat-dim of normalized teacher targets) | → 0 (**collapse**) | stays > 0 |
| **Grad-norm** (pre-clip) | exploding / NaN | bounded, smooth |
| **Frozen linear probe** every ~500 steps | flat / at chance | **rising** acc/F1 |

**The frozen probe is the most trustworthy "is it learning" signal for latent-regression SSL — but it must not be a single, domain-mismatched, possibly-leaky proxy.** Use **two DISJOINT probes** and require **both** to move:
- **Lab K-class probe:** `new_training_data/` (45,429 WAVs, **K-label in filename prefix** — verified) — a 12-class+noise orca probe. *Caveat:* this is a **narrow single-population lab set**; a good broadband-odontocete encoder could leave it flat (or vice versa). **It cannot be the sole go/no-go.**
- **Watkins species probe:** `confit/wmms-parquet` — a taxonomically broader probe.
- **Leakage guard:** **confirm neither probe's recordings are in the SSL pretraining corpus** (Orcasound is in pretrain — if K-call clips overlap Orcasound recordings, the lab probe is partially in-domain; the §6.2.1 recording-ID audit covers this).
- **Independent collapse guards:** keep **target-variance > 0** and **bounded grad-norm** as separate signals. **Do not declare victory on probe-rise alone** — require *both probes rising AND target-var > 0*.

### 3.3 Ranked failure modes — animal2vec DEFAULT on OUR data
Default recipe: **LR 1e-4**, AdamW wd=0.01, **10k linear warmup + cosine**, 100 epochs, batch 1020 s, 8 kHz, **16 layers/16 heads/dim 1024/315M params**, feat-extractor RF 46 ms→200 Hz.

> ⚠️ **UNVERIFIED hyperparameters (same caveat as the EMA-tau schedule below):** the masking config is often quoted as `mask p=0.15 / span M=2 → ~96% masked, mode 22 ms`. **The "~96%" is arithmetically impossible from those values** — `p=0.15` span-starts with span length 2 gives at most ~30% masked (0.15×2 before overlap). A ~96% masked fraction needs either a much higher start probability (~0.5+) with long/block spans or a different scheme entirely. **Do NOT treat the mask params as "verified from the paper."** Re-derive the masked fraction from the **actual animal2vec config** (`mask_prob`, `mask_length`, span sampling with/without replacement, block-masking) and correct either the percentage or the `(p, M)` values before using failure-mode #5 below.

EMA tau (data2vec v1, **UNVERIFIED for animal2vec**): **0.999→0.9999 over 30,000 steps** — applied as a placeholder; verify against the released config.

| # | Failure mode | Why it bites us | Symptom |
|---|---|---|---|
| 1 | **LR/warmup/EMA-ramp too hot for our scale** | step counts tuned for MeerKAT (1,068 h)+speech; on ~3,089 h if total steps < 30k, EMA never reaches final_tau, targets never stabilize | big/plateaued loss, flat probe |
| 2 | **fp16 instability** | project memory: *"NaN with amp autocast, needed FP32"*, *"GradScaler alone insufficient"* (RTX 5090); latent targets have wide dynamic range | loss spikes → NaN |
| 3 | **Amplitude/normalization across SR→8k** | per-source gain/DC/anti-alias differ (NOAA 48k/ONC 64k/Pacific/Orcasound) | early divergence |
| 4 | **Near-duplicate contiguous 10 s chunks** | long deployments → student ≈ teacher target → **trivial targets → collapse** | low loss BUT target-var→0, flat probe |
| 5 | **Mask ratio/span mismatch** | masking tuned to meerkat call stats; marine calls/clicks differ — **(note: the exact mask fraction is UNVERIFIED, see above)** | weak features |
| 6 | **EMA decay schedule mismatch** | 30k tau-ramp (unverified) wrong for our step budget | collapse / stalled |
| 7 | **Anvar's "wrong file format/structure"** | loader may read wrong dtype/SR/shape/zeros (Anvar says *separate* from loss — must verify) | undiagnosable loss |
| 8 | **315M params on ~3,089 h** | over-capacity → memorize trivial targets | "nice" falling loss, poor transfer |

### 3.4 Debug checklist (ordered, runnable THIS WEEK)
1. **[ ] Overfit one batch** — fixed seed, ONE minibatch, confirm loss → ~0 and a probe on *those clips* → ~100%. **Fails ⇒ bug is in data/format (Anvar) or loss wiring — stop and fix that first.**
2. **[ ] Instrument 3 signals + probe** — per-step loss, grad-norm (pre-clip), teacher target-variance; **frozen linear probe every 500 steps** on `new_training_data/` **and** Watkins. Rising probe + target-var > 0 ⇒ **model is fine, the loss number is a red herring.**
3. **[ ] LR sweep** — drop **3-10×** (try **3e-5, 1e-5**); **lengthen warmup as a fraction of total steps**; recompute cosine horizon **and** the 30k EMA-tau ramp relative to *our* actual step count.
4. **[ ] bf16, not fp16** (never fp32 except a one-off correctness check) — removes loss-scaling fragility; verified supported on this RTX 5090 (`is_bf16_supported()==True`).
5. **[ ] Gradient clip** global-norm **1.0-3.0**; log pre/post-clip to confirm it fires.
6. **[ ] Verify normalization + dedup** — RMS/peak-normalize each clip post-resample; flag silent/NaN/clipped; **shuffle minibatches across recordings** + cap chunks-per-recording-per-batch (kills #4).
7. **[ ] BEST-RQ baseline on the SAME pipeline** — its CE (starts ≈ `ln(8192)=9.0`, decreasing) is an unambiguous oracle: if BEST-RQ learns but animal2vec looks weird ⇒ problem is animal2vec-specific (EMA/norm/LR), not the corpus. (BEST-RQ may also just be the better Phase-1 encoder.)

**Quick wins (likely to fix it fast):** wire the **two** probes + target-variance logger **today** (highest signal-to-effort, ends the panic); **bf16 + grad-clip 1.0** (kills the fp16-NaN class); **LR→3e-5 + longer warmup**; **RMS-normalize + shuffle across recordings**; and **tell the team in writing** that the raw latent-regression loss is **not comparable across runs/objectives** — make the **probe-accuracy-vs-steps curve the official health metric** (and the paper's training-health figure: animal2vec vs BEST-RQ vs Perch-v2). *Health declared only when both disjoint probes rise AND target-var stays > 0 — not on a single lab probe.*

> **Caveats (verified):** the SSL trainer is **NOT in `cetaceans-filtering`** (Perch-embeddings only; all SSL/EMA/mask greps were false positives) — these target the *published animal2vec defaults*; confirm against the actual run's config. animal2vec's exact EMA-tau ramp **and mask params** were not extractable from its text → data2vec-v1's `0.999→0.9999/30k` applied as placeholders; **verify against the released config before quoting.**

---

## 4. Training Speed

> **Diagnose before you optimize.** Run `nvidia-smi dmon` + one `torch.profiler` trace. **GPU-util < 70% ⇒ pipeline-bound → do §4.1 first** (compute tricks barely help a starved model). **> 90% ⇒ compute-bound → §4.2.** Repo evidence (per-file WAV writes, per-step `librosa.resample`) + the fairseq DGX precedent strongly indicate **pipeline-bound** — and this is consistent with **Anvar's "wrong format / wrong file structure"** note.

### Priority table (gains multiply *within* a tier, not across)

| Pri | Lever | Change | Expected gain | Why / cite |
|---|---|---|---|---|
| **P0** | **Data sharding** | Pre-resample once to int16; pack into **WebDataset `.tar`** (~1-2 GB shards) or flat `.bin`+index; mmap/stream sequentially. Kills per-step `librosa.resample` + per-file `open()`. | **~1.5-2.5× alone** (util 40-60%→90%+) | repo writes 1 WAV/chunk (`audio_saver.py: write_wav`); fairseq [#3342](https://github.com/pytorch/fairseq/issues/3342) (8×A100 stuck ~50%, 4 GPUs = same throughput ⇒ CPU-bound), [#3114](https://github.com/pytorch/fairseq/issues/3114). **`webdataset 1.0.2` verified installed in venv** (but NOT in repo deps — must add). |
| **P0** | **BEST-RQ workhorse** | frozen RP-quantizer (`nn.Linear requires_grad=False` + frozen codebook ~8192) + **CE**. Deletes EMA-teacher forward. | **~2.4× on the cited speech setup (re-measure on our config)** | 109 vs 262 GPU-h / 200k steps, 8×V100, 83.0M, **speech** ([arXiv 2405.04296](https://arxiv.org/html/2405.04296v1)). **Loss interpretable:** chance = `ln(8192)=9.0`; log top-1 code acc. **Does NOT simply multiply with §4.1/§4.2 wins.** |
| **P1** | **bf16 autocast** | `torch.autocast('cuda', dtype=bfloat16)`, **no GradScaler** | ~1.3-1.7× + **removes a NaN class** | bf16 = fp32 exponent range ⇒ no loss-scaling; **5090 bf16 verified**; de-risks "VERY BIG loss". |
| **P1** | **torch.compile** | `mode='max-autotune'`, **static** 10 s shape | ~1.2-1.8× (post-warmup) | torch 2.9 verified. Fall back to `default` if sm_120 autotune misfires. |
| **P1** | **SDPA / flash** | route attention through `F.scaled_dot_product_attention` | folded into compile; big at long seq | flash + mem-efficient SDPA enabled. |
| **P1** | **Fused optimizer + grad-accum** | `AdamW(fused=True)`; accumulate to large effective batch | lifts MFU on 24 GB | reaches SSL-typical large batch the 24 GB card can't hold directly. |
| **P1** | **Dataloader knobs** | `num_workers=8-12`, `persistent_workers=True`, `pin_memory=True`, `prefetch_factor=4`, `drop_last=True` | feeds the GPU | 24 cores avail; fixed 10 s ⇒ **no bucketing/ragged collate**. |
| **P1** | **channels_last** | CNN **frontend only** (not the 1-D transformer path) | small, frontend-bound | conv tensor-core util. |
| **P2** | **Shrink the sweep** | ablations at **5 s** seq + small Conformer + 2-CNN mel frontend; full 10 s only for the winner | ~2-4× on attention | seq² attention. |
| **P2** | **Grad checkpointing** | **only if memory-bound** | conditional | trades ~20-30% compute for ~30-40% activation mem; skip if pipeline-bound. |
| **P2** | **5090 alloc fix** | env `PYTORCH_ALLOC_CONF=max_split_size_mb:128` (or `expandable_segments:True`); `empty_cache()+gc.collect()` at epoch bounds | prevents fragmentation crashes | project-verified (`memory/rtx5090_cuda_fix.md`). ⚠ **Keep cuDNN ON / full 24 GB** — do **not** copy `cudnn.enabled=False` / 70%-cap from the *inference* note; those hurt pretraining. |

### §4.1 Pipeline (P0 — the real bottleneck) + disk for BOTH 8 and 16 kHz
`audio_saver.py` calls `write_wav` once per 10 s chunk and resamples with `librosa` on CPU; `manifest_utils.py` emits JSONL of per-file paths. A trainer opening/decoding/(re)resampling **millions of tiny WAVs per step** is CPU-bound. **Action:** one-time job → mono int16 → **WebDataset `.tar`** shards (5-10k clips each) *or* packed `.bin`+index; trainer mmaps/streams. Fixed-length sample tensors ⇒ **pre-stack whole batches in the shard, skip collate**.

**Disk footprint (int16 PCM) — state BOTH rates; the headline is 16 kHz, so size for 16 kHz:**

| Corpus | @ 8 kHz | @ 16 kHz (**headline — size for this**) |
|---|---:|---:|
| ~3,089 h (est.) | **~178 GB** | **~356 GB** |
| ~30,000 h | **~1.7 TB** | **~3.5 TB** |

**Plus headroom for the native ≥32 kHz high-band subset** (a few hundred hours at 32-64 kHz adds tens-to-low-hundreds of GB). **Keep shards on local NVMe**; for the cloud run, size the **instance-local NVMe at the 16 kHz figure + high-band headroom** (so the 30k-h headline needs **~3.5 TB+** node-local, not the 1.7 TB an 8 kHz reading would imply).

### §4.2-4.5 Compute, method, scale, stability
- **Compute:** turn the stack on **together** then re-profile — bf16 (no scaler) + `torch.compile(max-autotune)` static shape + SDPA/flash + `AdamW(fused=True)` + channels_last conv + grad-accum.
- **Method:** BEST-RQ = no teacher, no EMA, no contrastive/diversity loss, frozen quantizer + codebook → cheaper *and* interpretable. Keep one data2vec run as the arch-ablation point + use the small mel/2-CNN frontend for the sweep. **(Fallback rule §5.1 if BEST-RQ underperforms.)**
- **Scale (PROJECTION — gate on measured clips/s from wk1-2):** ~3,089 h ≈ 1.1M ×10 s clips; *projected* post-opt ~150-300 clips/s ⇒ 1 epoch ≈ 1-2 h, a 100-200k-step BEST-RQ schedule ≈ **1-3 GPU-days on the laptop**. 30,000 h ≈ 10× ⇒ **~1.5-3 single-GPU weeks** → move to **4-8× H100 + DDP** (`webdataset split_by_node/worker`). **Stage shards to instance-local NVMe (16 kHz sizing, §4.1)** — don't stream per-step from object storage or you reintroduce §4.1 at cloud scale. **All clips/s and epoch-time numbers are projections that assume the full bf16+compile+SDPA+BEST-RQ stack lands on sm_120; re-measure before quoting.**
- **Cost (RANGE, not a point — gate on profiling; see §9 #5):**
  - *Optimistic:* 8×H100 × ~3 d @ ~$3/h ≈ **~$1.7-2.1k** compute, **assuming** the stack lands and the ~30k h are already on local NVMe.
  - *Pessimistic:* compile disabled / lower MFU / reruns from bf16 instability → **2-3× the compute hours**, i.e. **~$4-6k** compute.
  - *Not yet in either:* **data-acquisition cost** (you have ~3.4k h of the assumed ~30k h), **S3 Glacier-restore charges** for the 14,000 h Orcasound archive, **storage** (3.5 TB+ NVMe/object), and **egress**. Add these explicitly to the budget; the $ ask in §9 #5 is tied to having actually obtained the hours it assumes.
- **Stability:** alloc-conf + periodic `empty_cache()+gc.collect()`; **cuDNN ON, full 24 GB** for training.

**Sources:** [BEST-RQ study](https://arxiv.org/html/2405.04296v1) · [BEST-RQ/Chiu ICML 2022](https://proceedings.mlr.press/v162/chiu22a/chiu22a.pdf) · [animal2vec](https://arxiv.org/abs/2406.01253) · [fairseq #3342](https://github.com/pytorch/fairseq/issues/3342) · [#3114](https://github.com/pytorch/fairseq/issues/3114) · [H100 vs A100](https://www.bestgpusforai.com/gpu-comparison/a100-vs-h100)

---

## 5. SSL Method Recommendation + Ablation Matrix

**All four team pains indict the animal2vec/data2vec2 default:** unbounded regression onto an EMA target → scale-dependent, uninterpretable loss; stability hinges on EMA schedule + target normalization. That **is** "SSL loss is VERY BIG, unclear if it trained."

| Method | SSL family | Loss interpretability | Stability | Compute | Marine-tested | Verdict |
|---|---|---|---|---|---|---|
| **BEST-RQ** (Chiu, ICML 2022) | masked-frame **CE** over **frozen** RP codes | **High** — bounded CE + masked-frame **accuracy** | **High** — no EMA, no trained quantizer, no negatives | **<½ wav2vec2 (speech setup)** | **not yet** | **PRIMARY (with fallback §5.1)** |
| AVES / HuBERT ([2210.14493](https://arxiv.org/abs/2210.14493)) | masked-prediction (k-means targets) | Medium (CE) | Medium | Higher | **YES** (only animal SSL marine-tested) | **SECONDARY + baseline + fallback target** |
| AudioMAE / masked-spectrogram (cf. BirdMAE [2504.12880](https://arxiv.org/html/2504.12880v1)) | masked reconstruction | Medium (MSE) | High | Low | yes (generic) | **3rd arm (1 slot)** |
| animal2vec / data2vec2 ([2406.01253](https://arxiv.org/abs/2406.01253)) | regress to **EMA target** | **Low** — unbounded, scale-dependent | **Fragile** | High | NO | **demote → 1 motivation run** |

**Recommendation: switch PRIMARY to BEST-RQ.** Fixes pain (1) [bounded CE + readable masked-frame accuracy = a real convergence signal], pain (2) [matches wav2vec2 in <½ time *on speech*], removes EMA instability. **~3,089 h (est.) is workable** for a small encoder (AVES-all = 5,054 h; AVES-bio's 360 h is *continued*-pretraining, §1.4). Keep **HuBERT/AVES** as the credibility baseline (**ported weights verified local:** `/mnt/c/Users/Iaroslav/CETACEANS/voxaboxen_weights/aves-base-bio.torchaudio.pt`) and **AudioMAE** as a cheap third arm. Demote data2vec2 to a single "uninterpretable-loss" motivation run.

### 5.1 ⚠️ BEST-RQ FALLBACK rule (decision criterion)
BEST-RQ is **not yet marine-tested** — committing the whole workhorse path to it without an exit is a risk. **Decision rule, evaluated at the wk2-3 smoke-test and again after the first short ablation:**
- **Primary health gate:** BEST-RQ's masked-frame CE must fall from ≈`ln(8192)=9.0` and masked-frame top-1 accuracy must rise (training is happening at all).
- **Transfer gate:** on the two disjoint probes (§3.2) at matched compute, **BEST-RQ's marine-avg frozen-probe transfer must be within noise of (or above) the AVES-bio baseline.** Concretely: if BEST-RQ's marine-avg is **> 1 across-seed std BELOW AVES-bio** after the short schedule, **BEST-RQ does not become the headline encoder.**
- **Fallback ladder if BEST-RQ fails the transfer gate:** (a) try BEST-RQ codebook/RP-dim variants + warm-start from AVES once; if still short, (b) **fall back to AVES/HuBERT-style masked prediction as the workhorse** (keep BEST-RQ as a method-ablation point and for the interpretable-loss figure), and (c) keep data2vec2 only as the motivation run. The **filter→retrain headline is method-agnostic** — it can run on whichever encoder wins the transfer gate, so the headline is not hostage to BEST-RQ.

**Downstream bar:** beat/match **frozen Perch-v2** probes (Perch 2.0 = 101.8 M params, ~1.5 M recordings / ~14,597 species, **no underwater data**, yet transfers to whales — [arXiv 2508.04665](https://arxiv.org/abs/2508.04665)).

### 5.2 Ablation matrix (one-factor-at-a-time around a center cell; ~7-9 runs)
Center = **BEST-RQ / small (~20-30 M) / [SR — see §9]** (cheapest, already-downloaded data).

| Axis | Values | Notes |
|---|---|---|
| **Method** | {**BEST-RQ**★, HuBERT, AudioMAE, data2vec2-baseline} | ★ center |
| **Model size** | {**small ~20-30 M**★, base ~90-100 M} | base only for best method |
| **Sample rate** | {8 k, **16 k**, 32 k} | **8-vs-16 is guaranteed with data in hand**; the **32 k arm requires the native ≥32 kHz subset (built wk3-6, §8) and the file-structure fix** — if it does not land in time, **scope the headline SR result to 8-vs-16 and mark 32 kHz preliminary.** This axis tests the 8 kHz click/whistle ceiling (quantified §1.7). **Novelty framed cautiously:** *we are not aware of a controlled SR ablation on a marine-pretrained SSL encoder* — and we let the **result**, not the literature-gap claim, carry it (bandwidth/SR effects are studied in bird/general bioacoustics; cf. [arXiv 2508.01277](https://arxiv.org/html/2508.01277v1), which we do **not** lean on for a "none-exists" assertion). |

---

## 6. Benchmark Plan for the Paper

> **Thesis:** *at equal frozen linear probe, our marine-pretrained SSL encoder matches/beats AVES & Perch 2.0 on marine tasks; and filtering the corpus then retraining the SSL encoder measurably improves it — shown leak-free (disjoint eval source + recording-level holdout) against a same-size random-subset control, with the filter threshold pre-registered.* We are the **only marine-pretrained SSL encoder** in the grid.

### 6.0 ⚠️ Minimal-viable headline = tasks {1, 2, 4, 5}
The headline rests **only** on data in hand or with a working downloader:
- **Task 1 Filter** (the ~15 h manual set — local/planned),
- **Task 2 Watkins** (`download_watkins.py`, HF `confit/wmms-parquet`),
- **Task 4 Orcasound** (AWS Open Data; labeled subset),
- **Task 5 Olga** (local).

**Dominica sperm-whale clicks (task 3) and Dolph2Vec (task 6) are STRETCH/appendix** — Dominica is login-walled on IEEE DataPort with no open URL or feasible downloader (§7), Dolph2Vec has no open bulk endpoint. The OOD-clicks publishability story (old §6.4) **cannot be a pre-registered headline.** **Decision (§9 #8):** resolve IEEE/DOI access **before wk 5** or drop the OOD-clicks claim entirely; the paper is acceptable on {1,2,4,5}.

### 6.1 Task suite (SSL deck slide 7 → benchmark)

| # | Task | Dataset | Type | Metric(s) | Split | Role |
|---|---|---|---|---|---|---|
| 1 | **Sound Filtering** | ~15 h manual (Watkins master tapes + Orcasound); Sound/Noise/Artifact→Noise (`sed/classifier/labeling.py`, 5 s window, ≥`min_overlap_s`→sound) | binary | **macro-F1** + AUROC | **recording-level** (§6.2.1), test=0.2/val=0.1, seed=42 | **HEADLINE** — also powers filter→retrain. ⚠ **source overlaps eval tasks 2 & 4 — see leakage discipline §6.2.1.** |
| 2 | **Watkins species** | Watkins MMSDB; HF `confit/wmms-parquet` | multiclass (**species count UNVERIFIED — confirm from parquet, ~32 assumed**) | **per-class F1 + macro-F1 + bal-acc**, bootstrap CIs | **stratified CV** (not the single shipped split) for the headline; well-supported subset if many classes <~10 clips | **HEADLINE (power-corrected, §6.2.2)** |
| 3 | **Sperm-whale click/coda** | Dominica (Dtag, 192 kHz) | binary [+opt coda multiclass] | **mAP**/AUROC | file/individual-level | **STRETCH/appendix — OOD, only if access resolved before wk5** |
| 4 | **Orcasound orca call ID** | Orcasound labeled (~1,900 h @44.1k) | multiclass call-type | acc + macro-F1 | **recording/deployment-disjoint** (§6.2.1) | **HEADLINE/Sanity** |
| 5 | **Olga K-class orca** | `new_training_data/` (12 K + noise) | multiclass (12) | acc + macro-F1 | existing train/val/test | **HEADLINE/Sanity** (topline **ConvNeXt V2 Pico 97.99%**) |
| 6 | **Dolph2Vec dolphin whistles** | [OpenReview](https://openreview.net/forum?id=QGAFX5kcR5) | multiclass + detection | macro-F1/mAP | per-source | **STRETCH/appendix (if obtainable)** |

### 6.2 Protocol — frozen probe PRIMARY, fine-tune as upper bound
Mirrors Perch 2.0 ("strongest linear probing on BEANS … *without any fine-tuning*"), AVES (best-layer linear probing), and the Perch-2.0 underwater paper (few-shot logistic-regression, ROC-AUC). Reuse the **harness the repo already ships**:
1. **Embedding extraction** — fixed windows (Perch 5 s; AVES/ours native frame); swap encoders via `filtering/embed/perch_v2_embed.py` (Scenario 3 = `model_name=surfperch`); probe is dimension-agnostic (Perch=1280-d, AVES/ours=768-d).
2. **PRIMARY probe — linear** — `StandardScaler → LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)` (**verified verbatim** in `filtering/watkins/classifier/models.py`). Mean±std over ≥3 seeds.
3. **Aggregation** — per-file = **mean** over window probs (primary); **max** as ablation (`metrics.py: aggregate_file_predictions`). *Note: only `{mean, max}` are shipped — "attentive pool" is NOT in the repo; treat as optional future work, not a claim.*
4. **Few-shot curve** — k∈{4,8,16,32}, 5 runs, headline tasks (skip k=32 for tiny Olga K27=78/K14=94), per [2512.03219](https://arxiv.org/html/2512.03219).
5. **DISCOVERY probe** — k-NN (cosine, k∈{5,20}) + cluster-purity/silhouette → fills the roadmap's *Discovery* leg.
6. **Upper bound (appendix only)** — fine-tune end-to-end; report Δ over frozen probe. **Do not headline fine-tune** (fits "training is slow" + keeps the comparison fair).

#### 6.2.1 ⚠️ Recording-level split + cross-set disjointness audit (GATING step, not a footnote)
With long continuous deployments chunked into 10 s clips, **clip-level shuffling = near-certain leakage** (adjacent clips from one recording in both train and test). This is the **single most important integrity control for the headline**, and it interacts with the filter→retrain circularity:
- **Recording-ID manifest.** From the §1.0 audit, every clip carries a `recording_id` / `deployment_id`. **All splits (train/val/test) are formed at the recording level** — no recording's clips appear in two splits, for **every** task.
- **Cross-set disjointness (the leakage audit).** Build a manifest proving **pairwise-disjoint recording IDs across {SSL pretrain corpus, filter-training set, each eval task's splits}.** Concretely:
  - The **Watkins** and **Orcasound** recordings used in **eval tasks 2 and 4** are **held OUT of (a) the filter-training set AND (b) the SSL pretraining corpus**, at the recording level. (Orcasound is currently in *both* pretrain and an eval task — this must be split.)
  - The filter (trained on Watkins+Orcasound) **must not have seen any recording used to evaluate tasks 2/4**, or "v1 > v0" on those tasks can be an artifact of the filter selecting in-domain-looking audio rather than genuine representation gain.
- **Report the filter→retrain delta primarily on a DISJOINT source.** The headline filter→retrain Δ is reported on a **task whose source is disjoint from the filter-training corpus** — e.g. a **NOAA/SanctSound-derived detection task** (filter trained on Watkins+Orcasound, evaluated on SanctSound) — **not** primarily on Watkins/Orcasound. (Watkins/Orcasound deltas may be reported secondarily, *with the leakage caveat stated*.)
- **Gating:** this audit is a **wk1-2 deliverable that BLOCKS the headline** (§8). No filter→retrain number is reported until the disjointness manifest passes.

#### 6.2.2 ⚠️ Watkins statistical power (the headline is otherwise underpowered)
Watkins MMSDB is **~5 h over ~32 species (~9 min/species)** — several classes will have a handful of test clips, so a 32-way macro-F1 has huge variance and "beat AVES/Perch at equal probe" may not be significant. Mandatory:
- **Verify the species count** against `confit/wmms-parquet` (do **not** assume ~32).
- Report **per-class support and per-class F1**, not just macro.
- Use **stratified CV** for the headline, not the single shipped split.
- **Bootstrap CIs** on macro-F1; apply the same **≥3-seed + CI** discipline as Table A.
- If many classes have support **<~10**, **restrict the headline to the well-supported species subset** and move full-32 to the appendix.

#### 6.2.3 ⚠️ Filter→retrain: significance, negative control, threshold pre-registration
The central claim is "v1 > v0", but the **SSL pretraining is currently single-run per arm (n=1)** while the plan demands ≥3 seeds for the *probe*. A small marine-avg delta then has no error bars. Required before this can headline:
- **Minimum effect size + replication.** State a **minimum marine-avg Δ** that counts as a result (pre-registered). **Replicate the PRETRAINING across ≥2 seeds/data-subsets per arm** so the Δ has error bars; **if compute forbids, label the delta honestly as anecdotal / n=1-per-arm** in the paper.
- **NEGATIVE CONTROL (attribute the gain to filtering, not to a smaller/cleaner/2nd-pass corpus).** Add a third arm: **SSL-v_rand = retrain on a RANDOM subset of the SAME size as the filtered corpus.** The claim holds only if **v1 (filtered) > v_rand (random same-size)**, not merely v1 > v0. This rules out "the gain is just from a smaller/cleaner corpus or a second training pass."
- **Filter-THRESHOLD pre-registration + robustness sweep.** The % of corpus removed is a **free hyperparameter** that directly determines v1 (researcher degrees of freedom). **Pre-register the threshold** (sound-recall vs % kept operating point), and **sweep it** (e.g. remove {10%, 25%, 40%}) — **v1 > v0 (and > v_rand) must hold across thresholds**, or the result is not robust.
- **Effective-hours framing.** Report the **event-bearing fraction** before/after filtering (§1.4) so the mechanism (filtering raises signal density seen by the SSL target) is explicit.

### 6.3 Baseline grid (all through the IDENTICAL frozen probe)

| Encoder | SSL? | Pretrain hrs | From-scratch vs warm-started | Native SR | Marine-tested? | Role |
|---|---|---|---|---|---|---|
| **Ours — marine SSL** | ✅ | ~3,089 h est. (→5-10k) | from scratch (unless we warm-start — state it) | 8→16 kHz | **this paper** | candidate |
| AVES-bio | ✅ | 360 h | **continued from HuBERT (960 h speech)** | 16 kHz | ✅ | SSL marine baseline |
| BirdAVES | ✅ | AVES-bio + bird-heavy | continued | 16 kHz | ✅ (bird-biased) | SSL baseline |
| Perch 2.0 | ❌ (sup.) | ~14,597 species | supervised | 32 kHz | ✅ (strong) | **toughest baseline** |
| BEATs | ✅ | AudioSet-2M (~5,800 h) | from scratch on AudioSet | 16 kHz | generic | SSL generic baseline |
| wav2vec2-base | ✅ | 960 h LibriSpeech | **from scratch on speech** | 16 kHz | generic | SSL speech baseline |
| animal2vec | ✅ | 1,068 h MeerKAT | from scratch | 8 kHz | ❌ | SSL animal, NOT-marine |
| log-mel / handcrafted | — | none | — | — | — | **floor** |

### 6.4 Why it's publishable
1. **Beat AVES/Perch 2.0 on marine tasks at equal frozen probe.** The Perch-2.0 underwater paper already publishes the bar in our exact protocol (DCLDE killer whale **0.977** ROC-AUC @k=16; NOAA PIPAN baleen **0.924**).
2. **Filter→retrain→improves, shown LEAK-FREE** (the roadmap's core loop, novel): SSL-v0 (raw) → train filter [task 1] → filter corpus → SSL-v1 (cleaned) → re-probe → **v1 > v0 on a DISJOINT eval source, > v_rand (same-size random control), robust across pre-registered thresholds** (§6.2.1, §6.2.3). No competitor runs a self-filtering retrain loop. Converts "loss is huge, did it train?" into a controlled result — *only* because the leakage/circularity is handled.
3. **Sample-rate ablation (8/16[/32] kHz)** — plausibly a dominant lever (moans→broadband clicks), **grounded in the §1.7 information-loss numbers**; 32 kHz arm preliminary unless the high-band subset lands.
4. **Model-size + architecture ablation** — report probe accuracy **and** pretraining wall-clock → evidence for "is it training well / can we speed it up."

### 6.5 Results skeletons (fill-ready)
**Table A — Headline frozen linear probe** (mean±std, ≥3 seeds; **bold** only if Δ > across-seed std): rows = {Ours, AVES-bio, BirdAVES, Perch 2.0, BEATs, wav2vec2, animal2vec, log-mel floor, *ConvNeXt V2 Pico topline (Olga only = 97.99)*}; cols = (1) Filter F1 | (1) AUROC | (2) Watkins **per-class + macro** F1 (w/ CIs) | (4) Orca acc | (5) Olga acc | **Marine avg (headline tasks {1,2,4,5})**. *Dominica/Dolph2Vec columns appendix-only if obtained.*
**Table B — Filter→retrain ablation** (our encoder): **SSL-v0 (raw) vs SSL-v1 (filtered) vs SSL-v_rand (random same-size control)**; primary col = **disjoint-source detection Δ (e.g. SanctSound)**; secondary cols (Watkins/Orca, *leakage-caveated*); rows per **threshold {10/25/40%}**; **Δ vs v0 AND Δ vs v_rand**, with seeds/CIs or an explicit "n=1-per-arm, anecdotal" label.
**Table C — SR/size/arch ablations** (marine avg + k-NN purity): Ours @8/16[/32] kHz, Ours BEST-RQ vs fallback, Ours small/large, with **pretrain wall-clock** and the **§1.7 >4 kHz energy fractions**.
**Table D — Discovery** (k-NN acc @k=20, silhouette, cluster-purity) per encoder on tasks 2 & 5.

### 6.6 Decision rules (pre-registered)
Frozen probe = **primary** everywhere; fine-tune appendix-only. ≥3 probe seeds (5 for few-shot). Bold a win only if Δ > across-seed std. **Native-SR per model = primary; common-8 kHz = controlled ablation (stated).** **All splits RECORDING/deployment-disjoint with a cross-set disjointness manifest (§6.2.1) — gating, not optional.** Watkins uses stratified CV + per-class support + CIs (§6.2.2). Filter→retrain reported with a **same-size random-subset negative control + pre-registered threshold sweep + a disjoint eval source** (§6.2.3). test=0.2/val=0.1/seed=42. Keep an honest **"where we lose"** subsection.

#### 6.6.1 ⚠️ Environmental/temporal confound note (recording-disjoint is necessary but not sufficient)
Marine passive audio has strong **site / season / diel** structure. **Recording-level disjointness does not remove same-site-different-time leakage** — a test recording from the same hydrophone/season as a train recording shares acoustic conditions (channel, ambient, soundscape), inflating transfer. **Where feasible, also split by SITE and/or TIME (season/diel)** for the headline (or report a site-held-out variant), and **note this confound explicitly** in limitations. This applies to both the eval splits and the filter→retrain (don't let the filter learn a site rather than "sound vs noise").

#### 6.6.2 ⚠️ Dataset LICENSING / usage note (for the benchmark + paper release)
Before publishing results or releasing a benchmark, **document the license/usage terms** for each dataset we redistribute or report on: **Watkins MMSDB, Orcasound, NOAA SanctSound, ONC, Pacific Sound** (and Dominica/Dolph2Vec if used). Confirm redistribution/derived-results rights and required attributions; this gates the "we will publish a benchmark" claim and any released splits/manifests.

**Sources:** [Perch 2.0](https://arxiv.org/abs/2508.04665) · [Perch 2.0 underwater](https://arxiv.org/html/2512.03219) · [AVES](https://arxiv.org/abs/2210.14493) · [BEANS](https://arxiv.org/abs/2210.12300) · [NatureLM/BEANS-Zero](https://arxiv.org/abs/2411.07186) · [BirdSet](https://arxiv.org/html/2403.10380v3) · [WhaleNet](https://arxiv.org/abs/2402.17775) · [Dolph2Vec](https://openreview.net/forum?id=QGAFX5kcR5)

---

## 7. cetaceans-filtering PR Proposal

### Decision
Ship **candidate (a), scoped to NOAA only**: enable the 4 documented NOAA SSL deployment families in `data_loading.yaml`. Highest-value, lowest-risk. **Candidate (b) (new downloader) is deferred** — every missing plan dataset turned out infeasible/too-large for a small PR (below).

> ⚠️ **STATUS / CONTRADICTION TO RESOLVE (verified):** the diff is **ALREADY applied to the working tree** — `data_loading.yaml` shows `dclde/ pifsc/ nefsc/ afsc/` uncommented and is `M` (modified) on branch **`main`**, but the branch `data/enable-noaa-ssl-prefixes` **does NOT exist** and nothing is committed. **Action:** create the branch, move the uncommitted edit onto it, commit, push, open the draft PR. Do not commit straight to `main`.

### Why NOAA-prefix-enable wins
- **Config-only, reuses tested code** — no new script/dep/token; drives the working `download_noaa_onms.py`.
- **Directly answers "is it enough data?"** — today only a ~30-deployment SanctSound sliver is active; the documented **~773,000 h** NOAA universe (`dclde/pifsc/nefsc/afsc/`) was unreachable from config.
- **Verified safe** against live `noaa-passive-bioacoustic`: families resolve; real 48 kHz+ FLAC above the 8 kHz floor (`pifsc/audio/pipan_10/.../*.flac` ~11.5 MB; `afsc/audio/ga13/.../*.flac` 21-46 MB; `nefsc/audio/monh/.../*.flac` 333-376 MB). Downloader lists **recursively (no delimiter), filters by audio extension**, so a bare family prefix is valid; `products/` siblings dropped. Each prefix = **one deployment** → output bounded by `max_files_per_deployment: 20` / `hours_per_deployment`.

### What the diff does (`+16 / −4`, one block, one file; active prefixes 20 → 24)
- Uncomment `dclde/ pifsc/ nefsc/ afsc/`; add a comment block on per-deployment sampling semantics + CLI override + 8 kHz floor caveat.
- **Leave SanctSound per-site entries untouched; keep `# - sanctsound/` commented** (avoids double-listing) — no regression. ONC unchanged.

### Missing datasets (and why each is NOT this PR)
- **Orcasound archive (~14,000 h @48k FLAC)** — biggest open SSL corpus after NOAA/ONC; genuinely uncovered (repo's `download_orcasound.py` targets `acoustic-sandbox`, archive lives in **`archive-orcasound-net`**). **Blocker:** 100% sampled objects are **S3 GLACIER** → `InvalidObjectState`; needs async restore-then-poll (**+restore/egress cost, §4.5**). **Strongest follow-up.**
- **Dominica (~17.8 GB, 192 kHz)** — IEEE DataPort, login-walled, no open URL. **Stretch/appendix only (§6.0).**
- **Dolph2Vec (~180k whistles)** — no open bulk endpoint; validation target, not pretraining. **Stretch/appendix only.**
- **ONC wildcard families** (`PVIPH.*`, `SGC.*`, …) — wildcard child-location groups; `download_onc_hydrophones.py` passes `locationCode` verbatim with no `includeChildren` expansion + needs date window + `ONC_TOKEN`. Separate PR.
- (Watkins & Voices already have downloaders.)

### Branch + draft PR
- **Branch:** `data/enable-noaa-ssl-prefixes`
- **Title:** `data(noaa): enable dclde/pifsc/nefsc/afsc SSL deployment prefixes for SSL pretraining corpus`
- **Smoke test:** `uv run python utils/datasets_downloads/download_noaa_onms.py data_loading.sources.noaa.only_new_files=true data_loading.sources.noaa.max_files_per_deployment=1 data_loading.raw_segment_duration=-1` → expect ~1 file/family into `data/noaa_onms/<family>/.../audio/` + `manifest.jsonl`.
- **File touched (in clone):** `/tmp/cetaceans-filtering/configs/data_loading/data_loading.yaml` (edit applied locally, **not committed/pushed**).

> **Coupling note:** if §1's P0 (16 kHz + the verified `raw_skip_below_sample_rate` decision) is adopted, fold those config flips into this same PR (they live in the same file) — but flag for the team since 16 kHz triggers a full re-download, **and only after §1.5 #3 confirms the skip tests the real header SR** (don't ship a flag flip that silently upsamples sub-rate sources).

---

## 8. Sequenced Next Steps (4-8 weeks)

Owners: **Iaroslav** = SHAP [done] / speed / PR / data · **Anvar / @kekstroke** = dataset format fix + SSL retrain · **@board_and_sword** = paper draft + other architectures (Perch2 / AnimalSpot / animal2vec). `‖` = runs in parallel.

| Wk | Serial (critical path) | ‖ Parallel | Owner |
|---|---|---|---|
| **1** | **(1) Diagnose the loss, don't buy compute:** wire **two** frozen probes (`new_training_data/` **and** Watkins) + 3-signal logger (loss-trend / grad-norm pre-clip / target-variance); **overfit-one-batch** test. Declare loss fine **in writing only if BOTH probes rise AND target-var > 0.** | **(2) Fix Anvar's file-structure bug**; re-validate dataloader output (shape/dtype/SR/count). **(3)** Train the **filter** on the 15 h set (`sed/train_classifier.py`). | (1) Iaroslav ‖ (2) Anvar ‖ (3) @board_and_sword |
| **1** | **(4) Land the NOAA PR:** branch `data/enable-noaa-ssl-prefixes`, commit the staged edit, push, open draft PR; smoke-test. | **(5)** **§1.0 manifest+header audit** (real `soundfile.info` per file): measured hours, reconcile 3,359-vs-3,089, emit **recording-ID manifest**. Verify the skip uses real header SR; then decide SR (8 vs 16) + keep-native-vs-upsample. | (4) Iaroslav ‖ (5) Iaroslav+Anvar |
| **1-2** | **(6) GATING leakage audit (§6.2.1):** build cross-set recording-ID disjointness manifest across {pretrain, filter-train, each eval task}; hold Watkins/Orcasound eval recordings OUT of pretrain + filter-train. **No headline number until this passes.** | **(6b) §1.7 information-loss quantification** (>4 kHz energy on corpus sample + K-call band-limited accuracy). **(7)** Embed full corpus with Perch-v2 (filter step B). | (6/6b) Iaroslav+Anvar ‖ (7) @board_and_sword |
| **1-2** | **(8) Re-shard data → WebDataset/int16** (add `webdataset` to repo deps); **profile** (`nvidia-smi dmon`+profiler) to confirm pipeline-bound + **measure clips/s** (gates §4.5 cost); stack **bf16 + compile + SDPA + fused AdamW**. | **(8b)** Confirm **dataset licenses** (§6.6.2) for the benchmark release. | (8) Iaroslav ‖ (8b) @board_and_sword |
| **2-3** | **(9) Stand up BEST-RQ trainer;** smoke-test on a subset; **confirm masked-frame accuracy rises** (loss starts ≈ `ln(8192)=9.0`). **Evaluate the §5.1 fallback gate** vs AVES-bio. | **(10)** Build **filtered corpus** (step C); **pre-register the threshold + set up the sweep {10/25/40%}** and the **random same-size control corpus** (§6.2.3). **(11)** Build **BEANS-style frozen-probe harness** over headline tasks {1,2,4,5} + Perch-v2 & BEATs baselines + **probe local AVES** (`voxaboxen_weights/aves-base-bio.torchaudio.pt`). | (9) Anvar/Iaroslav ‖ (10) @board_and_sword ‖ (11) @board_and_sword |
| **3-6** | **(12) Run the ablation matrix** (method × size × SR, short schedules; 8-vs-16 guaranteed, 32k if high-band subset ready); rank by *relative* probe quality + wall-clock. | **(13)** Acquire labeled eval suite (Watkins — **verify species count + per-class support**; Orcasound labeled). **Dominica/Dolph2Vec only if access resolved before wk5 (§6.0).** **(14)** Build native **≥32 kHz high-band subset** (Pacific 256k + ONC 64k). | (12) Anvar+Iaroslav ‖ (13) @board_and_sword ‖ (14) Iaroslav |
| **5-7** | **(15) Filter-vs-raw-vs-random A/B** (BEST-RQ-or-fallback, identical config) → **headline delta on a DISJOINT source, vs v_rand, across the threshold sweep, with seeds/CIs (or labeled n=1-per-arm).** **(16)** Long-run the single best config (rent 4-8× H100 if scaling to ~30k h — **budget gated on measured clips/s + acquired hours + restore/egress, §4.5/§9 #5**). | **(17)** Figures + repro scripts; training-health figure (animal2vec vs BEST-RQ vs Perch-v2). | (15-16) Anvar+Iaroslav ‖ (17) @board_and_sword |
| **6-8** | **(18) Write the paper** (encoder + leak-free benchmark + filter-then-retrain-with-control + ablation; Phases 2-4 = 1 future-work paragraph). | — | @board_and_sword + all |

---

## 9. Open Decisions for the Team

1. **Sample rate for the headline run — 8 vs 16 kHz?** *(Experts split.)* Data/Loss/Speed → **16 kHz P0** (recovers whistle/call peak, matches AVES/wav2vec2). Roadmap kept **8 kHz center cell** (cheapest, already-downloaded). **16 kHz forces a full re-download** and **doubles disk** (~178→356 GB @3,089 h; ~1.7→3.5 TB @30k h, §4.1). **✅ RESOLVED by §0.5-B (measured this session): 8→16 kHz = +10.2 pts accuracy / +0.16 macro-F1 on the K-call tasks → adopt 16 kHz headline + 8/16/32 kHz ablation.** Remaining human call: eat the full re-download *now*, or stage it after the §1.0 manifest audit? (16 kHz still doubles disk — §4.1.)
2. **Is the "VERY BIG loss" actually a problem?** All experts agree the *number* is uninterpretable for animal2vec. **Gate:** run **two** probes + overfit-one-batch this week. **Declare fine only if BOTH probes rise AND target-var > 0** — not on a single lab probe. Confirm everyone accepts the (two-probe) curve as the health metric.
3. **PR hygiene:** the NOAA edit is **uncommitted on `main`** with no feature branch. Confirm we branch it before committing — and whether to **fold the 16 kHz + the (verified) `raw_skip_below_sample_rate` decision into the same PR** (same file) or keep separate. **Do not ship a flag flip that silently upsamples sub-rate sources (§1.5 #3).**
4. **Primary SSL method — commit to BEST-RQ as workhorse?** Strong consensus (interpretable loss, ~2.4× faster *on speech*, no EMA), **but with the §5.1 fallback rule** (if BEST-RQ's marine transfer is >1 std below AVES-bio at matched compute, fall back to AVES/HuBERT as workhorse). Confirms demoting data2vec2 to a single motivation run. Anvar currently owns "SSL retrain on the default" — switch to BEST-RQ or keep the animal2vec run for the motivation figure?
5. **Compute budget for the big run:** approve a **RANGE, not a point** — optimistic **~$1.7-2.1k** (8×H100×3 d) vs pessimistic **~$4-6k** (compile off / lower MFU / reruns), **plus** data-acquisition + Glacier-restore + storage (3.5 TB+) + egress. **Gate the ask on measured clips/s (wk1-2 profiling) and on having actually obtained the ~30k h it assumes** (you have ~3.4k). Or stay on the laptop (~1-3 GPU-days at the unaudited 3,089 h).
6. **Target unlabeled hours — 5,000 or 10,000 h?** Gaps (+1,900 / +6,900) are vs the **unaudited** 3,089 — **re-decide after the §1.0 audit and the effective-event-bearing-hours estimate.** Drives Orcasound-archive / NOAA pulling and whether we tackle the **Glacier-restore** flow for the 14,000 h archive.
7. **Manifest/file-structure contract:** Anvar flagged a "wrong format/structure" (says *separate* from the loss). Who owns freezing the `{sample_rate, segment_duration, mono, dtype}` schema, and do we block scaling until the bigger dataset re-validates?
8. **Headline scope on obtainable data only:** the minimal-viable headline = **tasks {1 Filter, 2 Watkins, 4 Orcasound, 5 Olga}**. **Dominica (IEEE login-wall) and Dolph2Vec (no open endpoint) are stretch/appendix.** **Decision:** resolve IEEE/DOI access **before wk5** or **drop the OOD-clicks claim**; confirm the paper is acceptable on {1,2,4,5} (it is).
9. **Aggregation claim:** repo ships only `{mean, max}` (verified); "attentive pool" is not in the repo. Confirm we drop attentive-pool from the protocol (or scope it as optional engineering).
10. **Filter→retrain integrity (NEW — blocks the headline):** confirm the team will (a) run the **recording-level cross-set disjointness audit (§6.2.1)** as a wk1-2 gate, (b) report the Δ on a **disjoint source** (SanctSound-derived) not primarily Watkins/Orcasound, (c) include the **same-size random-subset negative control** and the **pre-registered threshold sweep**, and (d) either **replicate pretraining across ≥2 seeds/subsets** or **label the delta n=1-per-arm/anecdotal.** Without these the central result is not publishable.

---

## 10. Review notes — corrections applied

This section records what changed from the draft after the adversarial review, so the team can see the diff in reasoning. (The draft's strongest parts — probe-as-health-metric, the §3.4 debug checklist, the "diagnose before optimize" pipeline analysis, the quality-over-scale thesis with the repo-verified double-decimation + silent-skip findings, the BEST-RQ pivot, Phase-1-only scoping, and the minimal config-only PR — are **preserved**.)

**HIGH severity:**
- **Data leakage / circularity in the filter→retrain headline (most important):** added a **GATING recording-level + cross-set disjointness audit (§6.2.1)** — Watkins/Orcasound eval recordings held out of **both** pretrain and filter-train; the filter→retrain Δ is now reported **primarily on a DISJOINT source (NOAA/SanctSound-derived)**, not on Watkins/Orcasound (which are both filter-train inputs and eval tasks); recording-ID manifest proving pairwise disjointness across {pretrain, filter-train, each eval task} is a wk1-2 blocker. Made prominent in §0, §2, §6.0, §6.2.1, §6.4, §9 #10.
- **Headline on un-obtainable data:** **demoted Dominica sperm-whale clicks and Dolph2Vec to explicit STRETCH/appendix** (§1.3, §6.0, §6.1, §7); stated the **minimal-viable headline = tasks {1 Filter, 2 Watkins, 4 Orcasound, 5 Olga}** (§0, §6.0, §9 #8); access must resolve before wk5 or the OOD-clicks claim is dropped.
- **Arithmetic:** added **§1.0** — the §1.1 table **sums to ~3,359 h, not 3,089 h** (reconciled/flagged, not hidden); **all** hour counts relabeled **"estimated, unaudited (±)"** because they derive from `assume_sample_rate_hz` / `assume_minutes_per_file`; the **manifest+header audit is now P0** (read real `soundfile.info`) and replaces estimates before any number is quoted.
- **Disk math:** stated disk for **BOTH 8 and 16 kHz** (3,089 h ≈178 GB@8k / ≈356 GB@16k; 30k h ≈1.7 TB@8k / ≈3.5 TB@16k) and **sized NVMe/H100 for the 16 kHz headline + high-band headroom** (§4.1, §4.5, §9 #1/#5).

**MEDIUM severity:**
- Removed the impossible **"~96% masked from p=0.15/span=2"**; marked animal2vec **mask params UNVERIFIED** (same caveat as the EMA-tau schedule) and required re-derivation from the actual config (§3.3).
- **AVES-360h sufficiency overclaim:** distinguished **from-scratch vs CONTINUED-pretraining** hours (AVES/wav2vec2 marked warm-started) and based sufficiency on **EFFECTIVE event-bearing hours after filtering**, not raw passive hours (§1.4, §5, §6.3).
- **Watkins underpowered headline:** now requires **per-class support + per-class F1, stratified CV, bootstrap CIs**, well-supported-subset restriction if support <~10, and **verification of the species count against `confit/wmms-parquet`** rather than assuming ~32 (§6.1, §6.2.2).
- **`raw_skip_below_sample_rate` vs `assume_sample_rate_hz`:** before flipping the flag, **verify the skip tests the REAL decoded header SR**; the audit reads real headers per file; **explicit keep-native vs upsample decision** (no silent upsampling of sub-rate sources) (§1.5 #3, §1.6, §7).
- **Health-probe over-trust:** now requires **≥2 DISJOINT probes (lab K-class AND Watkins)**, both must move, and confirmation that neither probe's recordings are in the SSL pretraining corpus; target-var/grad-norm kept as independent collapse guards (§3.2, §3.4, §8 wk1, §9 #2).

**LOW severity (softened, not deleted):**
- One BEST-RQ speedup figure: **"~2.4× at iso-quality on the cited speech setup, to be re-measured on our config"**; stopped quoting 2.5× (§0, §3.1, §4 table, §5).
- Softened the SR-ablation novelty to **"we are not aware of a controlled SR ablation on a marine-pretrained SSL encoder,"** letting the result carry it; guaranteed only 8-vs-16, marked 32 kHz preliminary unless the high-band subset lands (§5.2, §6.4).
- **$ cost gated on MEASURED clips/s** from wk1-2 profiling; added a **pessimistic case** (compile disabled / lower MFU / reruns) and **data-acquisition / Glacier-restore / storage / egress** costs (§4.5, §9 #5).

**MISSING items added:**
- **Recording-level split + cross-set disjointness audit** as a GATING step (§6.2.1).
- **8 kHz information-loss quantification** (>4 kHz spectral energy on a corpus sample + K-call band-limited accuracy) to justify 16 kHz with numbers (§1.7).
- **Statistical-significance plan for v1>v0:** stated minimum effect size; replicate **pretraining across ≥2 seeds/subsets** or **label the delta anecdotal/n=1-per-arm** (§6.2.3).
- **Filter-threshold pre-registration + robustness sweep** (v1>v0 must hold across {10/25/40%} removed) (§6.2.3).
- **NEGATIVE CONTROL:** **random same-size subset** arm (v1 must beat v_rand, not just v0) to attribute the gain to filtering, not to a smaller/cleaner/2nd-pass corpus (§6.2.3).
- **Dataset licensing/usage note** (Watkins, Orcasound, SanctSound, ONC, Pacific Sound) for the benchmark/paper release (§6.6.2, §8 wk1-2).
- **BEST-RQ fallback rule** (decision criterion vs AVES baseline on marine transfer) (§5.1, §9 #4).
- **Environmental/temporal (site/season/diel) confound note** — recording-disjoint is necessary but not sufficient (§6.6.1).

**Added up front:** a **"⚠️ Confidence & unverified numbers" callout** marking every estimate pending the manifest/header audit.
