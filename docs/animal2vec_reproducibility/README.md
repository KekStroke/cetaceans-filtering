# animal2vec reproducibility record

This is the compact, evidence-backed record for the cetacean animal2vec
pretraining runs. It was reconstructed from the retained Fairseq TSV
manifests, resolved Hydra configs, JSON training logs, and the final 16 kHz
checkpoint. Large checkpoints, TensorBoard event files, and generated
validation reports are deliberately not committed.

Machine-readable corpus facts and hashes are in
[`corpora.yaml`](corpora.yaml). The paths in that file are logical identifiers
to be resolved below a local corpus root; they are not ML Space paths.

## Conclusions

- The original 8 kHz corpus contains exactly **3,089.756791042 hours**. The
  competing 3,359-hour figure is not supported by its retained
  `pretrain.tsv`.
- The 16 kHz r2 corpus contains **3,542.255555556 hours**.
- A final 32 kHz corpus does exist. It contains **3,370.945833333 hours** and
  was assembled from 2 through 4 July 2026.
- The retained 8, 16, and 32 kHz corpora are not a controlled sample-rate
  ablation: their source composition and clip duration differ. In particular,
  Orcasound contributes 576.666 hours to 8 kHz and 1,047.507 hours to 16 kHz.
- The 32 kHz corpus is the usable common parent for a future controlled
  ablation: downsample the same accepted rows to paired 16 and 8 kHz variants.
  It is normalized 32 kHz audio, not an archive of the original native-rate
  downloads.

## Corpus inventory

The row count excludes the Fairseq root line. Hours are computed from the
sample counts in `pretrain.tsv`, not from directory sizes or rounded clip
counts.

| Corpus | Rate / clip policy | Rows | Hours | Raw `pretrain.tsv` SHA-256 |
| --- | --- | ---: | ---: | --- |
| 8 kHz, 10 s maximum | 8,000 Hz; variable length up to 80,000 samples | 1,146,279 | 3,089.756791042 | `b850db76f42dd672bbbab2f640b869f2750ab09ae8b877f1d06501dbc3500a4c` |
| 16 kHz r2, 5 s | 16,000 Hz; exactly 80,000 samples | 2,550,424 | 3,542.255555556 | `f8e0a73ade5ebc9ee57537281cdfc1621fa7d85cadc51472028d76e7b6f3a3b0` |
| 32 kHz final, 2.5 s | 32,000 Hz; exactly 80,000 samples | 4,854,162 | 3,370.945833333 | `9e85eb4282aa4130820b854723b18f618b1d6a7aad7057846f696d521ad40d65` |

All three retained `valid_0.tsv` files are empty. Their raw SHA-256 is the
standard empty-file digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The hashes above are over the raw, decompressed TSV bytes. Hashes of the
distributed gzip files are also recorded in `corpora.yaml`.

### Source composition

| Corpus | Source | Rows | Hours |
| --- | --- | ---: | ---: |
| 8 kHz | NOAA ONMS | 642,734 | 1,784.840680868 |
| 8 kHz | ONC | 106,651 | 296.250069444 |
| 8 kHz | Orcasound | 241,374 | 576.666040729 |
| 8 kHz | Pacific Sound | 155,520 | 432.000000000 |
| 16 kHz r2 | NOAA ONMS | 1,284,899 | 1,784.581944444 |
| 16 kHz r2 | ONC | 205,260 | 285.083333333 |
| 16 kHz r2 | Orcasound | 754,205 | 1,047.506944444 |
| 16 kHz r2 | Pacific Sound | 306,060 | 425.083333333 |
| 32 kHz final | NOAA ONMS | 2,744,345 | 1,905.795138889 |
| 32 kHz final | ONC | 424,800 | 295.000000000 |
| 32 kHz final | Orcasound | 1,069,970 | 743.034722222 |
| 32 kHz final | Pacific Sound | 615,047 | 427.115972222 |

### What "32 kHz source" means

Source preparation began on 2 July 2026. The train-ready corpus completed on
4 July 2026 at 10:01:36 MSK. The download jobs used
`raw_sample_rate=32000` and `raw_skip_below_sample_rate=true`: recordings
below 32 kHz were rejected instead of upsampled, and accepted recordings were
converted to 32 kHz clips.

The downloaders removed their temporary native-rate downloads after
processing. A complete native-sample-rate archive was not retained. Therefore
the final 32 kHz clips can seed a paired 32/16/8 kHz experiment without another
download, but they cannot recover bandwidth above 16 kHz or the original
arbitrary native rates.

For audit only, the historical ML Space locations were:

- 8 kHz: `/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/datasets/audio/marine-mammal/animal2vec_pretraining_data/cetaceans_8khz_10s_2026-06-05_a2v`
- 16 kHz r2: `/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/datasets/audio/marine-mammal/animal2vec_pretraining_data/cetaceans_16khz_5s_2026-06-25_a2v`
- 32 kHz final: `/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/datasets/audio/marine-mammal/animal2vec_pretraining_data/cetaceans_32khz_2p5s_2026-07-04_wholistener_trainready_a2v`
- 32 kHz processed sources: `/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/datasets/audio/marine-mammal/animal2vec_pretraining_sources_32khz_2p5s_2026-07-02`

These paths are provenance evidence, not runnable defaults.

## Recovered run artifacts

The retained Hydra output directories are listed below. Each contains the
resolved `.hydra` configuration and the Fairseq training log; the paths are
historical evidence and are not used as defaults by this repository.

| Run | Historical output directory |
| --- | --- |
| first fp16 run | `/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/animal2vec/outputs/2026-05-13/08-53-25` |
| 8 kHz resume from 7,500 | `/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/animal2vec/outputs/2026-06-13/13-36-07` |
| 8 kHz, lr 4e-5 | `/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/animal2vec/outputs/2026-06-14/17-45-26` |
| 8 kHz resume from 13,581, lr 1e-4 | `/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/animal2vec/outputs/2026-06-15/20-49-05` |
| 16 kHz r2 | `/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/cetaceans-filtering/outputs/2026-06-26/21-41-03` |

The previously unknown TensorBoard locations were recovered as
`.../animal2vec/outputs/2026-05-13/08-53-25/tb` for the first fp16 run and
`.../animal2vec/outputs/2026-06-12/08-47-39/tb` for the stopped orange run.
The preserved handoff also contains the blue lr 5e-5, lr 4e-5, resume13,581,
and 16 kHz r2 event directories.

On 8 September 2026, the original fp16, lr 5e-5, lr 4e-5, resume7,500, and
resume13,581 checkpoint directories were no longer present. Their complete
historical checkpoint listings therefore cannot be reconstructed. The files
that survived in `models/animal2vec/selected_checkpoints` are enumerated in its
`MANIFEST.md`; the manifest also names some earlier files that had already
been pruned and should not be mistaken for a current disk listing.

The live 16 kHz run directory retained `checkpoint1.pt` through
`checkpoint4.pt`, updates 122k-129k, and `checkpoint_last.pt`. Its `validated`
directory additionally retained 92k, 108k, 117k, and 121k-129k. All of
`BEST.pt`, `BEST_WATKINS.pt`, `BEST_KCLASS.pt`, and `BEST_COMPOSITE.pt` point
to 129k; `BEST_FILTER.pt` points to 92k. Thus checkpoints after 27k are
confirmed, through update 129,000.

The later 32 kHz run at
`models/animal2vec/runs/a2v_32khz_2p5s_wholistener_scratch_lr1e4_20260704`
retained `checkpoint1.pt`, updates 113k-120k, and `checkpoint_last.pt`.

## 16 kHz training lineage

The initial run and three continuations form the productive lineage. Update
ranges below are the first and last JSON records in each log; overlapping
ranges are expected because continuation jobs resumed from the last saved
thousand-update checkpoint rather than from the unsaved tail.

| Job | Start (MSK) | Logged updates | Learning rate in retained log | Result |
| --- | --- | ---: | --- | --- |
| initial | 26 June 2026 | 2,050-31,850 | last: `9.05764e-5` | stopped at the one-epoch boundary; checkpoint at 31,852 |
| `r3` | 28 June 2026 17:03 | 31,900-56,150 | `9.67576e-5` -> `8.61357e-5` | productive continuation |
| `r4_olga` | 29 June 2026 20:56 | 56,050-128,800 | `8.61928e-5` -> `3.08276e-5` | productive continuation |
| `mainval` | 3 July 2026 09:36 | 128,050-129,600 | `3.14018e-5` -> `3.02185e-5` | productive continuation |

Three additional launches on 28 June did not advance training: one failed on
a missing `model.max_update` config field, and two produced no training JSON
records. They are not counted as continuation training jobs.

The final `checkpoint_last.pt` stores `num_updates=129000`. Its accumulated
`extra_state.previous_training_time` is
`522147.3967317343` seconds, or **145.040943536593 hours**. This is lineage
training time, not billed GPU-hours and not time spent in failed jobs or
unsaved tails.

### Schedule and stop conditions

- The initial 16 kHz config had `max_epoch=1` and `max_update=120000`.
  It stopped at the epoch boundary around update 31,852, not at 120,000.
- Continuation changed `max_epoch` from 1 to 5 and changed
  `optimization.max_update`, `lr_scheduler.max_update`, and
  `model.max_update` from 120,000 to 200,000.
- Base learning rate remained `1e-4`, warmup remained 10,000 updates, and
  optimizer and scheduler state were not reset. Extending the cosine horizon
  nevertheless raised the LR from `9.05764e-5` at 31,850 to
  `9.67576e-5` at 31,900 (about +6.8%). The continuation is therefore not a
  mathematically uninterrupted copy of the original 120k cosine curve.
- The separate 8 kHz `resume13581` run also kept optimizer and scheduler
  state. Its logged LR is `9.9736e-5` at update 13,600 with base
  `lr=1e-4`; this is continued cosine scheduling, not a fresh warmup.
- EMA annealing ends at update 300,000, beyond both the original 120,000 and
  continuation 200,000 limits. It never reached its configured endpoint in
  these runs. The retained configs and logs contain no rationale, so whether
  that mismatch was intentional cannot be established.
- The one-epoch limit ended the initial 16 kHz process and three productive
  jobs subsequently continued it. The retained evidence does not say whether
  that continuation was already planned when the initial job was launched.

### Throughput interpretation

`torch_compile=false` in the initial 16 kHz run and all continuation jobs.
The reported roughly three-fold per-update comparison is therefore not a
`torch.compile` speedup. Nor can it be attributed mechanically to five-second
clips: both compared recipes use 80,000 input samples per clip, about 80 source
clips per optimizer update, `clone_batch=3`, and `update_freq=10`. The 16 kHz
run represents half as many seconds of sound per update, but essentially the
same number of input samples seen by the model.

The resolved Hydra configs for the retained 8 kHz bf16 continuations and the
initial 16 kHz job agree on those batching values, and no JSON or restart log
records a mid-run change. The original blue checkpoint directory has been
deleted, so its unavailable segment cannot be checked more strongly than the
surviving resume/config evidence allows.

The retained runs differ in code and runtime (Torch 1.13 versus the Torch 2
port), optimizer wrapper (composite versus Adam), data layout/I/O, and A100
compatibility changes mentioned by the continuation logs. The logs do not
contain a controlled timing ablation that separates these effects. The only
defensible budget is therefore a short paired benchmark using identical 32
kHz-parent rows, sample counts, batching, code, and hardware.

All retained multi-GPU bf16 runs identify two NVIDIA A100-SXM4-80GB devices;
the first fp16 experiment used one A100-SXM4-80GB. The JSON logs also disprove
the blanket claim that clipping was 100% for every run: clipped-record counts
were 200/200 for resume7,500, 199/318 for lr 4e-5, 829/2,436 for
resume13,581, and 581/597 for the initial 16 kHz run. Gradient clipping is a
material confound, but it was not constant across all comparisons.

The saved Hydra configs and training logs do not embed a Git commit, branch, or
complete package-version fingerprint. They establish the Torch 1.13 versus
Torch 2 implementation families, but the exact repository revision for each
historical run cannot be recovered from any retained artifact.

## Watkins scores in checkpoint names

The legacy 0.89-0.91 values are exploratory, **not publication-canonical**.
The naming harness used the fixed BEANS Watkins train/test split (1,357 train,
340 test, 32 observed classes), disabled masking, extracted L0-L15 plus the
final representation, mean-pooled each representation, then fit a
`StandardScaler` and balanced logistic regression
(`C=1`, `max_iter=2000`). The stored score was the maximum test macro-F1
across all 17 representations.

That procedure selects the layer directly on the test set, so the number in a
checkpoint filename is test-leaky (an oracle best-layer test score). For a
paper, fix the final layer in advance or select a layer only on a validation
partition inside the training data, then evaluate the held-out test set once.
The recovered ML Space reports explain the approximately 0.90 variant. The
prototype in [PR #10](https://github.com/KekStroke/cetaceans-filtering/pull/10)
advertised approximately 0.70 at 8 kHz and 0.72 at 16 kHz from a different,
also non-canonical harness: it used the official train/test artifacts, did not
consistently force `mask=False`, selected the layer on test, and fit label
encoding across train and test. That branch committed no matching result
report. No retained report, script invocation, or config identifies the 0.54
variant.

An additional content audit found nine exact audio hashes shared by the legacy
official train and test artifacts, all with conflicting labels, plus 16
conflicting duplicate-content pairs inside train. The publication protocol in
[`configs/animal2vec_validation/watkins-split-v1.json`](../../configs/animal2vec_validation/watkins-split-v1.json)
keeps all 340 official test rows untouched, excludes the contaminated 41 train
rows, and fixes 1,051 train plus 265 validation rows before any checkpoint is
scored. The canonical harness is documented in
[`docs/marine_mammal_ssl/validation/README.md`](../marine_mammal_ssl/validation/README.md).
