# Canonical animal2vec checkpoint validation

Publication-facing Watkins scores use one entry point:

```bash
python -m animal2vec.validation watkins \
  --checkpoint checkpoints/checkpoint.pt \
  --data-dir data/watkins-arrow \
  --split-manifest configs/animal2vec_validation/watkins-split-v1.json \
  --output outputs/animal2vec/validation/result.json
```

The runtime reads `sample_rate`, `max_sample_size`, and normalization from the
checkpoint config. Missing or contradictory values are errors. Every encoder
call is made through `animal2vec.validation.runtime.extract_features`, which
always passes `mask=False`.

## Fixed protocol

The split manifest must conform to
[`split_manifest.schema.json`](../../../configs/animal2vec_validation/split_manifest.schema.json).
It accounts for every recording in non-overlapping `train`, `validation`,
`test`, and `excluded` lists and pins the source Arrow files by SHA256.
Recording IDs are strict
partition-qualified Arrow row indices: `train:<row>` for both the training and
validation subsets of the official train artifact, and `test:<row>` for the
official test artifact. Row indices are stable because the complete Arrow files
are hash-pinned; filenames are deliberately not identifiers because Watkins
contains repeated basenames. The audited split is checked in as
`configs/animal2vec_validation/watkins-split-v1.json`; do not regenerate it
during validation.

### Audited Watkins snapshot

The manifest is content-addressed because the upstream exporter revision was
not embedded in the local Arrow files and no contemporaneous build record was
available:

- `beans_watkins-train.arrow`: 1,179,465,368 bytes, 1,357 rows,
  SHA256 `d6b6c32b9425c6233c89b39246970997b61c9d278b8d0d42cb507698d6152544`;
- `beans_watkins-test.arrow`: 154,350,408 bytes, 340 rows,
  SHA256 `6f7eafa5f0a4efae0d17b9814feed712eb9082b1ecc6baae22914c83f47bcf80`.

The content audit was performed before evaluating any checkpoint. It found nine
audio SHA256 values shared by the official train and test artifacts; every
cross-partition pair has conflicting labels. Those nine train-side rows are
excluded and every official test row remains untouched. It also found 16
within-train duplicate-content groups whose two rows have conflicting labels,
so all 32 affected rows are excluded. There were no same-label duplicate groups
to collapse. Every source row remains accountable in the manifest:
1,051 train + 265 validation + 41 excluded = 1,357 source-train rows, plus all
340 source-test rows.

The remaining 1,316 train rows retain all 32 classes, with at least two eligible
rows per class. Validation membership is cross-version-stable: within each
class, rows are ranked by SHA256 of
`watkins-validation-v1|label|audio_sha256|row_index`; the first
`max(1, min(n - 1, (n + 2) // 5))` rows are validation and the rest are
training. This yields 265 validation rows (approximately 20%) while preserving
every class in both subsets. Do not regenerate or tune this membership.

The official test artifact contains 31 of the 32 training classes. Macro-F1 is
therefore averaged over the classes present in the evaluated split, and the
result records that exact class list and its supports; an absent test class is
not injected as an artificial zero.

The checked-in manifest contains only hashes and row indices, not audio. Users
must obtain the Watkins data under its applicable access and licensing terms.

Validate an audited manifest before scheduling GPU work:

```bash
python -m animal2vec.validation check-manifest \
  configs/animal2vec_validation/watkins-split-v1.json \
  --data-dir data/watkins-arrow
```

For every checkpoint the validator:

1. fits one frozen linear probe per encoder layer on `train`;
2. selects the layer by validation macro-F1 only;
3. refits that layer on `train + validation`;
4. evaluates the held-out test split once.

Use `--selection-only` while ranking training checkpoints. Test scores must not
be used to choose a layer, checkpoint, threshold, or configuration.

The result JSON includes the code commit and dirty flag; checkpoint, embedded
config, split-manifest, and dataset-artifact hashes; checkpoint update/epoch/SR;
the exact classifier config; package versions; validation scores; and the one
held-out test score. Absolute checkpoint, dataset, and output paths are rejected
from serialized results.

## Environment

Install the pinned animal2vec environment from `animal2vec/README.md`, then the
small validation-only additions from `requirements.txt`. The old PyPI
`fairseq==0.12.2` recipe is not equivalent to the pinned Fairseq source commit.

## Historical utility

`validate.py` remains temporarily available for checkpoint slimming, live
watching, K-class/filter diagnostics, and band occlusion. Its accumulated
best-layer-on-test Watkins numbers are exploratory and must not be used as the
paper's canonical score.
