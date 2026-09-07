# Canonical animal2vec checkpoint validation

Publication-facing Watkins scores use one entry point:

```bash
python -m animal2vec.validation watkins \
  --checkpoint checkpoints/checkpoint.pt \
  --data-dir data/watkins-arrow \
  --split-manifest manifests/watkins-split-v1.json \
  --output outputs/animal2vec/validation/result.json
```

The runtime reads `sample_rate`, `max_sample_size`, and normalization from the
checkpoint config. Missing or contradictory values are errors. Every encoder
call is made through `animal2vec.validation.runtime.extract_features`, which
always passes `mask=False`.

## Fixed protocol

The split manifest must conform to
[`split_manifest.schema.json`](../../../configs/animal2vec_validation/split_manifest.schema.json).
It names every recording in non-overlapping `train`, `validation`, and `test`
lists and pins the source Arrow files by SHA256. Machine-local paths are not
valid recording IDs. No real IDs are included in the repository until the
audited dataset split is available; do not generate a replacement split during
a validation run.

Validate an audited manifest before scheduling GPU work:

```bash
python -m animal2vec.validation check-manifest watkins-split-v1.json \
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
