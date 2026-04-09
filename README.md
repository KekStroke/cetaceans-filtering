# cetaceans-filtering

Cetacean audio filtering and classification built on [Perch](https://github.com/google-research/perch-hoplite) embeddings.

## Setup

```bash
uv sync              # base deps
uv sync --group perch  # Perch + TensorFlow (needed for embedding)
```

All scripts use [Hydra](https://hydra.cc). Override any config value inline: `key=value`.
Full parameter docs live in each YAML under `configs/`.

---

## Project structure

```cp
filtering/
├── embed/
│   └── perch_v2_embed.py           # compute Perch embeddings for any audio dataset
├── watkins/
│   ├── train_classifier.py         # multiclass species classifier
│   └── classifier/                 # data loading, metrics, reporting, pipeline
└── sed/
    ├── convert_annotations.py      # annotations.json → manifest.csv
    ├── train_classifier.py         # binary sound/noise classifier
    └── classifier/                 # labeling, data loading, pipeline

utils/
└── datasets_downloads/
    ├── download_watkins.py         # download Watkins marine mammal dataset
    ├── download_noaa_onms.py       # sample small subsets from NOAA ONMS / SanctSound
    └── download_manual_sed.py      # download manual SED dataset from Google Drive
```

---

## Scenario 1 — Watkins species classifier

```bash
# 1. Download dataset
uv run python utils/datasets_downloads/download_watkins.py

# 2. Compute embeddings
uv run python filtering/embed/perch_v2_embed.py

# 3. Train multiclass classifier  (labels parsed from filenames)
uv run python filtering/watkins/train_classifier.py
```

---

## Scenario 2 — SED sound/noise binary classifier

```bash
# 1. Download folder from Google Drive

# 2. Convert annotations JSON → flat manifest CSV
uv run python filtering/sed/convert_annotations.py

# 3. Compute embeddings  (set audio_dir + dataset_name in perch_embeddings config)
uv run python filtering/embed/perch_v2_embed.py

# 4. Train binary classifier  (window-level: sound vs noise)
uv run python filtering/sed/train_classifier.py
```

**Label logic per 5s window:**

- overlaps a `sound` event by ≥ `min_overlap_s` → `sound`
- otherwise → `noise`
- artifact windows: kept as `noise` (`treat_artifact_as_noise=true`) or excluded (`false`)

---

## Scenario 3 — Re-embed with a different model

```bash
uv run python filtering/embed/perch_v2_embed.py \
  perch_embeddings.model_name=surfperch \
  perch_embeddings.drop_existing_db=true
# then re-run the relevant classifier script
```

---

## Scenario 4 - NOAA ONMS / SanctSound

Deployment list is configured in `configs/data_loading/data_loading.yaml`
via `noaa_deployment_prefixes`.

Download only new files, about 4 hours per run (no chunking):

```powershell
uv run python utils/datasets_downloads/download_noaa_onms.py data_loading.noaa_only_new_files=true data_loading.noaa_hours_per_deployment=1.34 data_loading.raw_segment_duration=-1
```

Download only new files, exactly 1 file from each deployment:

```powershell
uv run python utils/datasets_downloads/download_noaa_onms.py data_loading.noaa_only_new_files=true data_loading.noaa_max_files_per_deployment=1 data_loading.raw_segment_duration=-1
```

Download only new files, large pack (up to 10 files from each deployment):

```powershell
uv run python utils/datasets_downloads/download_noaa_onms.py data_loading.noaa_only_new_files=true data_loading.noaa_max_files_per_deployment=10 data_loading.noaa_hours_per_deployment=999 data_loading.raw_segment_duration=-1
```

Same 4-hour run, but split output into 10-second WAV chunks:

```powershell
uv run python utils/datasets_downloads/download_noaa_onms.py data_loading.noaa_only_new_files=true data_loading.noaa_hours_per_deployment=1.34 data_loading.raw_segment_duration=10
```

Output paths:

- Download cache (original NOAA files): `data/noaa_onms/downloads/...`
- Output audio for labeling/training: `data/noaa_onms/audio/...`

Please cite NOAA SanctSound/ONMS data according to deployment metadata
(DOI: https://doi.org/10.25921/saca-sp25).

---

## Outputs

| File             | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `embeddings.npy` | `[N, 1280]` float32 array, one row per 5s window     |
| `manifest.csv`   | Window index, filename, start/end time               |
| `model.joblib`   | `{"model": Pipeline, "label_encoder": LabelEncoder}` |
| `metrics.json`   | Full per-class metrics and confusion matrices        |
| `summary.json`   | Macro-F1 per split + report image paths              |

```python
import joblib
bundle = joblib.load("outputs/.../model.joblib")
labels = bundle["label_encoder"].inverse_transform(
    bundle["model"].predict(X)  # X: [M, 1280] numpy array
)
```
