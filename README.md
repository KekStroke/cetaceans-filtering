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
    ├── download_orcasound.py       # download and process Orcasound AWS Open Data
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

## Scenario 5 - Orcasound (AWS Open Data)

Large run (all files from all default sources):

```powershell
uv run python utils/datasets_downloads/download_orcasound.py data_loading.orcasound_max_files_per_source=null data_loading.orcasound_target_hours_total=null
```

Small run capped to ~5 hours by actual decoded audio duration:

```powershell
uv run python utils/datasets_downloads/download_orcasound.py data_loading.orcasound_target_hours_total=5 data_loading.orcasound_assume_minutes_per_file=5 data_loading.orcasound_max_files_per_source=null
```

Small run (~5 hours) using mostly ~5-minute files (4-6 min filter):

```powershell
uv run python utils/datasets_downloads/download_orcasound.py data_loading.orcasound_target_hours_total=5 data_loading.orcasound_duration_min_minutes=4 data_loading.orcasound_duration_max_minutes=6 data_loading.orcasound_assume_minutes_per_file=5 data_loading.orcasound_max_files_per_source=null
```

Optional: run only selected prefixes from the default list:

```powershell
uv run python utils/datasets_downloads/download_orcasound.py data_loading.orcasound_selected_prefixes=[2019-Orcasound-examples/,humpbacks/] data_loading.orcasound_target_hours_total=5
```

Available values for `data_loading.orcasound_selected_prefixes`:

- `2017-09-27_OS_SRKW-wav/`
- `2017-09-05-SRKW-highlight-hour/`
- `2017-09-05-SRKW/`
- `2017-09-27-OS-continuous-wavs/`
- `2017_8_VesselsAndWavS/`
- `2018-sperm-whale-Yukusam/`
- `2019-11-14_PT_SRKW_HLS/`
- `2019-Orcasound-examples/`
- `2020-06-26-SRKW-Lpod/`
- `2021_9_12_OS_YearsBestVocalPassby/`
- `acoustic-separation/`
- `clap-model/`
- `data-audio-raw/`
- `humpbacks/`
- `labeled-data/`
- `machineLearningFile/`
- `orcaal-dev/`
- `orcasounds/`
- `wholistener/`

Output paths:

- Download cache (original Orcasound files): `data/orcasound/downloads/...`
- Output audio for labeling/training: `data/orcasound/audio/...`

Sources are from AWS Open Data Orcasound registry:
https://registry.opendata.aws/orcasound/

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
