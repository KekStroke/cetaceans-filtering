# cetaceans-filtering

Cetacean audio filtering and classification built on [Perch](https://github.com/google-research/perch-hoplite) embeddings.

## Setup

```powershell
uv sync
uv sync --group perch
```

All scripts use [Hydra](https://hydra.cc). Override any config value inline: `key=value`.
Full parameter docs live in each YAML under `configs/`.

Hydra timestamp folders are disabled in `configs/config.yaml`; script artifacts go to explicit `data/...` and `outputs/...` paths.

---

## Project structure

```text
filtering/
|-- embed/
|   `-- perch_v2_embed.py           # compute Perch embeddings for audio datasets
|-- watkins/
|   |-- train_classifier.py         # multiclass species classifier
|   `-- classifier/                 # data loading, metrics, reporting, pipeline
|-- sed/
|   |-- convert_annotations.py      # annotations.json -> manifest.csv
|   |-- train_classifier.py         # binary sound/noise classifier
|   `-- classifier/                 # labeling, data loading, pipeline
`-- voxaboxen/
    |-- prepare_dataset.py          # Label Studio JSON -> Voxaboxen/Raven files
    |-- run_training.py             # call external Voxaboxen training
    |-- run_inference.py            # call external Voxaboxen inference
    `-- report_results.py           # collect plots and metrics for a run

utils/
`-- datasets_downloads/
    |-- download_watkins.py
    |-- download_noaa_onms.py
    |-- download_orcasound.py
    |-- download_pacific_sound.py
    |-- download_voices_in_the_sea.py
    `-- download_onc_hydrophones.py
```

---

## Scenario 1 - Watkins species classifier

```powershell
# 1. Download dataset
uv run python utils/datasets_downloads/download_watkins.py

# 2. Compute embeddings
uv run python filtering/embed/perch_v2_embed.py perch_embeddings.audio_dir=data/watkins perch_embeddings.dataset_name=watkins perch_embeddings.db_path=outputs/perch/watkins/db perch_embeddings.export_dir=outputs/perch/watkins/export

# 3. Train multiclass classifier  (labels parsed from filenames)
uv run python filtering/watkins/train_classifier.py
```

---

## Scenario 2 - SED sound/noise binary classifier

```powershell
# 1. Download folder from Google Drive
#    Expected layout:
#    - Audio: data/sound-event-detection/*.wav
#    - Label Studio JSON: data/sound-event-detection/annotations.json

# 2. Convert annotations JSON -> flat manifest CSV
uv run python filtering/sed/convert_annotations.py

# 3. Compute embeddings  (set audio_dir + dataset_name in perch_embeddings config)
uv run python filtering/embed/perch_v2_embed.py

# 4. Train binary classifier  (window-level: sound vs noise)
uv run python filtering/sed/train_classifier.py
```

**Label logic per 5s window:**

- overlaps a `sound` event by >= `sed_training.min_overlap_s` -> `sound`
- otherwise -> `noise`
- artifact windows: kept as `noise` (`sed_training.treat_artifact_as_noise=true`) or excluded (`false`)

---

## Scenario 3 - Re-embed with a different model

```powershell
uv run python filtering/embed/perch_v2_embed.py perch_embeddings.model_name=surfperch perch_embeddings.drop_existing_db=true
# then re-run the relevant classifier script
```

---

## Scenario 4 - NOAA ONMS / SanctSound

Deployment list is configured in `configs/data_loading/data_loading.yaml`
via `data_loading.sources.noaa.deployment_prefixes`.

Download only new files, about 4 hours per run (no chunking):

```powershell
uv run python utils/datasets_downloads/download_noaa_onms.py data_loading.sources.noaa.only_new_files=true data_loading.sources.noaa.hours_per_deployment=1.34 data_loading.raw_segment_duration=-1
```

Download only new files, exactly 1 file from each deployment:

```powershell
uv run python utils/datasets_downloads/download_noaa_onms.py data_loading.sources.noaa.only_new_files=true data_loading.sources.noaa.max_files_per_deployment=1 data_loading.raw_segment_duration=-1
```

Download only new files, large pack (up to 10 files from each deployment):

```powershell
uv run python utils/datasets_downloads/download_noaa_onms.py data_loading.sources.noaa.only_new_files=true data_loading.sources.noaa.max_files_per_deployment=10 data_loading.sources.noaa.hours_per_deployment=999 data_loading.raw_segment_duration=-1
```

Same 4-hour run, but split output into 10-second WAV chunks:

```powershell
uv run python utils/datasets_downloads/download_noaa_onms.py data_loading.sources.noaa.only_new_files=true data_loading.sources.noaa.hours_per_deployment=1.34 data_loading.raw_segment_duration=10
```

Output paths:

- Download cache (original NOAA files): `data/noaa_onms/<source>/downloads/...`
- Output audio for labeling/training: `data/noaa_onms/<source>/audio/...`
- Manifest: `data/noaa_onms/<source>/manifest.jsonl`

Please cite NOAA SanctSound/ONMS data according to deployment metadata:
https://doi.org/10.25921/saca-sp25

---

## Scenario 5 - Orcasound (AWS Open Data)

Full Orcasound run (all default sources):

```powershell
uv run python utils/datasets_downloads/download_orcasound.py data_loading.sources.orcasound.max_files_per_source=null data_loading.sources.orcasound.target_hours_total=null
```

Small run (~5 hours total) with near-5-minute clips (4-6 min):

```powershell
uv run python utils/datasets_downloads/download_orcasound.py data_loading.sources.orcasound.target_hours_total=5 data_loading.sources.orcasound.duration_min_minutes=4 data_loading.sources.orcasound.duration_max_minutes=6 data_loading.sources.orcasound.assume_minutes_per_file=5 data_loading.sources.orcasound.max_files_per_source=null
```

Note: files outside the duration filter are downloaded for probing and then removed by default
(`data_loading.sources.orcasound.delete_nonmatching_downloads=true`).

Output paths:

- Download cache (original Orcasound files): `data/orcasound/<source>/downloads/...`
- Output audio for labeling/training: `data/orcasound/<source>/audio/...`
- Manifest: `data/orcasound/<source>/manifest.jsonl`

Sources are from AWS Open Data Orcasound registry:
https://registry.opendata.aws/orcasound/

---

## Scenario 6 - Voxaboxen external SED

Voxaboxen is used as an external checkout, not copied into this repository.
By default, the config expects the full Voxaboxen repo next to this project as `../voxaboxen`.

Put inputs here or override the paths in the command:

- Audio: `data/voxaboxen/audio/`
- Label Studio JSON: `data/voxaboxen/annotations/annotations.json`
- Config: `configs/voxaboxen/voxaboxen.yaml`

```powershell
# 1. Convert Label Studio annotations to Voxaboxen/Raven format
uv run python filtering/voxaboxen/prepare_dataset.py

# Example: run a named dataset without editing YAML
uv run python filtering/voxaboxen/prepare_dataset.py voxaboxen.dataset_name=my_dataset voxaboxen.audio_dir="data/my_dataset/audio" voxaboxen.annotations_json="data/my_dataset/annotations.json"

# 2. Train Voxaboxen through the external checkout
uv run python filtering/voxaboxen/run_training.py

# 3. Collect plots and metrics for viewing/downloading
uv run python filtering/voxaboxen/report_results.py

# 4. Run inference with the trained experiment
uv run python filtering/voxaboxen/run_inference.py
```

Default smoke-test settings are small on purpose:

- `n_epochs=2`
- `batch_size=4`
- `encoder_type=beats`
- `bidirectional=true`
- `n_map=51`

For a longer run without editing YAML:

```powershell
uv run python filtering/voxaboxen/run_training.py voxaboxen.n_epochs=8 voxaboxen.experiment_name=beats_binary_8ep
```

Output paths:

- Prepared Voxaboxen dataset: `outputs/voxaboxen/datasets/<dataset_name>/`
- Voxaboxen project config and runs: `outputs/voxaboxen/projects/<dataset_name>_experiment/`
- Compact report: `<experiment>/report/report.md`
- Metrics table: `<experiment>/report/metrics.csv`
- Loss plot: `<experiment>/report/loss.png`
- Main comparison metrics: `mean_ap@0.5`, `mean_ap@0.8`
- `unknown_labels` are treated as background/noise for binary whale-sound detection

---

## Outputs

| File | Description |
| --- | --- |
| `embeddings.npy` | `[N, 1280]` float32 array, one row per 5s window |
| `manifest.csv` | Window index, filename, start/end time |
| `model.joblib` | `{"model": Pipeline, "label_encoder": LabelEncoder}` |
| `metrics.json` | Full per-class metrics and confusion matrices |
| `summary.json` | Macro-F1 per split + report image paths |

```python
import joblib

bundle = joblib.load("outputs/.../model.joblib")
labels = bundle["label_encoder"].inverse_transform(
    bundle["model"].predict(X)  # X: [M, 1280] numpy array
)
```

---

## Marine Mammal SSL — progress (session summary)

Work in support of the SSL pretraining effort:

- **Data inventory & sufficiency** — ~3,089 h unlabeled @ 8 kHz (estimated, *unaudited* — the source table foots to ~3,359 h; needs a real manifest audit) + labeled K-class / field / Watkins. Verdict: **quality > scale**.
- **Sample rate — 8 → 16 kHz justified** — measured **+10.2 pts** macro-F1 on Olga K-calls; returns flatten above 16 kHz.
- **"Loss is huge" is a non-problem** — animal2vec = data2vec-2.0, whose latent-MSE is scale-free / uninterpretable; judge by **loss-trend + target-variance + grad-norm + a frozen probe**.
- **animal2vec training speed** — #1 lever = raise the data2vec-2.0 **multimask `M`** (amortizes the EMA-teacher forward); then bf16 (not fp16), `torch.compile`/SDPA, and a sharded data pipeline.
- **BEST-RQ reference trainer — built & validated** — no EMA teacher, **interpretable CE loss from ln(8192)=9.0** + a frozen-probe health metric; **~1455 clips/s** on a laptop RTX 5090. Drop-in for the SSL run (swap the audio loader for the corpus).
- **Frozen-probe benchmark** — fair per-encoder best-layer: AVES **0.894** / wav2vec2 **0.875** / whisper **0.875** / log-mel 0.654 (all SSL ≫ handcrafted; last-layer probing flips conclusions).
- **SHAP / interpretability** — attribution↔energy **r = 0.867** (K21: 52 % energy, 77 % attribution > 4 kHz) → mechanistic *why* for 16 kHz.
- **This PR** — enable the NOAA `dclde/pifsc/nefsc/afsc` SSL prefixes.
