# animal2vec Dataset Preparation

Short workflow for downloading cetacean audio and preparing an animal2vec pretraining manifest.

## Inputs

Use the source downloaders in `utils/datasets_downloads/`:

- `download_noaa_onms.py`
- `download_orcasound.py`
- `download_pacific_sound.py`
- `download_onc_hydrophones.py`

For each run choose local or cluster storage paths:

- `sample_rate`: for example `16000`
- `segment_duration`: for example `5`
- `frames = sample_rate * segment_duration`: for `16000 * 5`, use `80000`
- `DATA_ROOT`: a filesystem with enough free space for downloaded WAVs
- `SOURCE_ROOT="$DATA_ROOT/animal2vec_pretraining_sources_<run_id>"`
- `STAGING_ROOT="$DATA_ROOT/animal2vec_pretraining_staging_<run_id>"`
- `FINAL_ROOT="$DATA_ROOT/animal2vec_pretraining_<run_id>_a2v"`

## Download

Run downloads on a machine with enough network, CPU, and disk. For large runs, use your lab's scheduler or a long-lived server session; do not run them on a weak login node.

Common Hydra overrides:

```bash
data_loading.raw_datasets_path="$SOURCE_ROOT"
data_loading.raw_segment_duration="$SEGMENT_SECONDS"
data_loading.raw_sample_rate="$SAMPLE_RATE"
data_loading.raw_skip_below_sample_rate=true
```

Use source-specific workers based on available CPU and I/O capacity:

```bash
data_loading.sources.noaa.download_workers=16
data_loading.sources.orcasound.download_workers=16
data_loading.sources.pacific_sound.download_workers=16
data_loading.sources.onc.download_workers=16
```

Wait for all source downloads before building the combined manifest. If a source wrote WAV files but missed `manifest.jsonl`, rebuild that manifest first with `scripts/animal2vec_dataset/write_manifest_parallel.py`.

## Build animal2vec Layout

Scripts are in `scripts/animal2vec_dataset/`.

Build a deduplicated TSV:

```bash
python scripts/animal2vec_dataset/build_pretrain_tsv.py \
  "$SOURCE_ROOT/noaa_onms" \
  "$SOURCE_ROOT/orcasound" \
  "$SOURCE_ROOT/pacific_sound" \
  "$SOURCE_ROOT/onc" \
  --output "$STAGING_ROOT/pretrain_all.tsv" \
  --root "$SOURCE_ROOT" \
  --default-sample-rate "$SAMPLE_RATE" \
  --check-exists
```

Keep only exact-length segments:

```bash
python scripts/animal2vec_dataset/filter_exact_tsv.py \
  --input "$STAGING_ROOT/pretrain_all.tsv" \
  --output "$STAGING_ROOT/pretrain_exact.tsv" \
  --frames "$FRAMES" \
  --report "$LOG_DIR/filter_exact_report.txt"
```

Create the animal2vec-compatible structure:

```bash
python scripts/animal2vec_dataset/prepare_manifest_for_animal2vec.py \
  --input-manifest "$STAGING_ROOT/pretrain_exact.tsv" \
  --output-root "$FINAL_ROOT" \
  --split-name pretrain \
  --mode symlink

: > "$FINAL_ROOT/manifest/valid_0.tsv"
```

## Validate

```bash
MAN="$FINAL_ROOT/manifest/pretrain.tsv"
head -n 1 "$MAN"
echo rows=$(( $(wc -l < "$MAN") - 1 ))
tail -n +2 "$MAN" | cut -f2 | grep -vc "^$FRAMES$"
```

Also spot-check that a first and last row have both:

- `$FINAL_ROOT/wav/...wav`
- `$FINAL_ROOT/lbl/...h5`

## Reference Result

The 2026-06-26 16 kHz / 5 s build produced:

- rows: `2555743`
- required frames: `80000`
- bad frame-count rows: `0`
- missing audio during final conversion: `0`

Orcasound `wholistener` was recovered by rebuilding `manifest.jsonl` from `577014` already-written WAV files.
