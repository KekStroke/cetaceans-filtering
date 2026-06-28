# animal2vec Training

This directory contains the animal2vec pretraining code vendored into
`cetaceans-filtering` as a normal Python package. It can be run from a clone of
this repository without MLS, shared mounts, or machine-specific paths.

## Layout

```text
animal2vec/
|-- train.py                         # Hydra/Fairseq entrypoint
|-- requirements-torch2.txt           # tested Torch 2.x training stack
|-- Dockerfile                        # optional CUDA container
|-- configs/
|   `-- cetaceans/                    # portable configs for this repository
|-- nn/                               # Fairseq task, model, criterion, trainer
`-- scripts/
    `-- animal2vec_manifest.py        # generic manifest builder
```

The default config is `cetaceans/pretrain_16khz_5s_torch2`.

## Environment

Use Python 3.10. The pinned stack tested for Torch 2 training is in
`animal2vec/requirements-torch2.txt`.

```bash
python3.10 -m venv .venv-animal2vec
source .venv-animal2vec/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r animal2vec/requirements-torch2.txt
```

For a containerized setup:

```bash
docker build -f animal2vec/Dockerfile -t cetaceans-filtering-animal2vec .
```

## Data

The portable configs expect this Fairseq-style manifest directory by default:

```text
data/animal2vec/manifest/
|-- pretrain.tsv
`-- valid_0.tsv
```

For 16 kHz 5 second chunks, every training row should contain exactly `80000`
frames. The default config enforces this with:

```yaml
task:
  sample_rate: 16000
  min_sample_size: 80000
  max_sample_size: 80000
  enable_padding: false
```

You can point at any compatible manifest without editing files:

```bash
python -m animal2vec.train task.data=/path/to/manifest
```

## Training

The default recipe is the 16 kHz / 5 second Torch 2 recipe:

```bash
python -m animal2vec.train \
  task.data=/path/to/manifest \
  checkpoint.save_dir=outputs/animal2vec/checkpoints/pretrain_16khz_5s_torch2 \
  common.tensorboard_logdir=outputs/animal2vec/tensorboard/pretrain_16khz_5s_torch2
```

Important defaults:

- `lr: 1e-4`
- `max_tokens: 320000`
- `update_freq: 10`
- `clone_batch: 3`
- `bf16: true`
- `torch_compile: false`
- `save_interval_updates: 1000`
- `keep_interval_updates: 8`
- `keep_interval_updates_pattern: -1`
- `keep_best_checkpoints: -1`

The checkpoint retention values are set explicitly because Fairseq 0.12 can
crash when those fields are left as `None`.

Additional 16 kHz / 5 second experimental recipes:

- `cetaceans/pretrain_16khz_5s_scale_aware_torch2`: single frontend, one mask scale sampled per batch.
- `cetaceans/pretrain_16khz_5s_mixed_mask_torch2`: single frontend, several mask scales overlaid together.
- `cetaceans/pretrain_16khz_5s_multires_torch2`: low/mid/high frontend branches, mixed masks, and branch dropout.

Example:

```bash
python -m animal2vec.train --config-name cetaceans/pretrain_16khz_5s_multires_torch2 \
  task.data=/path/to/manifest
```

The default `cetaceans/pretrain_16khz_5s_torch2` recipe keeps the original
single-scale `mask_prob`/`mask_length` behavior.

Masking modes:

- `random_per_batch`: choose one `(mask_length, mask_prob)` pair for the whole batch. This keeps the objective focused on one time scale at a time.
- `mixed`: compute masks for all configured scales and union them. This asks the model to recover short transients and longer context in the same update, so the per-scale probabilities should be lower.

With the current 16 kHz frontend stride, lengths `2`, `16`, `80`, and `200`
are approximately 5 ms, 40 ms, 200 ms, and 500 ms.

## Multi-GPU

For one process per GPU on a single machine, set the distributed fields through
Hydra overrides. Example for two GPUs:

```bash
python -m animal2vec.train \
  distributed_training.distributed_world_size=2 \
  distributed_training.distributed_num_procs=2 \
  distributed_training.nprocs_per_node=2 \
  task.data=/path/to/manifest
```

If your launcher supplies distributed environment variables, keep the config
values aligned with the number of local processes it starts.

## Resuming

By default, training resumes from `checkpoint_last.pt` in `checkpoint.save_dir`
when that file exists. To resume from a numbered checkpoint:

```bash
python -m animal2vec.train \
  checkpoint.save_dir=outputs/animal2vec/checkpoints/pretrain_16khz_5s_torch2 \
  checkpoint.restore_file=checkpoint_1_1000.pt
```

Keep `reset_optimizer`, `reset_lr_scheduler`, `reset_meters`, and
`reset_dataloader` as `false` for a true continuation.

## Notes

- The code applies small compatibility patches before Fairseq imports so that
  the original animal2vec/Fairseq path runs on Torch 2.x.
- `model.torch_compile` is configurable, but the portable default is `false`
  because compile overhead and graph breaks did not beat eager mode in current
  smoke benchmarks.
- Large model training needs substantial VRAM. Reduce `dataset.max_tokens` or
  `model.clone_batch` when validating on smaller GPUs.
