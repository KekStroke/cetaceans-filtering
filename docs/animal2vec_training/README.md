# animal2vec Training

The training code is packaged under `animal2vec/` and is documented in
`animal2vec/README.md`.

Start there for:

- Python and Docker setup
- manifest layout
- smoke tests
- full 16 kHz / 5 second pretraining
- multi-GPU overrides
- checkpoint resume behavior

The portable entrypoint is:

```bash
python -m animal2vec.train --config-name cetaceans/smoke_16khz_5s_torch2 task.data=/path/to/manifest
```

The full default recipe is:

```bash
python -m animal2vec.train task.data=/path/to/manifest
```
