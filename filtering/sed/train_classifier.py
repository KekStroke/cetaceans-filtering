"""Hydra entrypoint for SED binary classifier training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hydra
from omegaconf import DictConfig, OmegaConf
from classifier.pipeline import run_training


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    cfg = config["sed_training"]
    args = SimpleNamespace(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    args.embeddings = Path(hydra.utils.to_absolute_path(str(args.embeddings)))
    args.embeddings_manifest = Path(hydra.utils.to_absolute_path(str(args.embeddings_manifest)))
    args.annotations_manifest = Path(hydra.utils.to_absolute_path(str(args.annotations_manifest)))
    args.output_dir = Path(hydra.utils.to_absolute_path(str(args.output_dir)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_training(args)


if __name__ == "__main__":
    main()
