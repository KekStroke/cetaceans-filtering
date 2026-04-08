"""Hydra entrypoint for Watkins species classifier training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hydra
from omegaconf import DictConfig, OmegaConf
from classifier.models import validate_aggregation_method
from classifier.pipeline import run_training


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    cfg = config["perch_training"]

    args = SimpleNamespace(
        **OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
    )
    args.embeddings = Path(hydra.utils.to_absolute_path(str(args.embeddings)))
    args.manifest = Path(hydra.utils.to_absolute_path(str(args.manifest)))
    args.output_dir = Path(hydra.utils.to_absolute_path(str(args.output_dir)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.aggregation_method = validate_aggregation_method(args.aggregation_method)
    run_training(args)


if __name__ == "__main__":
    main()
