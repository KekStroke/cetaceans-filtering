"""Run Voxaboxen training from this repository without vendoring Voxaboxen."""

from __future__ import annotations

import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    cfg = config.voxaboxen
    voxaboxen_dir = Path(hydra.utils.to_absolute_path(str(cfg.voxaboxen_dir)))
    python_exe = Path(hydra.utils.to_absolute_path(str(cfg.voxaboxen_python)))
    project_dir = Path(hydra.utils.to_absolute_path(str(cfg.output_project_dir)))
    project_config = project_dir / "project_config.yaml"
    experiment_dir = project_dir / str(cfg.experiment_name)

    print(f"Experiment directory: {experiment_dir}")
    print("During/after training, check train_progress.svg, train_history.yaml, val_results.yaml, test_results.yaml.")
    print("After training, run: uv run python filtering/voxaboxen/report_results.py")

    cmd = [
        str(python_exe),
        "main.py",
        "train-model",
        "--project-config-fp",
        str(project_config),
        "--name",
        str(cfg.experiment_name),
        "--n-epochs",
        str(cfg.n_epochs),
        "--batch-size",
        str(cfg.batch_size),
        "--encoder-type",
        "beats",
        "--beats-checkpoint-fp",
        str(Path(hydra.utils.to_absolute_path(str(cfg.beats_checkpoint)))),
        "--exists-strategy",
        str(cfg.exists_strategy),
        "--num-workers",
        str(cfg.num_workers),
        "--n-map",
        str(cfg.n_map),
        "--n-val-fit",
        str(cfg.n_val_fit),
    ]
    if bool(cfg.bidirectional):
        cmd.append("--bidirectional")

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=voxaboxen_dir, check=True)


if __name__ == "__main__":
    main()
