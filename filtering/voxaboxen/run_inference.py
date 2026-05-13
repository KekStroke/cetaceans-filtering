"""Run Voxaboxen inference from this repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig


def _split_info_fp(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / f"{split}_info.csv"


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    cfg = config.voxaboxen
    voxaboxen_dir = Path(hydra.utils.to_absolute_path(str(cfg.voxaboxen_dir)))
    python_exe = Path(hydra.utils.to_absolute_path(str(cfg.voxaboxen_python)))
    project_dir = Path(hydra.utils.to_absolute_path(str(cfg.output_project_dir)))
    dataset_dir = Path(hydra.utils.to_absolute_path(str(cfg.output_dataset_dir)))
    params_fp = project_dir / str(cfg.experiment_name) / "params.yaml"
    file_info_fp = _split_info_fp(dataset_dir, str(cfg.inference_split))

    cmd = [
        str(python_exe),
        "main.py",
        "inference",
        "--model-args-fp",
        str(params_fp),
        "--file-info-for-inference",
        str(file_info_fp),
        "--detection-threshold",
        str(cfg.detection_threshold),
        "--classification-threshold",
        str(cfg.classification_threshold),
    ]
    if bool(cfg.disable_bidirectional_inference):
        cmd.append("--disable-bidirectional")

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=voxaboxen_dir, check=True)


if __name__ == "__main__":
    main()
