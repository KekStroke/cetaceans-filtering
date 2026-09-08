"""Portable validation-result provenance."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .protocol import ProtocolError, SplitManifest, sha256_file


_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_absolute_local_path(value: str) -> bool:
    return value.startswith(("/", "\\")) or bool(_WINDOWS_ABS_RE.match(value))


def assert_portable_payload(value: Any, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_portable_payload(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_portable_payload(child, f"{location}[{index}]")
    elif isinstance(value, str) and _is_absolute_local_path(value):
        raise ProtocolError(
            f"portable result contains an absolute local path at {location}: {value!r}"
        )


def repository_state(repo_root: os.PathLike[str] | str) -> dict[str, Any]:
    root = Path(repo_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(dirty)}


def environment_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.system().lower(),
    }
    for package in (
        "torch",
        "torchaudio",
        "fairseq",
        "numpy",
        "scikit-learn",
        "datasets",
    ):
        try:
            distribution = importlib.metadata.distribution(package)
            entry: dict[str, Any] = {"version": distribution.version}
            direct_url = distribution.read_text("direct_url.json")
            if direct_url:
                source = json.loads(direct_url)
                vcs = source.get("vcs_info") or {}
                if vcs.get("commit_id"):
                    entry["vcs"] = vcs.get("vcs")
                    entry["commit_id"] = vcs["commit_id"]
                    if str(source.get("url", "")).startswith(("https://", "git+https://")):
                        entry["url"] = source["url"]
            versions[package] = entry
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def build_provenance(
    *,
    repo_root: os.PathLike[str] | str,
    checkpoint_path: os.PathLike[str] | str,
    checkpoint_metadata: Mapping[str, Any],
    config_sha256: str,
    split_manifest_path: os.PathLike[str] | str,
    split_manifest: SplitManifest,
    stripped_config_keys: list[str],
    ignored_state_keys: list[str],
    device: str,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    manifest = Path(split_manifest_path)
    payload = {
        "code": repository_state(repo_root),
        "checkpoint": {
            "file": checkpoint.name,
            "sha256": sha256_file(checkpoint),
            "config_sha256": config_sha256,
            **{
                key: checkpoint_metadata.get(key)
                for key in (
                    "num_updates",
                    "epoch",
                    "sample_rate",
                    "max_sample_size",
                    "normalize",
                )
            },
            "stripped_config_keys": sorted(stripped_config_keys),
            "ignored_state_keys": sorted(ignored_state_keys),
        },
        "dataset": {
            "name": split_manifest.dataset_name,
            "revision": split_manifest.dataset_revision,
            "record_id_scheme": split_manifest.record_id_scheme,
            "artifacts": dict(sorted(split_manifest.artifacts.items())),
            "source_partitions": dict(split_manifest.source_partitions),
            "validation_split_audit": dict(split_manifest.validation_split_audit),
            "split_manifest_file": manifest.name,
            "split_manifest_sha256": sha256_file(manifest),
            "split_counts": {
                role: len(values)
                for role, values in split_manifest.splits.items()
            },
            "exclusion_counts_by_reason": {
                reason: sum(
                    detail["reason"] == reason
                    for detail in split_manifest.exclusions.values()
                )
                for reason in sorted(
                    {detail["reason"] for detail in split_manifest.exclusions.values()}
                )
            },
        },
        "environment": {**environment_versions(), "device": device},
    }
    assert_portable_payload(payload)
    return payload


def write_portable_json(
    path: os.PathLike[str] | str, payload: Mapping[str, Any]
) -> None:
    assert_portable_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
