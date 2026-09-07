"""Versioned split manifests and validation-only model selection."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence, TypeVar


class ProtocolError(ValueError):
    """Raised when a validation protocol is ambiguous or leaks data."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")
_T = TypeVar("_T")


def sha256_file(
    path: os.PathLike[str] | str, chunk_size: int = 8 * 1024 * 1024
) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_identifier(value: str, field: str) -> str:
    value = value.strip()
    if not value:
        raise ProtocolError(f"{field} must not be empty")
    if value.startswith(("/", "\\")) or _WINDOWS_ABS_RE.match(value):
        raise ProtocolError(
            f"{field} must be a logical ID, not an absolute path: {value!r}"
        )
    if ".." in PurePosixPath(value.replace(chr(92), "/")).parts:
        raise ProtocolError(f"{field} must not escape its dataset root: {value!r}")
    return value


@dataclasses.dataclass(frozen=True)
class SplitManifest:
    schema_version: int
    protocol_id: str
    dataset_name: str
    dataset_revision: str
    record_id_field: str
    artifacts: Mapping[str, str]
    source_partitions: Mapping[str, str]
    splits: Mapping[str, tuple[str, ...]]
    seed: int
    classifier: Mapping[str, Any]

    @property
    def all_ids(self) -> frozenset[str]:
        return frozenset(
            record_id for values in self.splits.values() for record_id in values
        )

    def partition(
        self, records: Mapping[str, _T], *, require_exact: bool = True
    ) -> dict[str, list[_T]]:
        known = set(records)
        missing = sorted(self.all_ids - known)
        extra = sorted(known - self.all_ids)
        if missing or (require_exact and extra):
            raise ProtocolError(
                "dataset IDs do not match the fixed manifest: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        return {
            role: [records[record_id] for record_id in self.splits[role]]
            for role in ("train", "validation", "test")
        }

    def verify_artifacts(self, data_dir: os.PathLike[str] | str) -> None:
        root = Path(data_dir).resolve()
        for relative_name, expected in self.artifacts.items():
            path = (root / relative_name).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ProtocolError(
                    f"dataset artifact escapes data root: {relative_name!r}"
                ) from exc
            if not path.is_file():
                raise ProtocolError(f"dataset artifact is missing: {relative_name}")
            actual = sha256_file(path)
            if actual != expected:
                raise ProtocolError(
                    f"dataset artifact hash mismatch for {relative_name}: "
                    f"expected {expected}, got {actual}"
                )


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{field} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], field: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ProtocolError(
            f"{field} keys differ from schema: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def load_split_manifest(path: os.PathLike[str] | str) -> SplitManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    root = _require_mapping(raw, "manifest")
    _require_exact_keys(
        root,
        {"schema_version", "protocol_id", "dataset", "splits", "protocol"},
        "manifest",
    )
    if root.get("schema_version") != 1:
        raise ProtocolError("split manifest schema_version must be 1")
    protocol_id = _portable_identifier(
        str(root.get("protocol_id", "")), "protocol_id"
    )
    dataset = _require_mapping(root.get("dataset"), "dataset")
    _require_exact_keys(
        dataset,
        {
            "name",
            "revision",
            "record_id_field",
            "artifacts",
            "source_partitions",
        },
        "dataset",
    )
    dataset_name = _portable_identifier(
        str(dataset.get("name", "")), "dataset.name"
    )
    revision = _portable_identifier(
        str(dataset.get("revision", "")), "dataset.revision"
    )
    record_id_field = _portable_identifier(
        str(dataset.get("record_id_field", "")), "dataset.record_id_field"
    )

    artifacts_raw = _require_mapping(dataset.get("artifacts"), "dataset.artifacts")
    if not artifacts_raw:
        raise ProtocolError(
            "dataset.artifacts must contain at least one hashed artifact"
        )
    artifacts: dict[str, str] = {}
    for name, digest in artifacts_raw.items():
        safe_name = _portable_identifier(str(name), "dataset.artifacts key")
        digest = str(digest).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ProtocolError(
                f"dataset artifact {safe_name!r} has an invalid SHA256"
            )
        artifacts[safe_name] = digest

    partitions_raw = _require_mapping(
        dataset.get("source_partitions"), "dataset.source_partitions"
    )
    if set(partitions_raw) != {"train", "test"}:
        raise ProtocolError(
            "dataset.source_partitions must contain exactly train and test"
        )
    source_partitions = {
        role: _portable_identifier(
            str(partitions_raw[role]), f"dataset.source_partitions.{role}"
        )
        for role in ("train", "test")
    }
    unknown_artifacts = set(source_partitions.values()) - set(artifacts)
    if unknown_artifacts:
        raise ProtocolError(
            "source partitions reference unhashed artifacts: "
            f"{sorted(unknown_artifacts)}"
        )

    split_raw = _require_mapping(root.get("splits"), "splits")
    if set(split_raw) != {"train", "validation", "test"}:
        raise ProtocolError(
            "splits must contain exactly train, validation, and test"
        )
    splits: dict[str, tuple[str, ...]] = {}
    seen: dict[str, str] = {}
    for role in ("train", "validation", "test"):
        values = split_raw[role]
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
        ):
            raise ProtocolError(f"splits.{role} must be a non-empty array")
        identifiers = tuple(
            _portable_identifier(str(value), f"splits.{role}") for value in values
        )
        if len(set(identifiers)) != len(identifiers):
            raise ProtocolError(f"splits.{role} contains duplicate IDs")
        for record_id in identifiers:
            if record_id in seen:
                raise ProtocolError(
                    f"record {record_id!r} occurs in both {seen[record_id]} and {role}"
                )
            seen[record_id] = role
        splits[role] = identifiers

    protocol = _require_mapping(root.get("protocol"), "protocol")
    _require_exact_keys(
        protocol,
        {
            "mask",
            "pooling",
            "layer_selection",
            "selection_metric",
            "seed",
            "classifier",
        },
        "protocol",
    )
    if protocol.get("mask") is not False:
        raise ProtocolError("protocol.mask must be false")
    if protocol.get("layer_selection") != "validation":
        raise ProtocolError("protocol.layer_selection must be 'validation'")
    if protocol.get("selection_metric") != "macro_f1":
        raise ProtocolError("protocol.selection_metric must be 'macro_f1'")
    if protocol.get("pooling") != "mean":
        raise ProtocolError("protocol.pooling must be 'mean'")
    classifier = _require_mapping(
        protocol.get("classifier"), "protocol.classifier"
    )
    _require_exact_keys(
        classifier,
        {"name", "C", "class_weight", "max_iter", "solver"},
        "protocol.classifier",
    )
    if classifier.get("name") != "logistic_regression":
        raise ProtocolError(
            "the canonical v1 classifier must be logistic_regression"
        )
    try:
        seed = int(protocol["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("protocol.seed must be an integer") from exc
    try:
        c_value = float(classifier["C"])
        max_iter = int(classifier["max_iter"])
    except (TypeError, ValueError) as exc:
        raise ProtocolError("classifier C/max_iter have invalid types") from exc
    if not math.isfinite(c_value) or c_value <= 0 or max_iter <= 0:
        raise ProtocolError("classifier C and max_iter must be positive")
    if classifier["class_weight"] not in ("balanced", None):
        raise ProtocolError("classifier.class_weight must be 'balanced' or null")
    if not str(classifier["solver"]).strip():
        raise ProtocolError("classifier.solver must not be empty")

    return SplitManifest(
        schema_version=1,
        protocol_id=protocol_id,
        dataset_name=dataset_name,
        dataset_revision=revision,
        record_id_field=record_id_field,
        artifacts=artifacts,
        source_partitions=source_partitions,
        splits=splits,
        seed=seed,
        classifier=dict(classifier),
    )


def select_layer_from_validation(validation_scores: Mapping[str, float]) -> str:
    if not validation_scores:
        raise ProtocolError("at least one validation-layer score is required")
    checked: list[tuple[str, float]] = []
    for layer, score in validation_scores.items():
        value = float(score)
        if not math.isfinite(value):
            raise ProtocolError(
                f"validation score for {layer!r} is not finite"
            )
        checked.append((str(layer), value))
    return sorted(checked, key=lambda item: (-item[1], item[0]))[0][0]


def evaluate_selected_layer(
    layers: Sequence[str],
    validation_evaluator: Callable[[str], float],
    test_evaluator: Callable[[str], _T],
) -> tuple[str, dict[str, float], _T]:
    """Select exclusively on validation and touch test once for the chosen layer."""

    if not layers:
        raise ProtocolError("no encoder layers were provided")
    validation_scores = {
        str(layer): float(validation_evaluator(str(layer))) for layer in layers
    }
    selected = select_layer_from_validation(validation_scores)
    return selected, validation_scores, test_evaluator(selected)
