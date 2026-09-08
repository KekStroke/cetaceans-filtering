"""Canonical Watkins species frozen-probe validation.

Layer selection happens on the fixed validation split. The official test split
is scored once, after refitting the selected linear probe on train+validation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import (
    ProtocolError,
    SplitManifest,
    evaluate_selected_layer,
    load_split_manifest,
)
from .provenance import build_provenance, write_portable_json
from .runtime import LoadedEncoder, extract_features, load_encoder, seed_everything


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device=cuda requested, but CUDA is not available")
    return requested


def _index_dataset(
    dataset: Any, source_partition: str
) -> dict[str, tuple[Any, int]]:
    if source_partition not in {"train", "test"}:
        raise ProtocolError(f"unknown source partition {source_partition!r}")
    return {
        f"{source_partition}:{row_index}": (dataset, row_index)
        for row_index in range(len(dataset))
    }


def _load_fixed_records(
    data_dir: Path, manifest: SplitManifest
) -> tuple[dict[str, list[Mapping[str, Any]]], list[str]]:
    from datasets import Dataset

    manifest.verify_artifacts(data_dir)
    train_dataset = Dataset.from_file(
        str(data_dir / manifest.source_partitions["train"])
    )
    test_dataset = Dataset.from_file(
        str(data_dir / manifest.source_partitions["test"])
    )
    train_label_names = list(getattr(train_dataset.features["label"], "names", []))
    test_label_names = list(getattr(test_dataset.features["label"], "names", []))
    if not train_label_names or train_label_names != test_label_names:
        raise ProtocolError(
            "source train/test artifacts must share one non-empty ClassLabel mapping"
        )
    train_index = _index_dataset(train_dataset, "train")
    test_index = _index_dataset(test_dataset, "test")

    expected_training = set(manifest.splits["train"]) | set(
        manifest.splits["validation"]
    ) | set(manifest.splits["excluded"])
    expected_test = set(manifest.splits["test"])
    if set(train_index) != expected_training:
        raise ProtocolError(
            "source train artifact does not exactly match manifest "
            "train+validation+excluded IDs"
        )
    if set(test_index) != expected_test:
        raise ProtocolError(
            "source test artifact does not exactly match manifest test IDs"
        )
    combined = {**train_index, **test_index}
    partitioned = manifest.partition(combined)
    records = {
        role: [dataset[row_index] for dataset, row_index in partitioned[role]]
        for role in ("train", "validation", "test")
    }
    return records, train_label_names


def _prepare_audio(
    array: Any, source_rate: int, encoder: LoadedEncoder
) -> Any:
    import numpy as np
    from scipy.signal import resample_poly

    spec = encoder.input_spec
    audio = np.asarray(array, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    native_limit = max(
        1, int(round(spec.max_sample_size * int(source_rate) / spec.sample_rate))
    )
    if len(audio) > native_limit:
        offset = (len(audio) - native_limit) // 2
        audio = audio[offset : offset + native_limit]
    if int(source_rate) != spec.sample_rate:
        divisor = math.gcd(int(source_rate), spec.sample_rate)
        audio = resample_poly(
            audio, spec.sample_rate // divisor, int(source_rate) // divisor
        ).astype(np.float32)
    if len(audio) > spec.max_sample_size:
        offset = (len(audio) - spec.max_sample_size) // 2
        audio = audio[offset : offset + spec.max_sample_size]
    if len(audio) < 400:
        audio = np.pad(audio, (0, 400 - len(audio)))
    if spec.normalize:
        audio = audio - float(audio.mean())
        std = float(audio.std())
        if std > 1e-8:
            audio = audio / std
    return audio.astype(np.float32)


def _sequence(tensor: Any) -> Any:
    if tensor.ndim != 3:
        raise RuntimeError(f"expected a rank-3 encoder tensor, got {tensor.shape}")
    if tensor.shape[0] == 1:
        return tensor[0]
    if tensor.shape[1] == 1:
        return tensor[:, 0]
    raise RuntimeError(
        f"canonical validator expects batch size one, got tensor {tensor.shape}"
    )


def _pooled_layers(output: Mapping[str, Any]) -> dict[str, Any]:
    pooled: dict[str, Any] = {}
    for index, layer_result in enumerate(output.get("layer_results") or []):
        tensor = (
            layer_result[0]
            if isinstance(layer_result, (tuple, list))
            else layer_result
        )
        pooled[f"L{index}"] = _sequence(tensor).float().mean(0).cpu().numpy()
    pooled["final"] = _sequence(output["x"]).float().mean(0).cpu().numpy()
    return pooled


def _embed_records(
    encoder: LoadedEncoder,
    records: Sequence[Mapping[str, Any]],
    device: str,
) -> tuple[dict[str, Any], list[Any]]:
    import numpy as np
    import torch

    by_layer: dict[str, list[Any]] = {}
    labels: list[Any] = []
    for row in records:
        audio_value = row.get("path")
        if not isinstance(audio_value, Mapping):
            raise ProtocolError("Watkins row 'path' must be a decoded audio mapping")
        audio = _prepare_audio(
            audio_value["array"], int(audio_value["sampling_rate"]), encoder
        )
        source = torch.from_numpy(audio).view(1, -1).to(device)
        pooled = _pooled_layers(
            extract_features(encoder.model, source, torch_module=torch)
        )
        if by_layer and set(pooled) != set(by_layer):
            raise RuntimeError("encoder returned a different layer set between records")
        for layer, vector in pooled.items():
            by_layer.setdefault(layer, []).append(vector)
        labels.append(row["label"])
    return {
        layer: np.stack(vectors).astype(np.float32)
        for layer, vectors in by_layer.items()
    }, labels


def _classifier(manifest: SplitManifest) -> Any:
    from sklearn.linear_model import LogisticRegression

    cfg = manifest.classifier
    supported = {"name", "C", "class_weight", "max_iter", "solver"}
    unknown = set(cfg) - supported
    if unknown:
        raise ProtocolError(
            f"unsupported canonical classifier fields: {sorted(unknown)}"
        )
    return LogisticRegression(
        C=float(cfg.get("C", 1.0)),
        class_weight=cfg.get("class_weight", "balanced"),
        max_iter=int(cfg.get("max_iter", 2000)),
        solver=str(cfg.get("solver", "lbfgs")),
        random_state=manifest.seed,
        n_jobs=1,
    )


def _evaluation_class_indices(y_eval: Sequence[Any]) -> list[int]:
    return sorted({int(value) for value in y_eval})


def _semantic_class_names(
    encoded_values: Sequence[Any], dataset_label_names: Sequence[str]
) -> list[str]:
    names: list[str] = []
    for value in encoded_values:
        index = int(value)
        if index < 0 or index >= len(dataset_label_names):
            raise ProtocolError(
                f"label ID {value!r} is outside the dataset ClassLabel mapping"
            )
        names.append(str(dataset_label_names[index]))
    return names


def _fit_and_score(
    x_train: Any,
    y_train: Any,
    x_eval: Any,
    y_eval: Any,
    classes: Sequence[str],
    manifest: SplitManifest,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), _classifier(manifest))
    model.fit(x_train, y_train)
    predicted = model.predict(x_eval)
    labels = np.asarray(_evaluation_class_indices(y_eval), dtype=np.int64)
    per_class = f1_score(
        y_eval, predicted, labels=labels, average=None, zero_division=0
    )
    evaluated_classes = [str(classes[int(index)]) for index in labels]
    return {
        "macro_f1": float(
            f1_score(
                y_eval,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_eval, predicted)),
        "evaluated_classes": evaluated_classes,
        "per_class_f1": {
            str(classes[int(class_index)]): float(per_class[position])
            for position, class_index in enumerate(labels)
        },
        "per_class_support": {
            str(classes[int(class_index)]): int(
                (np.asarray(y_eval) == class_index).sum()
            )
            for class_index in labels
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    manifest = load_split_manifest(args.split_manifest)
    seed_everything(manifest.seed)
    records, dataset_label_names = _load_fixed_records(args.data_dir, manifest)
    device = _resolve_device(args.device)
    encoder = load_encoder(args.checkpoint, device, seed=manifest.seed)

    train_x, train_labels = _embed_records(encoder, records["train"], device)
    validation_x, validation_labels = _embed_records(
        encoder, records["validation"], device
    )
    label_encoder = LabelEncoder().fit(train_labels)
    classes = _semantic_class_names(
        label_encoder.classes_, dataset_label_names
    )
    try:
        y_train = label_encoder.transform(train_labels)
        y_validation = label_encoder.transform(validation_labels)
    except ValueError as exc:
        raise ProtocolError(
            "validation contains a class absent from the training split"
        ) from exc

    if set(train_x) != set(validation_x):
        raise RuntimeError("train and validation encoder layer sets differ")

    def score_validation(layer: str) -> float:
        return _fit_and_score(
            train_x[layer],
            y_train,
            validation_x[layer],
            y_validation,
            classes,
            manifest,
        )["macro_f1"]

    def score_test(layer: str) -> dict[str, Any] | None:
        if args.selection_only:
            return None
        test_x, test_labels = _embed_records(encoder, records["test"], device)
        try:
            y_test = label_encoder.transform(test_labels)
        except ValueError as exc:
            raise ProtocolError(
                "test contains a class absent from the training split"
            ) from exc
        x_fit = np.concatenate([train_x[layer], validation_x[layer]], axis=0)
        y_fit = np.concatenate([y_train, y_validation], axis=0)
        return _fit_and_score(
            x_fit, y_fit, test_x[layer], y_test, classes, manifest
        )

    selected, validation_scores, test_metrics = evaluate_selected_layer(
        sorted(train_x), score_validation, score_test
    )
    result = {
        "schema_version": 1,
        "protocol": {
            "id": manifest.protocol_id,
            "mask": False,
            "pooling": "mean",
            "layer_selection": "validation",
            "selected_layer": selected,
            "audio_preprocessing": {
                "sample_rate": "checkpoint_config",
                "max_sample_size": "checkpoint_config",
                "crop": "center",
                "short_input_padding": "only_below_400_samples",
                "resampling": "scipy.signal.resample_poly",
                "normalization": "checkpoint_config",
            },
            "seed": manifest.seed,
            "classifier": dict(manifest.classifier),
            "selection_only": bool(args.selection_only),
        },
        "selection": {
            "metric": "macro_f1",
            "validation_by_layer": validation_scores,
            "selected_macro_f1": validation_scores[selected],
        },
        "test": test_metrics,
        "provenance": build_provenance(
            repo_root=REPO_ROOT,
            checkpoint_path=args.checkpoint,
            checkpoint_metadata=encoder.checkpoint_metadata,
            config_sha256=encoder.config_sha256,
            split_manifest_path=args.split_manifest,
            split_manifest=manifest,
            stripped_config_keys=encoder.stripped_config_keys,
            ignored_state_keys=encoder.ignored_state_keys,
            device=device,
        ),
    }
    write_portable_json(args.output, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="select on validation without evaluating the held-out test split",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args)
    selected = result["protocol"]["selected_layer"]
    validation_score = result["selection"]["selected_macro_f1"]
    print(f"selected {selected} on validation macro-F1={validation_score:.4f}")
    if result["test"] is not None:
        print(f"held-out test macro-F1={result['test']['macro_f1']:.4f}")
    print(f"wrote {args.output}")
