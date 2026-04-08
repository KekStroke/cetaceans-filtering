from __future__ import annotations

import json
import random

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .data import (
    build_split,
    load_embeddings,
    load_manifest,
    split_files,
    unique_files_with_labels,
)
from .metrics import evaluate_split
from .models import build_model
from .reporting import create_report_images, save_json, save_split_manifest


def run_training(args) -> None:
    seed = int(args.random_state)
    random.seed(seed)
    np.random.seed(seed)

    X = load_embeddings(args.embeddings)
    manifest = load_manifest(args.manifest, args)
    if len(manifest) != len(X):
        raise ValueError(
            f"Manifest rows ({len(manifest)}) do not match embeddings rows ({len(X)})."
        )

    file_df = unique_files_with_labels(manifest)
    train_files, val_files, test_files = split_files(
        file_df=file_df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    encoder = LabelEncoder()
    encoder.fit(file_df["label"].to_numpy())
    train_split = build_split("train", manifest, X, train_files, encoder)
    val_split = build_split("val", manifest, X, val_files, encoder)
    test_split = build_split("test", manifest, X, test_files, encoder)
    if len(train_split.indices) == 0:
        raise ValueError("Training split is empty. Adjust test/val sizes.")

    save_split_manifest(train_split, args.output_dir)
    save_split_manifest(val_split, args.output_dir)
    save_split_manifest(test_split, args.output_dir)

    model = build_model(args)
    model.fit(train_split.X, train_split.y)
    metrics = {
        "config": {
            "embeddings": str(args.embeddings),
            "manifest": str(args.manifest),
            "test_size": args.test_size,
            "val_size": args.val_size,
            "random_state": args.random_state,
            "max_iter": args.max_iter,
            "filename_col": args.filename_col,
            "start_col": args.start_col,
            "end_col": args.end_col,
            "label_col": args.label_col,
            "label_regex": args.label_regex,
            "scale": not args.no_scale,
            "aggregation_method": args.aggregation_method,
            "generate_report_images": args.generate_report_images,
        },
        "classes": encoder.classes_.tolist(),
        "split_counts": {
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
            "train_windows": len(train_split.indices),
            "val_windows": len(val_split.indices),
            "test_windows": len(test_split.indices),
        },
        "train": evaluate_split(train_split, model, encoder),
        "val": evaluate_split(val_split, model, encoder)
        if len(val_split.indices)
        else None,
        "test": evaluate_split(test_split, model, encoder)
        if len(test_split.indices)
        else None,
    }

    joblib.dump(
        {"model": model, "label_encoder": encoder}, args.output_dir / "model.joblib"
    )
    save_json(metrics, args.output_dir / "metrics.json")

    summary = {
        "selected_file_aggregation_method": args.aggregation_method,
        "model_type": "logreg",
        "train_window_macro_f1": metrics["train"]["window"]["macro_f1"],
        "train_file_mean_macro_f1": metrics["train"]["file"]["mean"]["macro_f1"],
        "train_file_max_macro_f1": metrics["train"]["file"]["max"]["macro_f1"],
        "train_file_selected_macro_f1": metrics["train"]["file"][
            args.aggregation_method
        ]["macro_f1"],
    }
    for split_name in ("val", "test"):
        split_metrics = metrics.get(split_name)
        if split_metrics is None:
            continue
        summary[f"{split_name}_window_macro_f1"] = split_metrics["window"]["macro_f1"]
        summary[f"{split_name}_file_mean_macro_f1"] = split_metrics["file"]["mean"][
            "macro_f1"
        ]
        summary[f"{split_name}_file_max_macro_f1"] = split_metrics["file"]["max"][
            "macro_f1"
        ]
        summary[f"{split_name}_file_selected_macro_f1"] = split_metrics["file"][
            args.aggregation_method
        ]["macro_f1"]

    if args.generate_report_images:
        summary["report_images"] = create_report_images(
            metrics=metrics,
            class_names=encoder.classes_.tolist(),
            output_dir=args.output_dir,
            primary_file_method=args.aggregation_method,
        )
    save_json(summary, args.output_dir / "summary.json")

    print("Saved model to:", args.output_dir / "model.joblib")
    print("Saved metrics to:", args.output_dir / "metrics.json")
    print("Saved summary to:", args.output_dir / "summary.json")
    if args.generate_report_images:
        print("Saved report images to:", args.output_dir / "report_images")
    print("\nClass labels:")
    print(", ".join(encoder.classes_.tolist()))
    print("\nSplit counts:")
    print(json.dumps(metrics["split_counts"], indent=2))
    print("\nMacro-F1 summary:")
    print(json.dumps(summary, indent=2))
