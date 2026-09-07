#!/usr/bin/env python3
"""CPU-only tests for the canonical animal2vec validation contract."""

from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path

from animal2vec.validation.protocol import (
    ProtocolError,
    evaluate_selected_layer,
    load_split_manifest,
    sha256_file,
)
from animal2vec.validation.provenance import (
    assert_portable_payload,
    build_provenance,
    write_portable_json,
)
from animal2vec.validation.runtime import (
    ConfigError,
    extract_features,
    input_spec_from_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeTorch:
    entered = False

    @classmethod
    def inference_mode(cls):
        @contextlib.contextmanager
        def context():
            cls.entered = True
            yield

        return context()


class _FakeModel:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {"x": "features"}


def _manifest_payload(train_hash: str, test_hash: str) -> dict:
    return {
        "schema_version": 1,
        "protocol_id": "watkins-species-linear-probe-v1",
        "dataset": {
            "name": "beans-watkins",
            "revision": "audited-revision",
            "record_id_field": "path.path",
            "artifacts": {
                "train.arrow": train_hash,
                "test.arrow": test_hash,
            },
            "source_partitions": {
                "train": "train.arrow",
                "test": "test.arrow",
            },
        },
        "splits": {
            "train": ["train-a.wav"],
            "validation": ["validation-a.wav"],
            "test": ["test-a.wav"],
        },
        "protocol": {
            "mask": False,
            "pooling": "mean",
            "layer_selection": "validation",
            "selection_metric": "macro_f1",
            "seed": 42,
            "classifier": {
                "name": "logistic_regression",
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 2000,
                "solver": "lbfgs",
            },
        },
    }


class RuntimeTests(unittest.TestCase):
    def test_extract_features_always_disables_masking(self):
        model = _FakeModel()
        result = extract_features(model, "audio", torch_module=_FakeTorch)
        self.assertEqual(result, {"x": "features"})
        self.assertTrue(_FakeTorch.entered)
        self.assertEqual(
            model.kwargs,
            {"source": "audio", "features_only": True, "mask": False},
        )

    def test_input_spec_is_config_derived_for_each_sample_rate(self):
        for sample_rate in (8000, 16000, 32000):
            with self.subTest(sample_rate=sample_rate):
                spec = input_spec_from_config(
                    {
                        "task": {
                            "sample_rate": sample_rate,
                            "max_sample_size": 80000,
                            "normalize": True,
                        }
                    }
                )
                self.assertEqual(spec.sample_rate, sample_rate)
                self.assertEqual(spec.max_sample_size, 80000)
                self.assertAlmostEqual(spec.window_seconds, 80000 / sample_rate)
        nested = input_spec_from_config(
            {
                "model": {
                    "modalities": {
                        "audio": {
                            "sample_rate": 32000,
                            "max_sample_size": 80000,
                        }
                    }
                }
            }
        )
        self.assertEqual(nested.sample_rate, 32000)

    def test_input_spec_rejects_missing_or_conflicting_rate(self):
        with self.assertRaises(ConfigError):
            input_spec_from_config({"task": {"max_sample_size": 80000}})
        with self.assertRaises(ConfigError):
            input_spec_from_config(
                {
                    "task": {"sample_rate": 8000, "max_sample_size": 80000},
                    "model": {"sample_rate": 16000},
                }
            )


class ProtocolTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        train = root / "train.arrow"
        test = root / "test.arrow"
        train.write_bytes(b"fixed train artifact")
        test.write_bytes(b"fixed test artifact")
        manifest = root / "watkins-split-v1.json"
        manifest.write_text(
            json.dumps(_manifest_payload(sha256_file(train), sha256_file(test))),
            encoding="utf-8",
        )
        return train, test, manifest

    def test_manifest_is_disjoint_exact_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, path = self._write_fixture(root)
            manifest = load_split_manifest(path)
            manifest.verify_artifacts(root)
            partition = manifest.partition(
                {
                    "train-a.wav": 1,
                    "validation-a.wav": 2,
                    "test-a.wav": 3,
                }
            )
            self.assertEqual(partition, {"train": [1], "validation": [2], "test": [3]})
            with self.assertRaises(ProtocolError):
                manifest.partition({"train-a.wav": 1, "test-a.wav": 3})

    def test_manifest_rejects_overlap_and_absolute_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test, path = self._write_fixture(root)
            payload = _manifest_payload(sha256_file(train), sha256_file(test))
            payload["splits"]["test"] = ["train-a.wav"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ProtocolError):
                load_split_manifest(path)
            payload["splits"]["test"] = ["C:\\private\\test.wav"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ProtocolError):
                load_split_manifest(path)

    def test_manifest_rejects_noncanonical_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test, path = self._write_fixture(root)
            payload = _manifest_payload(sha256_file(train), sha256_file(test))
            payload["protocol"]["mask"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ProtocolError):
                load_split_manifest(path)

    def test_layer_is_selected_on_validation_and_test_is_called_once(self):
        test_calls = []
        validation = {"L0": 0.95, "L1": 0.60, "final": 0.70}

        def test_evaluator(layer):
            test_calls.append(layer)
            # Test would prefer L1, but it is never consulted during selection.
            return {"L0": 0.10, "L1": 0.99, "final": 0.80}[layer]

        selected, scores, test_score = evaluate_selected_layer(
            ["L0", "L1", "final"], validation.__getitem__, test_evaluator
        )
        self.assertEqual(selected, "L0")
        self.assertEqual(scores, validation)
        self.assertEqual(test_calls, ["L0"])
        self.assertEqual(test_score, 0.10)

    def test_provenance_contains_hashes_but_no_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, manifest_path = self._write_fixture(root)
            manifest = load_split_manifest(manifest_path)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint fixture")
            payload = build_provenance(
                repo_root=REPO_ROOT,
                checkpoint_path=checkpoint,
                checkpoint_metadata={
                    "num_updates": 27000,
                    "epoch": 1,
                    "sample_rate": 16000,
                    "max_sample_size": 80000,
                    "normalize": True,
                },
                config_sha256="a" * 64,
                split_manifest_path=manifest_path,
                split_manifest=manifest,
                stripped_config_keys=[],
                ignored_state_keys=["_ema"],
                device="cpu",
            )
            serialized = json.dumps(payload)
            self.assertNotIn(str(root), serialized)
            self.assertEqual(payload["checkpoint"]["file"], "checkpoint.pt")
            self.assertEqual(payload["checkpoint"]["sha256"], sha256_file(checkpoint))
            output = root / "result.json"
            write_portable_json(output, payload)
            self.assertTrue(output.is_file())
            with self.assertRaises(ProtocolError):
                assert_portable_payload({"checkpoint": "C:\\private\\checkpoint.pt"})


if __name__ == "__main__":
    unittest.main()
