# ruff: noqa: S603
"""Measure fitted representation widths and small CPU training workflows."""

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import warnings
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from lightning.pytorch import Callback
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.core import set_seed
from deeptab.data import TabularDataModule
from deeptab.models import MLPClassifier, MLPRegressor


class TrainingTimer(Callback):
    def on_train_start(self, trainer, pl_module):
        self.started = time.perf_counter()

    def on_train_end(self, trainer, pl_module):
        self.seconds = time.perf_counter() - self.started


def run_case(task, width, seed, epochs):
    torch.set_num_threads(1)
    set_seed(seed)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="lightning")
    dataset = load_breast_cancer(as_frame=True) if task == "classification" else load_diabetes(as_frame=True)
    features = dataset.data.copy()
    if task == "classification":
        features["radius_group"] = np.where(features["mean radius"] >= 15, "large", "small")
    else:
        features["sex_group"] = np.where(features["sex"] > 0, "positive", "negative")
    targets = dataset.target.to_numpy()
    indices = np.arange(len(features))
    development, test = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=targets if task == "classification" else None
    )
    train, validation = train_test_split(
        development,
        test_size=0.25,
        random_state=seed,
        stratify=targets[development] if task == "classification" else None,
    )
    center, scale = (float(targets[train].mean()), float(targets[train].std())) if task == "regression" else (0, 1)
    training_targets = (targets - center) / scale if task == "regression" else targets
    preprocessing = PreprocessingConfig(
        numerical_method="ple" if task == "classification" else "bspline",
        categorical_method="int",
        output_dim=width,
        target_aware=task == "classification",
        placement_strategy="cart" if task == "classification" else "uniform",
    )
    durations = {}
    original = TabularDataModule.preprocess_data

    def timed_preprocessing(datamodule, *args, **kwargs):
        started = time.perf_counter()
        result = original(datamodule, *args, **kwargs)
        durations["preprocessing_seconds"] = time.perf_counter() - started
        return result

    timer = TrainingTimer()
    estimator_class = MLPClassifier if task == "classification" else MLPRegressor
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
        model = estimator_class(
            model_config=MLPConfig(layer_sizes=[32, 16], dropout=0),
            preprocessing_config=preprocessing,
            trainer_config=TrainerConfig(
                max_epochs=epochs,
                batch_size=64,
                lr=1e-3,
                patience=max(1, epochs - 1),
                lr_patience=max(1, epochs - 1),
                scheduler_type=None,
                checkpoint_path=directory,
            ),
            random_state=seed,
        )
        started = time.perf_counter()
        with patch.object(TabularDataModule, "preprocess_data", timed_preprocessing):
            model.fit(
                features.iloc[train],
                training_targets[train],
                X_val=features.iloc[validation],
                y_val=training_targets[validation],
                accelerator="cpu",
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[timer],
            )
        fit_seconds = time.perf_counter() - started
        predictions = model.predict(features.iloc[test])
        np.testing.assert_equal(predictions.shape, (len(test),))
        np.testing.assert_equal(np.isfinite(predictions).all(), True)
        transformed = model._preprocessor.transform(features.iloc[test])
        widths = model._preprocessor.output_dims_
        for column in features:
            prefix = "cat_" if features[column].dtype == object else "num_"
            np.testing.assert_equal(transformed[prefix + column].shape, (len(test), widths[column]))
            np.testing.assert_equal(np.isfinite(transformed[prefix + column]).all(), True)
        if task == "classification":
            metrics = {
                "accuracy": accuracy_score(targets[test], predictions),
                "auroc": roc_auc_score(targets[test], model.predict_proba(features.iloc[test])[:, 1]),
            }
        else:
            predictions = predictions * scale + center
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(targets[test], predictions))),
                "r2": r2_score(targets[test], predictions),
            }
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        artifact = str(Path(directory) / "model.deeptab")
        model.save(artifact)
        restored = estimator_class.load(artifact, device="cpu")
        np.testing.assert_array_equal(model.predict(features.iloc[test]), restored.predict(features.iloc[test]))
        np.testing.assert_equal(restored._preprocessor.output_dims_, widths)
        return {
            "task": task,
            "dataset": "sklearn_breast_cancer" if task == "classification" else "sklearn_diabetes",
            "requested_width": width,
            "effective_widths": widths,
            "seed": seed,
            "split_counts": [len(train), len(validation), len(test)],
            "split_sha256": hashlib.sha256(np.concatenate([train, validation, test]).tobytes()).hexdigest(),
            "epochs": model._trainer.current_epoch,
            "parameters": model.get_number_of_params(),
            "fit_seconds": fit_seconds,
            "training_seconds": timer.seconds,
            "peak_rss_mib": peak / (1024 * 1024 if sys.platform == "darwin" else 1024),
            "metrics": metrics,
            "checks": "schema, categorical, finite outputs, exact CPU save/load",
            **durations,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "regression"])
    parser.add_argument("--width", type=int, choices=[7, 15, 30, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path(os.environ["TMPDIR"]) / "deeptab-dimensions.json")
    args = parser.parse_args()
    if args.task is not None and args.width is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_case(args.task, args.width, args.seed, args.epochs)
        print(json.dumps(result))
        return
    results = []
    for task in ("classification", "regression"):
        for width in (7, 15, 30, 64):
            for seed in (42, 101, 202):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--task",
                    task,
                    "--width",
                    str(width),
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(args.epochs),
                ]
                result = json.loads(subprocess.check_output(command, text=True))
                results.append(result)
                print(task, width, seed, result["metrics"], flush=True)
    report = {
        "commit": subprocess.check_output(["/usr/bin/git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "threads": 1,
        "regression_target_scaling": "training-only mean and standard deviation",
        "preprocessing_timing": "datamodule fitting and train/validation transformations",
        "peak_memory": "fresh-process peak RSS through inference, including imports; excludes reload",
        "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pretab")},
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Results:", args.output)


if __name__ == "__main__":
    main()
