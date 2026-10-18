"""Verify training and artifact reload using an installed distribution."""

import argparse
import hashlib
import json
from email.parser import BytesParser
from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

import deeptab
from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MLPClassifier, MLPRegressor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[2]
    package_path = Path(deeptab.__file__).resolve()
    if package_path.is_relative_to(checkout):
        raise RuntimeError(f"Expected an installed wheel, but imported the checkout at {package_path}")
    installed = distribution("deeptab")
    if (
        installed.version != deeptab.__version__
        or Path(installed.locate_file("deeptab/__init__.py")).resolve() != package_path
    ):
        raise RuntimeError("Imported DeepTab does not match the installed distribution metadata")
    if args.wheel:
        wheel = args.wheel.resolve()
        with ZipFile(wheel) as archive:
            (metadata_path,) = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            metadata = BytesParser().parsebytes(archive.read(metadata_path))
        origin = json.loads(installed.read_text("direct_url.json") or "{}")
        hashes = origin.get("archive_info", {}).get("hashes", {})
        if (
            installed.version != metadata["Version"]
            or origin.get("url") != wheel.as_uri()
            or hashes.get("sha256") != hashlib.sha256(wheel.read_bytes()).hexdigest()
        ):
            raise RuntimeError("Installed DeepTab does not match the requested wheel version, origin and SHA256")
    print(f"Testing installed DeepTab {deeptab.__version__}: {package_path}")
    generator = np.random.default_rng(17)
    features = generator.normal(size=(48, 3)).astype(np.float32)

    with TemporaryDirectory() as root:
        for model_class in (MLPClassifier, MLPRegressor):
            targets = (
                np.tile([10, 20, 30], 16)
                if model_class is MLPClassifier
                else features[:, 0] + 0.1 * generator.normal(size=48)
            )
            for width in (None, 15):
                model = model_class(
                    model_config=MLPConfig(layer_sizes=[8]),
                    preprocessing_config=PreprocessingConfig(numerical_method="bspline", output_dim=width),
                    trainer_config=TrainerConfig(
                        max_epochs=2, batch_size=8, patience=1, lr_patience=1, checkpoint_path=root
                    ),
                    random_state=17,
                )
                model.fit(
                    features[:36],
                    targets[:36],
                    X_val=features[36:],
                    y_val=targets[36:],
                    accelerator="cpu",
                    logger=False,
                    enable_progress_bar=False,
                )
                before = model.predict(features[36:])
                if not np.isfinite(before).all():
                    raise RuntimeError(f"{model_class.__name__} produced nonfinite predictions")
                path = str(Path(root) / f"{model_class.__name__}-{width}.deeptab")
                model.save(path)
                restored = model_class.load(path, device="cpu")
                np.testing.assert_array_equal(restored.predict(features[36:]), before)
                np.testing.assert_array_equal(restored.predict(features[:1]), model.predict(features[:1]))
                if isinstance(model, MLPClassifier):
                    if not isinstance(restored, MLPClassifier):
                        raise RuntimeError("Reloaded classifier has the wrong estimator type")
                    probabilities = model.predict_proba(features[36:])
                    np.testing.assert_array_equal(restored.predict_proba(features[36:]), probabilities)
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                    if not set(before).issubset({10, 20, 30}):
                        raise RuntimeError("Classifier did not restore the original target labels")
                print(f"{model_class.__name__}, output_dim={width}: fit/save/CPU-reload passed")


if __name__ == "__main__":
    main()
