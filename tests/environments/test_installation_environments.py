"""Validate pre-provisioned dependencies and portable core fitting."""

import importlib.util
import json
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
from packaging.specifiers import SpecifierSet

from deeptab.configs import MLPConfig, TrainerConfig
from deeptab.core.optional_deps import require_mlflow, require_structlog, require_tensorboard
from deeptab.models import MLPClassifier


@pytest.mark.parametrize(
    "version,supported",
    [("3.10.19", False), ("3.11.0", True), ("3.12.0", True), ("3.13.0", True), ("3.14.0", False)],
)
def test_python_support_metadata(version, supported):
    path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    assert config["project"]["requires-python"] == ">=3.11,<3.14"
    assert (version in SpecifierSet(config["project"]["requires-python"])) is supported
    assert config["tool"]["pyright"]["pythonVersion"] == "3.11"
    assert config["tool"]["ruff"]["target-version"] == "py311"
    for dependency in config["tool"]["poetry"]["group"]["docs"]["dependencies"].values():
        assert dependency["python"] == ">=3.12,<3.14"


@pytest.mark.parametrize("profile", ["core", "extras"])
@pytest.mark.parametrize("system", ["linux", "win32"])
def test_endpoint_requirements_follow_wheel_markers(tmp_path, profile, system):
    path = Path(__file__).resolve().parents[2] / ".github/scripts/dependency_endpoints.py"
    spec = importlib.util.spec_from_file_location("dependency_endpoints", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wheel = tmp_path / "example.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            "example-1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: example\nVersion: 1\n"
            "Requires-Dist: numpy>=2,<3\n"
            "Requires-Dist: structlog>=24; extra == 'logs'\n"
            "Requires-Dist: mlflow>=2.10; extra == 'tracking'\n"
            "Requires-Dist: colorama; sys_platform == 'win32'\n\n",
        )
    result = module.runtime_requirements(wheel, profile, {"sys_platform": system})
    assert "numpy<3,>=2" in result
    assert ("structlog>=24" in result) == (profile == "extras")
    assert ("mlflow>=2.10" in result) == (profile == "extras")
    assert ("colorama" in result) == (system == "win32")
    assert all(";" not in requirement for requirement in result)
    versions = {module.canonicalize_name(module.Requirement(value).name): "2.0.0" for value in result}
    versions["opentelemetry-semantic-conventions"] = "0.66b1"
    module.validate_stable_roots(result, versions)
    versions["numpy"] = "2.5.0rc1"
    with pytest.raises(RuntimeError, match="stable direct runtime dependency: numpy"):
        module.validate_stable_roots(result, versions)
    with pytest.raises(ValueError, match="Unknown dependency profile"):
        module.runtime_requirements(wheel, "unknown")


@pytest.mark.parametrize("mismatch", ["version", "origin", "hash", "import"])
def test_installed_wheel_rejects_identity_mismatch(tmp_path, monkeypatch, mismatch):
    path = Path(__file__).resolve().parents[2] / ".github/scripts/check_installed_wheel.py"
    spec = importlib.util.spec_from_file_location("check_installed_wheel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wheel = tmp_path / "deeptab.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("deeptab-1.dist-info/METADATA", "Metadata-Version: 2.1\nName: deeptab\nVersion: 1\n\n")
    package = tmp_path / "site-packages/deeptab/__init__.py"
    origin = {
        "url": wheel.as_uri(),
        "archive_info": {"hashes": {"sha256": module.hashlib.sha256(wheel.read_bytes()).hexdigest()}},
    }
    if mismatch == "origin":
        origin["url"] = (tmp_path / "other.whl").as_uri()
    if mismatch == "hash":
        origin["archive_info"]["hashes"]["sha256"] = "0" * 64
    installed = SimpleNamespace(
        version="2" if mismatch == "version" else "1",
        locate_file=lambda name: tmp_path / "other.py" if mismatch == "import" else package,
        read_text=lambda name: json.dumps(origin),
    )
    monkeypatch.setattr(module, "deeptab", SimpleNamespace(__file__=str(package), __version__=installed.version))
    monkeypatch.setattr(module, "distribution", lambda name: installed)
    monkeypatch.setattr("sys.argv", [str(path), "--wheel", str(wheel)])
    with pytest.raises(RuntimeError, match="does not match"):
        module.main()


@pytest.mark.parametrize(
    "package,extra,require",
    [
        ("structlog", "logs", require_structlog),
        ("mlflow", "mlflow", require_mlflow),
        ("tensorboard", "tensorboard", require_tensorboard),
    ],
)
def test_declared_dependency_profile(pytestconfig, tmp_path, package, extra, require):
    profile = pytestconfig.getoption("--dependency-profile")
    if profile is None:
        pytest.skip("Select --dependency-profile=core or extras for an isolated installation check")
    available = importlib.util.find_spec(package) is not None
    assert available == (profile == "extras"), f"{package} availability contradicts {profile} profile"
    if profile == "core":
        with pytest.raises(ImportError, match=re.escape(f"pip install 'deeptab[{extra}]'")):
            require()
    else:
        dependency = require()
        assert dependency is not None
        if package == "tensorboard":
            assert callable(dependency)
            with require_tensorboard()(log_dir=str(tmp_path)) as writer:
                writer.add_hparams({"learning_rate": 0.001}, {"accuracy": 0.75})
            assert any(tmp_path.rglob("events.out.tfevents.*"))
        else:
            assert dependency.__name__ == package


@pytest.mark.integration
def test_default_core_cpu_fit_without_observability(tmp_path):
    generator = np.random.default_rng(17)
    features = generator.normal(size=(48, 3)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(int)
    model = MLPClassifier(
        model_config=MLPConfig(layer_sizes=[8]),
        trainer_config=TrainerConfig(
            max_epochs=2, batch_size=16, patience=1, lr_patience=1, checkpoint_path=str(tmp_path)
        ),
        random_state=17,
    )
    model.fit(
        features[:32],
        labels[:32],
        X_val=features[32:],
        y_val=labels[32:],
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    predictions = model.predict(features[32:])
    probabilities = model.predict_proba(features[32:])
    assert predictions.shape == labels[32:].shape
    assert set(predictions).issubset(set(model.classes_))
    assert probabilities.shape == (16, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
