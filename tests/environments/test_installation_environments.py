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
from packaging.markers import Marker
from packaging.requirements import Requirement
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


@pytest.mark.parametrize(
    "package,minimum,older",
    [
        ("torch", "2.13.0", "2.12.0"),
        ("setuptools", "83.0.0", "82.0.1"),
        ("lightning", "2.6.6", "2.6.5"),
        ("accelerate", "1.15.0", "1.14.0"),
    ],
)
def test_security_dependency_minima(package, minimum, older):
    path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    supported = SpecifierSet(config["tool"]["poetry"]["dependencies"][package])
    assert minimum in supported
    assert older not in supported
    for extra in ("tensorboard", "tracking", "all"):
        tensorboard = next(
            Requirement(value)
            for value in config["project"]["optional-dependencies"][extra]
            if Requirement(value).name == "tensorboard"
        )
        assert "2.21.0" in tensorboard.specifier
        assert "2.20.0" not in tensorboard.specifier


def test_readthedocs_html_cleaner_matches_patched_lock():
    root = Path(__file__).resolve().parents[2]
    requirements = [
        Requirement(line)
        for line in (root / "docs" / "requirements_docs.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    cleaner = next(requirement for requirement in requirements if requirement.name == "lxml-html-clean")
    with (root / "poetry.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    version = next(package["version"] for package in lock["package"] if package["name"] == "lxml-html-clean")
    assert version in SpecifierSet(">=0.4.4")
    assert version in cleaner.specifier
    assert "0.4.0" not in cleaner.specifier
    assert "0.4.3" not in cleaner.specifier


def test_tabr_dependencies_are_an_explicit_extra():
    path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    native = {"delu", "faiss-cpu"}
    assert native.isdisjoint(config["tool"]["poetry"]["dependencies"])
    extras = config["project"]["optional-dependencies"]
    assert {Requirement(value).name for value in extras["tabr"]} == native
    for extra in ("logs", "mlflow", "tensorboard", "tracking", "all"):
        assert native.isdisjoint(Requirement(value).name for value in extras[extra])


@pytest.mark.parametrize("extras", [("tensorboard",), ("mlflow",), ("tracking",), ("all",), ("tensorboard", "mlflow")])
def test_locked_observability_protobuf_satisfies_consumers(extras):
    path = Path(__file__).resolve().parents[2] / "poetry.lock"
    with path.open("rb") as stream:
        lock = tomllib.load(stream)
    packages = [
        package
        for package in lock["package"]
        if package["name"] in {"protobuf", "tensorboard", "opentelemetry-proto"}
        and any(Marker(package["markers"]).evaluate({"extra": extra}) for extra in extras)
    ]
    versions = {package["version"] for package in packages if package["name"] == "protobuf"}
    assert len(versions) == 1, f"Conflicting locked protobuf versions for {extras}: {versions}"
    version = versions.pop()
    for package in packages:
        if package["name"] != "protobuf":
            assert version in SpecifierSet(package["dependencies"]["protobuf"])


@pytest.mark.parametrize("profile", ["core", "extras", "tabr"])
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
            "Requires-Dist: faiss-cpu>=1.13; extra == 'tabr'\n"
            "Requires-Dist: delu>=0.0.26; extra == 'tabr'\n"
            "Requires-Dist: colorama; sys_platform == 'win32'\n\n",
        )
    result = module.runtime_requirements(wheel, profile, {"sys_platform": system})
    assert "numpy<3,>=2" in result
    assert ("structlog>=24" in result) == (profile in {"extras", "tabr"})
    assert ("mlflow>=2.10" in result) == (profile in {"extras", "tabr"})
    assert ("faiss-cpu>=1.13" in result) == (profile == "tabr")
    assert ("delu>=0.0.26" in result) == (profile == "tabr")
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


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
@pytest.mark.parametrize("endpoint", ["minimum", "newest"])
def test_endpoint_requirements_retain_cpu_repository(tmp_path, monkeypatch, system, endpoint):
    path = Path(__file__).resolve().parents[2] / ".github/scripts/dependency_endpoints.py"
    spec = importlib.util.spec_from_file_location("dependency_endpoints", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wheel = tmp_path / "deeptab.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            "deeptab-1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: deeptab\nVersion: 1\nRequires-Dist: torch>=2.13\n\n",
        )
    output = tmp_path / "evidence"
    commands = []
    installed = {"deeptab": {"version": "1"}, "torch": {"version": "2.13.0+cpu"}}

    def run(command, log, cwd=None):
        commands.append(command)
        contents = {
            "interpreter.json": {"platform_system": system},
            "runtime-before.json": installed,
            "installed.json": installed,
        }.get(log.name, {})
        log.write_text(json.dumps(contents))
        if log.name == "resolution.log":
            lock = Path(command[command.index("--output-file") + 1])
            lock.write_text("--index-url https://pypi.org/simple\ntorch==2.13.0+cpu --hash=sha256:" + "0" * 64 + "\n")

    monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(module.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "env"))
    monkeypatch.setattr(
        "sys.argv",
        [
            str(path),
            "--wheel",
            str(wheel),
            "--python",
            "3.11",
            "--endpoint",
            endpoint,
            "--profile",
            "core",
            "--output",
            str(output),
            "--provision-only",
        ],
    )
    module.main()
    requirements = (output / "runtime.txt").read_text()
    assert ("--extra-index-url https://download.pytorch.org/whl/cpu" in requirements) == (system == "Linux")
    assert "--index-url https://pypi.org/simple" in requirements
    assert "--hash=sha256:" + "0" * 64 in requirements
    installation = next(command for command in commands if "--require-hashes" in command)
    assert installation[installation.index("-r") + 1] == str(output / "runtime.txt")
    assert json.loads((output / "manifest.json").read_text())["status"] == "provisioned"


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
        pytest.skip("Select --dependency-profile=core, extras, or tabr for an isolated installation check")
    available = importlib.util.find_spec(package) is not None
    assert available == (profile in {"extras", "tabr"}), f"{package} availability contradicts {profile} profile"
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


@pytest.mark.parametrize("package", ["faiss", "delu"])
def test_native_dependency_profile(pytestconfig, package):
    profile = pytestconfig.getoption("--dependency-profile")
    if profile is None:
        pytest.skip("Select a dependency profile for an isolated installation check")
    assert (importlib.util.find_spec(package) is not None) == (profile == "tabr")


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
