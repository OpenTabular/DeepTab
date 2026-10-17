"""Release publishing must depend on validation of the distributed package."""

from pathlib import Path
from shlex import split
from unittest.mock import MagicMock

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


@pytest.mark.parametrize(
    "directory,marker",
    [
        ("integration", "integration"),
        ("contracts", "contract"),
        ("learnability", "learnability"),
        ("lifecycle", "lifecycle"),
        ("metrics/oracles", "metric_oracle"),
        ("boundaries", "boundary"),
        ("environments", "environment"),
    ],
)
def test_category_collection_marks_nested_files(pytestconfig, directory, marker):
    root = WORKFLOWS.parents[1] / "tests"
    plugin = pytestconfig.pluginmanager.get_plugin(str(root / "conftest.py"))
    assert plugin is not None
    item = MagicMock()
    item.path = root / directory / "nested" / "test_behavior.py"
    plugin.pytest_collection_modifyitems([item])
    item.add_marker.assert_called_once()
    assert item.add_marker.call_args.args[0].name == marker
    assert any(value.startswith(f"{marker}:") for value in pytestconfig.getini("markers"))


def _workflow(name):
    workflow = yaml.safe_load((WORKFLOWS / name).read_text())
    if True in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


@pytest.mark.parametrize(
    "name,publisher",
    [("publish-pypi.yml", "publish"), ("publish-testpypi.yml", "publish-rc")],
)
def test_publishing_requires_successful_release_checks(name, publisher):
    workflow = _workflow(name)
    jobs = workflow["jobs"]
    job = jobs[publisher]
    assert set(job["needs"]) == {"ci", "docs", "package"}
    assert workflow["permissions"] == {"contents": "read"}
    expected_condition = "${{ !contains(github.ref_name, 'rc') }}" if publisher == "publish" else None
    assert job.get("if") == expected_condition
    assert jobs["ci"]["needs"] == "package"
    assert jobs["ci"]["with"] == {"distribution-artifact": "dist"}
    for gate, target in [("ci", "ci.yml"), ("docs", "docs.yml"), ("package", "build-check.yml")]:
        assert jobs[gate]["uses"] == f"./.github/workflows/{target}"
        assert jobs[gate].get("if") == expected_condition
        assert jobs[gate].get("continue-on-error", "false") == "false"
        assert "workflow_call" in _workflow(target)["on"]


@pytest.mark.parametrize(
    "name,publisher",
    [("publish-pypi.yml", "publish"), ("publish-testpypi.yml", "publish-rc")],
)
def test_publishers_use_validated_artifacts_without_rebuilding(name, publisher):
    job = _workflow(name)["jobs"][publisher]
    download = job["steps"][0]
    assert download["uses"] == "actions/download-artifact@v4"
    assert download["with"] == {"name": "dist", "path": "dist/"}
    assert not any("run" in step for step in job["steps"])
    assert job["steps"][1]["uses"] == "pypa/gh-action-pypi-publish@release/v1"
    assert job["steps"][-1]["with"]["files"] == "dist/*"
    assert job["permissions"] == {"contents": "write", "id-token": "write"}


def test_package_gate_validates_tag_and_installed_wheel_before_upload():
    steps = _workflow("build-check.yml")["jobs"]["build-check"]["steps"]
    names = [step["name"] for step in steps]
    tag = names.index("Check release tag matches package version")
    build = names.index("Build wheel and sdist")
    check = names.index("Check package (twine)")
    wheel = names.index("Test installed-wheel training and reload")
    upload = names.index("Upload build artifacts")
    assert tag < build < check < wheel < upload
    assert steps[tag]["if"] == "startsWith(github.ref, 'refs/tags/v')"
    assert "${GITHUB_REF_NAME#v}" in steps[tag]["run"]
    assert "exit 1" in steps[tag]["run"]
    command = steps[wheel]["run"]
    assert 'python -m venv "$RUNNER_TEMP/deeptab-wheel-test"' in command
    assert "-m pip install dist/*.whl" in command
    assert 'cd "$RUNNER_TEMP"' in command
    assert "-I" in command
    assert '"$GITHUB_WORKSPACE/.github/scripts/check_installed_wheel.py"' in command
    assert steps[upload]["with"]["name"] == "dist"
    assert steps[upload]["with"]["path"] == "dist/"
    assert steps[upload]["with"]["if-no-files-found"] == "error"


def test_reusable_ci_includes_all_existing_quality_checks():
    jobs = _workflow("ci.yml")["jobs"]
    for name in ["lint", "typecheck", "build", "tests", "tests-extras", "smoke", "coverage"]:
        assert name in jobs
        assert jobs[name].get("continue-on-error", "false") == "false"
    assert jobs["tests"]["strategy"]["matrix"] == {
        "os": ["ubuntu-latest", "windows-latest", "macos-latest"],
        "python-version": ["3.11", "3.12", "3.13"],
    }


@pytest.mark.parametrize(
    "name,job",
    [
        ("ci.yml", "lint"),
        ("ci.yml", "typecheck"),
        ("ci.yml", "build"),
        ("build-check.yml", "build-check"),
        ("publish-testpypi.yml", "smoke-test-testpypi"),
    ],
)
def test_release_checks_use_minimum_supported_python(name, job):
    steps = _workflow(name)["jobs"][job]["steps"]
    setup = next(step for step in steps if step.get("uses") == "actions/setup-python@v5")
    assert setup["with"]["python-version"] == "3.11"
    for step in steps:
        if step.get("uses") == "actions/cache@v4":
            assert "-3.11-" in step["with"]["key"]


def test_ci_triggers_select_main_and_preserve_release_invocations():
    triggers = _workflow("ci.yml")["on"]
    assert {"workflow_call", "workflow_dispatch"}.issubset(triggers)
    for event in ["push", "pull_request"]:
        assert triggers[event]["branches"] == ["main"]


@pytest.mark.parametrize("job,profile", [("tests", "core"), ("tests-extras", "extras"), ("coverage", "extras")])
def test_ci_dependency_profiles_are_provisioned_and_verified(job, profile):
    steps = _workflow("ci.yml")["jobs"][job]["steps"]
    installation = next(step["run"] for step in steps if step.get("name") == "Install dependencies")
    arguments = split(installation)
    assert arguments[:4] == ["poetry", "sync", "--only", "main,dev"]
    if profile == "extras":
        assert arguments[4:] == ["--extras", "logs tracking"]
    else:
        assert len(arguments) == 4
    command = next(step["run"] for step in steps if "pytest" in step.get("run", ""))
    test_arguments = split(command.replace("\\\n", ""))
    assert "tests/" in test_arguments
    assert f"--dependency-profile={profile}" in test_arguments
    assert "--strict-markers" in test_arguments
    assert "-m" not in test_arguments
    cache = next(step["with"]["key"] for step in steps if step.get("name") == "Cache virtualenv")
    if job != "coverage":
        assert f"tests-{profile}-" in cache


def test_coverage_requires_locked_profiles_and_dependency_endpoints():
    jobs = _workflow("ci.yml")["jobs"]
    assert set(jobs["coverage"]["needs"]) == {"tests", "tests-extras", "dependency-endpoints"}


def test_dependency_endpoints_preserve_matrix_artifact_and_failure_gates():
    jobs = _workflow("ci.yml")["jobs"]
    job = jobs["dependency-endpoints"]
    assert job["needs"] == "build"
    assert job.get("continue-on-error", False) is False
    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"] == {
        "os": ["ubuntu-latest", "windows-latest", "macos-latest"],
        "profile": ["core", "extras"],
        "lane": [{"endpoint": "minimum", "python": "3.11"}, {"endpoint": "newest", "python": "3.13"}],
    }
    steps = job["steps"]
    download = next(step for step in steps if step.get("uses") == "actions/download-artifact@v4")
    build_upload = next(step for step in jobs["build"]["steps"] if step.get("uses") == "actions/upload-artifact@v4")
    assert build_upload["with"]["name"] == "candidate-dist"
    assert download["with"]["name"] == "${{ inputs.distribution-artifact || 'candidate-dist' }}"
    assert _workflow("ci.yml")["on"]["workflow_call"]["inputs"]["distribution-artifact"]["type"] == "string"
    command = next(step["run"] for step in steps if step.get("name") == "Verify independent dependency endpoint")
    assert "dependency_endpoints.py" in command and "--wheel candidate-dist" in command
    assert "--provision-only" not in command
    assert "poetry" not in command
    upload = steps[-1]
    assert upload["if"] == "always()"
    assert upload["with"]["if-no-files-found"] == "error"
    assert "${{ github.sha }}" in upload["with"]["name"]


def test_dependency_endpoint_cache_uses_step_runner_context():
    job = _workflow("ci.yml")["jobs"]["dependency-endpoints"]
    assert "UV_CACHE_DIR" not in job.get("env", {})
    assert job["env"]["UV_PYTHON_DOWNLOADS"] == "never"
    verification = next(step for step in job["steps"] if step.get("name") == "Verify independent dependency endpoint")
    assert verification["env"]["UV_CACHE_DIR"] == "${{ runner.temp }}/uv-cache"


@pytest.mark.parametrize("profile", ["core", "extras"])
def test_just_dependency_profiles_match_ci(profile):
    lines = (WORKFLOWS.parents[1] / "justfile").read_text().splitlines()
    installation = lines[lines.index(f"install-{profile}:") + 1].strip()
    ci_job = "tests" if profile == "core" else "tests-extras"
    steps = _workflow("ci.yml")["jobs"][ci_job]["steps"]
    assert installation == next(step["run"] for step in steps if step.get("name") == "Install dependencies")
    command = lines[lines.index(f"test-{profile}:") + 1].strip()
    arguments = split(command)
    assert "tests/" in arguments
    assert f"--dependency-profile={profile}" in arguments
    assert "--strict-markers" in arguments


@pytest.mark.parametrize("source", ["ci", "just"])
def test_coverage_commands_enforce_ninety_percent_line_coverage(source):
    if source == "ci":
        steps = _workflow("ci.yml")["jobs"]["coverage"]["steps"]
        command = next(step["run"] for step in steps if step.get("name") == "Run tests with coverage")
    else:
        lines = (WORKFLOWS.parents[1] / "justfile").read_text().splitlines()
        command = lines[lines.index("test:") + 1].strip()
    arguments = split(command.replace("\\\n", ""))
    assert "--cov=deeptab" in arguments
    assert "--cov-fail-under=90" in arguments
    assert "--cov-branch" not in arguments
