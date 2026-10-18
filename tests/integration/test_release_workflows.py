"""Release publishing must depend on validation of the distributed package."""

import os
import shutil
import subprocess
import sys
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


def _git(directory, *arguments):
    executable = shutil.which("git")
    assert executable is not None
    return subprocess.check_output(  # noqa: S603
        [executable, "-C", str(directory), *arguments], text=True, stderr=subprocess.STDOUT
    ).strip()


@pytest.fixture
def release_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    remote = tmp_path / "origin.git"
    checkout = tmp_path / "checkout"
    _git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    _git(tmp_path, "clone", str(remote), str(checkout))
    _git(checkout, "config", "user.name", "Release tests")
    _git(checkout, "config", "user.email", "release-tests@example.test")
    _git(checkout, "commit", "--allow-empty", "-m", "Initial commit")
    _git(checkout, "push", "origin", "main")
    _git(checkout, "checkout", "-b", "release/v2.1.0")
    _git(checkout, "commit", "--allow-empty", "-m", "Prepare release")
    _git(checkout, "push", "origin", "release/v2.1.0")
    return checkout


def _validate_release(checkout, tag, channel, *, event_sha=None, ref=None):
    return subprocess.run(  # noqa: S603
        [sys.executable, str(WORKFLOWS.parent / "scripts" / "check_release_ref.py"), "--channel", channel],
        cwd=checkout,
        env=os.environ
        | {
            "GITHUB_REF": ref if ref is not None else f"refs/tags/{tag}",
            "GITHUB_SHA": event_sha if event_sha is not None else _git(checkout, "rev-parse", "HEAD"),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("channel", ["stable", "rc"])
@pytest.mark.parametrize("annotated", [False, True], ids=["lightweight", "annotated"])
@pytest.mark.parametrize("advance_branch", [False, True], ids=["branch-tip", "older-ancestor"])
def test_release_ref_accepts_matching_remote_history(release_repository, channel, annotated, advance_branch):
    branch = "main" if channel == "stable" else "release/v2.1.0"
    tag = "v2.1.0" if channel == "stable" else "v2.1.0rc1"
    _git(release_repository, "checkout", branch)
    if annotated:
        _git(release_repository, "tag", "-a", tag, "-m", "Release test")
    else:
        _git(release_repository, "tag", tag)
    event_sha = _git(release_repository, "rev-parse", tag)
    if advance_branch:
        _git(release_repository, "commit", "--allow-empty", "-m", "Advance branch")
        _git(release_repository, "push", "origin", branch)
    _git(release_repository, "checkout", "--detach", tag)
    result = _validate_release(release_repository, tag, channel, event_sha=event_sha)
    assert result.returncode == 0, result.stderr
    assert f"is reachable from origin/{branch}." in result.stdout


@pytest.mark.parametrize(
    "channel,tag",
    [
        ("stable", "v2.1.0rc1"),
        ("stable", "v2.1.0a1"),
        ("stable", "v2.1.0b1"),
        ("stable", "v2.1.0.dev1"),
        ("stable", "v2.1.0.post1"),
        ("stable", "v02.1.0"),
        ("stable", "v2.1"),
        ("stable", "v2.1.0.1"),
        ("rc", "v2.1.0"),
        ("rc", "v2.1.0rc0"),
        ("rc", "v2.1.0rc01"),
        ("rc", "v2.1.0rc"),
        ("rc", "v2.1.0rc1extra"),
    ],
)
def test_release_ref_rejects_invalid_tag_formats(release_repository, channel, tag):
    result = _validate_release(release_repository, tag, channel)
    assert result.returncode != 0
    assert f"Invalid {channel} release tag" in result.stderr


@pytest.mark.parametrize("channel", ["stable", "rc"])
def test_release_ref_rejects_wrong_branch_history(release_repository, channel):
    tag = "v2.1.0" if channel == "stable" else "v2.1.0rc1"
    if channel == "rc":
        _git(release_repository, "checkout", "-b", "release/v2.2.0")
        _git(release_repository, "commit", "--allow-empty", "-m", "Different release")
        _git(release_repository, "push", "origin", "release/v2.2.0")
    _git(release_repository, "tag", tag)
    result = _validate_release(release_repository, tag, channel)
    assert result.returncode != 0
    branch = "main" if channel == "stable" else "release/v2.1.0"
    assert f"not reachable from origin/{branch}" in result.stderr


@pytest.mark.parametrize("channel", ["stable", "rc"])
def test_release_ref_rejects_deleted_remote_branch_despite_cached_ref(release_repository, channel):
    branch = "main" if channel == "stable" else "release/v2.1.0"
    tag = "v2.1.0" if channel == "stable" else "v2.1.0rc1"
    _git(release_repository, "checkout", branch)
    _git(release_repository, "tag", tag)
    assert _git(release_repository, "rev-parse", f"refs/remotes/origin/{branch}")
    _git(release_repository.parent / "origin.git", "update-ref", "-d", f"refs/heads/{branch}")
    result = _validate_release(release_repository, tag, channel)
    assert result.returncode != 0
    assert "Release validation failed:" in result.stderr


@pytest.mark.parametrize("mismatch", ["checkout", "event", "missing-tag", "tree-tag"])
def test_release_ref_rejects_commit_identity_mismatches(release_repository, mismatch):
    tag = "v2.1.0"
    event_sha = _git(release_repository, "rev-parse", "main")
    if mismatch == "tree-tag":
        _git(release_repository, "tag", tag, "main^{tree}")
    elif mismatch != "missing-tag":
        _git(release_repository, "tag", tag, "main")
    if mismatch != "checkout":
        _git(release_repository, "checkout", "main")
    if mismatch == "event":
        event_sha = _git(release_repository, "rev-parse", "release/v2.1.0")
    result = _validate_release(release_repository, tag, "stable", event_sha=event_sha)
    assert result.returncode != 0
    assert "Release validation failed:" in result.stderr


def test_release_ref_rejects_branch_events(release_repository):
    result = _validate_release(release_repository, "v2.1.0", "stable", ref="refs/heads/v2.1.0")
    assert result.returncode != 0
    assert "Invalid stable release tag" in result.stderr


@pytest.mark.parametrize("event_sha", ["", "main", "--help", "a" * 39])
def test_release_ref_requires_event_commit(release_repository, event_sha):
    result = _validate_release(release_repository, "v2.1.0", "stable", event_sha=event_sha)
    assert result.returncode != 0
    assert "GITHUB_SHA" in result.stderr


@pytest.mark.parametrize(
    "name,publisher",
    [("publish-pypi.yml", "publish"), ("publish-testpypi.yml", "publish-rc")],
)
def test_publishing_requires_successful_release_checks(name, publisher):
    workflow = _workflow(name)
    jobs = workflow["jobs"]
    job = jobs[publisher]
    assert set(job["needs"]) == {"provenance", "ci", "docs", "package"}
    assert workflow["permissions"] == {"contents": "read"}
    expected_condition = "${{ !contains(github.ref_name, 'rc') }}" if publisher == "publish" else None
    assert job.get("if") == expected_condition
    assert set(jobs["ci"]["needs"]) == {"provenance", "package"}
    assert jobs["ci"]["with"] == {"distribution-artifact": "dist"}
    provenance = jobs["provenance"]
    assert provenance.get("if") == expected_condition
    assert provenance.get("continue-on-error", False) is False
    assert provenance["permissions"] == {"contents": "read"}
    checkout, setup, validation = provenance["steps"]
    assert checkout["uses"] == "actions/checkout@v4"
    assert checkout["with"]["fetch-depth"] == 0
    assert setup["uses"] == "actions/setup-python@v5"
    assert setup["with"]["python-version"] == "3.11"
    channel = "stable" if publisher == "publish" else "rc"
    assert validation["run"] == f"python .github/scripts/check_release_ref.py --channel {channel}"
    for gate in ["docs", "package"]:
        assert jobs[gate]["needs"] == "provenance"
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
    for name in ["lint", "typecheck", "build", "tests", "tests-extras", "tests-tabr", "smoke", "coverage"]:
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


@pytest.mark.parametrize(
    "job,profile", [("tests", "core"), ("tests-extras", "extras"), ("tests-tabr", "tabr"), ("coverage", "tabr")]
)
def test_ci_dependency_profiles_are_provisioned_and_verified(job, profile):
    steps = _workflow("ci.yml")["jobs"][job]["steps"]
    installation = next(step["run"] for step in steps if step.get("name") == "Install dependencies")
    arguments = split(installation)
    assert arguments[:4] == ["poetry", "sync", "--only", "main,dev"]
    if profile != "core":
        assert arguments[4:] == ["--extras", "logs tracking tabr" if profile == "tabr" else "logs tracking"]
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
    assert set(jobs["coverage"]["needs"]) == {"tests", "tests-extras", "tests-tabr", "dependency-endpoints"}
    assert jobs["tests-tabr"]["strategy"]["matrix"] == {"os": ["ubuntu-latest", "windows-latest"]}


def test_dependency_endpoints_preserve_matrix_artifact_and_failure_gates():
    jobs = _workflow("ci.yml")["jobs"]
    job = jobs["dependency-endpoints"]
    assert job["needs"] == "build"
    assert job.get("continue-on-error", False) is False
    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"] == {
        "os": ["ubuntu-latest", "windows-latest", "macos-latest"],
        "profile": ["core", "extras", "tabr"],
        "lane": [{"endpoint": "minimum", "python": "3.11"}, {"endpoint": "newest", "python": "3.13"}],
        "exclude": [{"os": "macos-latest", "profile": "tabr"}],
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


@pytest.mark.parametrize("profile", ["core", "extras", "tabr"])
def test_just_dependency_profiles_match_ci(profile):
    lines = (WORKFLOWS.parents[1] / "justfile").read_text().splitlines()
    installation = lines[lines.index(f"install-{profile}:") + 1].strip()
    ci_job = "tests" if profile == "core" else f"tests-{profile}"
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
