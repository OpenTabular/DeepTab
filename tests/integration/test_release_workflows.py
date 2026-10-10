"""Release publishing must depend on validation of the distributed package."""

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


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
    for name in ["lint", "typecheck", "build", "tests", "smoke", "coverage"]:
        assert name in jobs
        assert jobs[name].get("continue-on-error", "false") == "false"
    assert jobs["tests"]["strategy"]["matrix"] == {
        "os": ["ubuntu-latest", "windows-latest", "macos-latest"],
        "python-version": ["3.10", "3.11", "3.12", "3.13"],
    }
