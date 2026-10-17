"""Resolve and verify dependency endpoints from a built wheel's metadata."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


def runtime_requirements(wheel: Path, profile: str, environment: dict[str, str] | None = None) -> list[str]:
    if profile not in {"core", "extras", "tabr"}:
        raise ValueError(f"Unknown dependency profile: {profile}")
    with ZipFile(wheel) as archive:
        paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(paths) != 1:
            raise ValueError("Expected exactly one wheel metadata record")
        metadata = BytesParser().parsebytes(archive.read(paths[0]))
    selected = set()
    extras = {""}
    if profile in {"extras", "tabr"}:
        extras.update({"logs", "tracking"})
    if profile == "tabr":
        extras.add("tabr")
    target = default_environment() if environment is None else environment
    for value in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(value)
        if requirement.marker is None or any(
            requirement.marker.evaluate({**target, "extra": extra}) for extra in extras
        ):
            requirement.marker = None
            selected.add(str(requirement))
    return sorted(selected)


def validate_stable_roots(requirements: list[str], versions: dict[str, str]) -> None:
    for value in requirements:
        name = canonicalize_name(Requirement(value).name)
        if Version(versions[name]).is_prerelease:
            raise RuntimeError(f"Expected a stable direct runtime dependency: {name}=={versions[name]}")


def run(command: list[str], log: Path, cwd: Path | None = None) -> None:
    with log.open("w") as output:
        result = subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, check=False)  # noqa: S603
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; see {log}: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--endpoint", choices=["minimum", "newest"], required=True)
    parser.add_argument("--profile", choices=["core", "extras", "tabr"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provision-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new endpoint evidence directory: {output}")
    output.mkdir(parents=True)
    wheel = args.wheel.resolve()
    if wheel.is_dir():
        (wheel,) = wheel.glob("*.whl")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Provision uv before running endpoint validation")
    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "endpoint": args.endpoint,
        "profile": args.profile,
        "platform": platform.platform(),
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "device": "cpu",
        "accelerator_validation": False,
        "status": "running",
        "commands": [],
    }
    environment = Path(tempfile.mkdtemp(prefix="deeptab-endpoint-", dir=os.environ.get("TMPDIR")))
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def execute(command: list[str], name: str, cwd: Path | None = None) -> None:
        manifest["commands"].append({"command": command, "log": name})
        run(command, output / name, cwd)

    try:
        checkout = Path(__file__).resolve().parents[2]
        os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        execute([uv, "--version"], "resolver.txt")
        execute(["git", "rev-parse", "HEAD"], "revision.txt", checkout)
        execute(["git", "diff", "--binary", "HEAD"], "worktree.patch", checkout)
        files = [
            path
            for root in ("deeptab", "tests", ".github")
            for path in (checkout / root).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ]
        files.extend(checkout / name for name in ("pyproject.toml", "poetry.lock", "justfile"))
        (output / "sources.json").write_text(
            json.dumps(
                {
                    str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(files)
                },
                indent=2,
            )
            + "\n"
        )
        execute([uv, "venv", "--python", args.python, str(environment)], "venv.log")
        code = (
            "import json,os,platform,sys; print(json.dumps({"
            "'python_version':platform.python_version().rsplit('.',1)[0],"
            "'python_full_version':platform.python_version(),"
            "'sys_platform':sys.platform,'os_name':os.name,"
            "'platform_machine':platform.machine(),'platform_system':platform.system(),"
            "'platform_release':platform.release(),'platform_version':platform.version(),"
            "'platform_python_implementation':platform.python_implementation(),"
            "'implementation_name':sys.implementation.name,"
            "'implementation_version':platform.python_version()}))"
        )
        execute([str(python), "-I", "-c", code], "interpreter.json")
        target = json.loads((output / "interpreter.json").read_text())
        manifest["interpreter"] = target
        manifest["environment"] = str(environment)
        requirements = runtime_requirements(wheel, args.profile, target)
        extras = {"core": "", "extras": "[logs,tracking]", "tabr": "[logs,tracking,tabr]"}[args.profile]
        artifact = f"deeptab{extras} @ {wheel.as_uri()}"
        inputs = output / "runtime.in"
        inputs.write_text("\n".join([artifact, *requirements]) + "\n")
        lock = output / "runtime.txt"
        backend = ["--torch-backend", "cpu"] if target["platform_system"] == "Linux" else []
        execute(
            [
                uv,
                "pip",
                "compile",
                str(inputs),
                "--python",
                str(python),
                "--resolution",
                "lowest-direct" if args.endpoint == "minimum" else "highest",
                "--refresh",
                "--prerelease",
                "if-necessary-or-explicit",
                "--only-binary",
                ":all:",
                "--default-index",
                "https://pypi.org/simple",
                "--generate-hashes",
                "--emit-index-url",
                "--emit-find-links",
                "--output-file",
                str(lock),
                *backend,
            ],
            "resolution.log",
        )
        if backend:
            with lock.open("a") as stream:
                stream.write("\n--extra-index-url https://download.pytorch.org/whl/cpu\n")
        execute(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--default-index",
                "https://pypi.org/simple",
                "pip==26.2.1",
            ],
            "pip-bootstrap.log",
        )
        pip = [str(python), "-m", "pip", "--isolated"]
        execute(
            [
                *pip,
                "install",
                "--only-binary",
                ":all:",
                "--require-hashes",
                "--report",
                str(output / "runtime-report.json"),
                "-r",
                str(lock),
            ],
            "runtime-install.log",
        )
        code = (
            "import importlib.metadata as metadata,json; from packaging.utils import canonicalize_name; print(json.dumps({"
            "canonicalize_name(distribution.metadata['Name']):{'version':distribution.version,"
            "'location':str(distribution.locate_file('')),'direct_url':distribution.read_text('direct_url.json')}"
            " for distribution in metadata.distributions()},indent=2))"
        )
        execute([str(python), "-I", "-c", code], "runtime-before.json")
        before = json.loads((output / "runtime-before.json").read_text())
        constraints = output / "runtime-constraints.txt"
        constraints.write_text(
            "".join(f"{name}=={distribution['version']}\n" for name, distribution in sorted(before.items()))
        )
        execute(
            [
                *pip,
                "install",
                "--index-url",
                "https://pypi.org/simple",
                "--constraint",
                str(constraints),
                "--report",
                str(output / "tooling-report.json"),
                "pytest>=9,<10",
                "pytest-cov>=4.1,<5",
            ],
            "test-tooling.log",
        )
        execute([str(python), "-I", "-c", code], "installed.json")
        installed = json.loads((output / "installed.json").read_text())
        before = json.loads((output / "runtime-before.json").read_text())
        for name, distribution in before.items():
            if installed[name]["version"] != distribution["version"]:
                raise RuntimeError(f"Test tooling replaced a runtime pin: {name}")
        validate_stable_roots(requirements, {name: distribution["version"] for name, distribution in installed.items()})
        execute([*pip, "check"], "pip-check.log")
        execute(
            [
                str(python),
                "-I",
                "-c",
                "import numpy,pandas,scipy,sklearn,torch,lightning,deeptab; import numpy as np; torch.from_numpy(np.ones(2,dtype=np.float32)); print('ABI imports and NumPy/Torch conversion passed')",
            ],
            "abi.log",
        )
        manifest["status"] = "provisioned"
        if not args.provision_only:
            execute(
                [str(python), "-I", str(checkout / ".github/scripts/check_installed_wheel.py"), "--wheel", str(wheel)],
                "installed-wheel.log",
                output,
            )
            code = (
                "from pathlib import Path; import deeptab,pytest,sys,torch; "
                "assert Path(deeptab.__file__).resolve().is_relative_to(Path.cwd()),'Expected source checkout'; "
                f"assert deeptab.__version__=={installed['deeptab']['version']!r},'Source/wheel version mismatch'; "
                "torch.set_num_threads(1); torch.cuda.is_available=lambda:False; "
                "torch.backends.mps.is_available=lambda:False; raise SystemExit(pytest.main(sys.argv[1:]))"
            )
            execute(
                [
                    str(python),
                    "-c",
                    code,
                    "tests/",
                    "--strict-markers",
                    f"--dependency-profile={args.profile}",
                    "--cov=deeptab",
                    "--cov-fail-under=90",
                    f"--cov-report=xml:{output / 'coverage.xml'}",
                    f"--junitxml={output / 'junit.xml'}",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--tb=short",
                ],
                "source-tests.log",
                checkout,
            )
            manifest["status"] = "passed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        manifest["evidence_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
            if path.is_file()
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
