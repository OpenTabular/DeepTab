"""Validate a release tag against its required remote branch history."""

import argparse
import os
import re
import shutil
import subprocess
import sys


def _git(*arguments: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("Git is required for release validation.")
    return subprocess.check_output([executable, *arguments], text=True).strip()  # noqa: S603


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=("stable", "rc"), required=True)
    args = parser.parse_args()
    ref = os.environ.get("GITHUB_REF", "")
    event_sha = os.environ.get("GITHUB_SHA", "")
    tag = ref.removeprefix("refs/tags/")
    release = re.fullmatch(
        r"v((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))(rc[1-9][0-9]*)?",
        tag,
    )
    if not ref.startswith("refs/tags/") or release is None or bool(release.group(2)) != (args.channel == "rc"):
        print(f"Invalid {args.channel} release tag: {ref!r}.", file=sys.stderr)
        return 1
    if not event_sha:
        print("Missing GITHUB_SHA for the release event.", file=sys.stderr)
        return 1
    if re.fullmatch(r"[0-9a-f]{40}", event_sha) is None:
        print("Invalid GITHUB_SHA for the release event.", file=sys.stderr)
        return 1

    branch = "main" if args.channel == "stable" else f"release/v{release.group(1)}"
    remote_ref = f"refs/remotes/origin/{branch}"
    try:
        tag_commit = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
        event_commit = _git("rev-parse", "--verify", f"{event_sha}^{{commit}}")
        checkout_commit = _git("rev-parse", "HEAD")
        if tag_commit != event_commit or checkout_commit != event_commit:
            raise ValueError("Tag, checked-out commit, and event commit must match.")
        _git("fetch", "--no-tags", "origin", f"+refs/heads/{branch}:{remote_ref}")
        try:
            _git("merge-base", "--is-ancestor", tag_commit, remote_ref)
        except subprocess.CalledProcessError as error:
            raise ValueError(f"Tag {tag} is not reachable from origin/{branch}.") from error
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Release validation failed: {error}", file=sys.stderr)
        return 1
    print(f"Validated {tag}: {tag_commit} is reachable from origin/{branch}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
