#!/usr/bin/env python3
"""Pre-push hook: fails if __version__ wasn't bumped relative to remote main."""
import subprocess
import sys
from pathlib import Path

VERSION_FILE = "backend/__version__.py"


def _read_version(source: str) -> str:
    ns: dict = {}
    exec(source, ns)  # noqa: S102
    return ns["__version__"]


def get_local_version() -> str:
    return _read_version(Path(VERSION_FILE).read_text())


def get_remote_version() -> str | None:
    result = subprocess.run(
        ["git", "show", f"origin/main:{VERSION_FILE}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return _read_version(result.stdout)


def parse_semver(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.split("."))


def main() -> int:
    local = get_local_version()
    remote = get_remote_version()

    if remote is None:
        print(f"[version-gate] No remote version found — first push. Local: {local}")
        return 0

    if parse_semver(local) <= parse_semver(remote):
        print(
            f"[version-gate] FAIL: bump __version__ before pushing.\n"
            f"  remote/main : {remote}\n"
            f"  local       : {local}"
        )
        return 1

    print(f"[version-gate] OK: {remote} → {local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
