"""Freeze dist/facemesh from a fresh environment holding only FaceMesh and its runtime dependencies.

Building outside the development environment keeps development tools out of the release, and
THIRD_PARTY_LICENSES.txt is written from that same environment.
"""

from __future__ import annotations

import shutil
import subprocess
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE_ENV = ROOT / "build" / "release-env"
RELEASE_PYTHON = RELEASE_ENV / "Scripts" / "python.exe"
DIST_DIR = ROOT / "dist" / "facemesh"


def run(*args) -> None:
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def main() -> int:
    shutil.rmtree(RELEASE_ENV, ignore_errors=True)
    venv.create(RELEASE_ENV, with_pip=True)
    run(RELEASE_PYTHON, "-m", "pip", "install", "--quiet", ".[release]")
    run(RELEASE_PYTHON, "-c", "from facemesh_app.face_landmarker import ensure_bundle; ensure_bundle()")
    shutil.rmtree(DIST_DIR, ignore_errors=True)
    run(RELEASE_PYTHON, "-m", "cx_Freeze", "build")
    run(RELEASE_PYTHON, ROOT / "scripts" / "third_party_licenses.py", DIST_DIR)
    print(f"built {DIST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
