"""Zip the PyInstaller --onedir output for distribution."""
from __future__ import annotations

import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist" / "facemesh"


def main() -> int:
    if not DIST_DIR.is_dir():
        print(f"error: {DIST_DIR} not found; run `task build-exe` first", file=sys.stderr)
        return 1

    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    out = ROOT / "dist" / f"facemesh-{version}-win64.zip"

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in DIST_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(DIST_DIR.parent))

    print(f"wrote {out} ({out.stat().st_size / 1_048_576:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
