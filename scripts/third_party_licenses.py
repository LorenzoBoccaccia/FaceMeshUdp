"""Write THIRD_PARTY_LICENSES.txt into a frozen release: the licence of every component it ships besides FaceMesh.

Runs in the environment the release was frozen from, so the packages and versions it lists are the
ones in the release.
"""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys
import zipfile
from pathlib import Path
from typing import Iterator, List, Set, Tuple

LICENSES_DIR = Path(__file__).resolve().parent.parent / "licenses"
STATIC_LICENSES = (
    ("MediaPipe Face Landmarker model bundle and the face landmarker graph port", "MediaPipe.txt"),
    ("Intel Integrated Performance Primitives, statically linked into OpenCV", "Intel-IPP.txt"),
)
SEPARATE_LICENSES = (
    ("cx_Freeze launcher in facemesh.exe", "frozen_application_license.txt"),
    ("Microsoft Visual C++ runtime DLLs", "share/licenses/vc_redist/LICENSE.txt"),
)
OWN_DISTRIBUTION = "facemesh"
LICENSE_WORDS = ("licen", "copying", "notice")
NOTICE_WORDS = ("licen", "copyright")
VENDORED_DIRS = {"extlibs", "vendor", "_vendor", "third_party"}
RULE = "=" * 78


def shipped_names(lib_dir: Path) -> Set[str]:
    names = {entry.name.split(".")[0] for entry in lib_dir.iterdir()}
    with zipfile.ZipFile(lib_dir / "library.zip") as library:
        names |= {Path(name).parts[0].split(".")[0] for name in library.namelist()}
    return names


def shipped_distributions(lib_dir: Path) -> List[metadata.Distribution]:
    owners = metadata.packages_distributions()
    names = {dist for name in shipped_names(lib_dir) for dist in owners.get(name, ())}
    return [metadata.distribution(name) for name in sorted(names - {OWN_DISTRIBUTION}, key=str.lower)]


def license_files(dist: metadata.Distribution) -> List[metadata.PackagePath]:
    return sorted(
        (
            path
            for path in dist.files or ()
            if path.parts[0].endswith(".dist-info")
            and any(word in path.name.lower() for word in LICENSE_WORDS)
        ),
        key=str,
    )


def vendored_notices(dist: metadata.Distribution) -> Iterator[Tuple[metadata.PackagePath, str]]:
    for path in sorted(dist.files or (), key=str):
        if path.suffix != ".py" or not VENDORED_DIRS & set(path.parts[:-1]):
            continue
        header = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not header and (line.startswith("#!") or not line.strip()):
                continue
            if not line.startswith("#"):
                break
            header.append(line[2:] if line.startswith("# ") else line[1:])
        text = "\n".join(header).strip()
        if any(word in text.lower() for word in NOTICE_WORDS):
            yield path, text


def section(title: str, body: str) -> str:
    return f"{RULE}\n{title}\n{RULE}\n\n{body.strip()}\n\n\n"


def distribution_title(dist: metadata.Distribution) -> str:
    return f"{dist.metadata['Name']} {dist.version}"


def distribution_section(dist: metadata.Distribution) -> str:
    expression = dist.metadata.get("License-Expression")
    parts = [f"License: {expression}" if expression else ""]
    for path in license_files(dist):
        parts.append(f"--- {path}\n\n{path.read_text(encoding='utf-8').strip()}")
    for path, notice in vendored_notices(dist):
        parts.append(f"--- notice of vendored {path}\n\n{notice}")
    return section(distribution_title(dist),"\n\n".join(part for part in parts if part))


def main() -> int:
    dist_dir = Path(sys.argv[1])
    distributions = shipped_distributions(dist_dir / "lib")
    python_title = f"Python {platform.python_version()} runtime, with the libraries its Windows build includes"
    titles = [python_title, *(distribution_title(dist) for dist in distributions)]
    titles += [title for title, _ in STATIC_LICENSES]

    index = [f"Third-party components of FaceMesh {metadata.version(OWN_DISTRIBUTION)} for Windows", ""]
    index += ["Licences in this file:", *(f"  - {title}" for title in titles), ""]
    index += ["Licences in their own file:", *(f"  - {title}: {path}" for title, path in SEPARATE_LICENSES)]
    sections = [
        "\n".join(index) + "\n\n\n",
        section(python_title, Path(sys.base_prefix, "LICENSE.txt").read_text(encoding="utf-8")),
        *(distribution_section(dist) for dist in distributions),
        *(section(title, (LICENSES_DIR / name).read_text(encoding="utf-8")) for title, name in STATIC_LICENSES),
    ]
    (dist_dir / "THIRD_PARTY_LICENSES.txt").write_text("".join(sections), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
