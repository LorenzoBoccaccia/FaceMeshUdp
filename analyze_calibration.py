#!/usr/bin/env python3
"""
Calibration analysis script.
Refits recorded calibration sessions and reports how well the gaze model explains each target.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))
from facemesh_app.calibration import (
    CALIBRATION_MODEL_VERSION,
    DEFAULT_VIEWING_DISTANCE_MM,
    POINT_NAMES,
    CalibrationPoint,
    CalibrationProfile,
    Screen,
)
from facemesh_app.facemesh_dao import MM_PER_CM


def session_profile(
    session: Dict[str, Any], viewing_distance_mm: Optional[float]
) -> Optional[CalibrationProfile]:
    """The calibration a session recorded, at its own viewing distance unless one is given."""
    screen = Screen.from_display(session.get("display") or {})
    points = tuple(CalibrationPoint.from_dict(p) for p in session.get("points") or [])
    if screen is None or {p.name for p in points} != set(POINT_NAMES):
        return None
    recorded = (session.get("model") or {}).get("viewingDistanceMm")
    distance = viewing_distance_mm or recorded or DEFAULT_VIEWING_DISTANCE_MM
    return CalibrationProfile(screen=screen, viewing_distance_mm=distance, points=points)


def capture_spread(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Per target, how much the raw eye reading and head pose moved while it was captured."""
    by_target: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        target = (sample.get("target") or {}).get("name")
        if sample.get("phase") == "capture" and sample.get("rawEye") and target:
            by_target.setdefault(target, []).append(sample)
    spread = {}
    for target, rows in by_target.items():
        if len(rows) < 2:
            continue
        spread[target] = {
            "frames": len(rows),
            "rawEyeYawSd": statistics.pstdev(r["rawEye"][0] for r in rows),
            "rawEyePitchSd": statistics.pstdev(r["rawEye"][1] for r in rows),
            "headYawSd": statistics.pstdev(r["headYaw"] for r in rows),
            "headPitchSd": statistics.pstdev(r["headPitch"] for r in rows),
        }
    return spread


def report(path: Path, viewing_distance_mm: Optional[float]) -> None:
    session = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n=== {path.name} (profile {session.get('profile')})")
    if session.get("modelVersion") != CALIBRATION_MODEL_VERSION:
        print(f"  recorded with model version {session.get('modelVersion')}; skipped")
        return
    profile = session_profile(session, viewing_distance_mm)
    if profile is None:
        print("  incomplete session or unknown display size; nothing to fit")
        return
    model = profile.fit()
    print(f"  {model.describe()}")
    spread = capture_spread(session.get("samples") or [])
    print("  target | gaze error px (x, y) | capture spread sd: raw eye yaw/pitch, head yaw/pitch")
    for point in profile.points:
        error = model.point_errors_px(point)
        stats = spread.get(point.name)
        error_text = "misses screen" if error is None else f"{error[0]:+7.1f}, {error[1]:+7.1f}"
        spread_text = (
            f"{stats['rawEyeYawSd']:.2f}/{stats['rawEyePitchSd']:.2f}, "
            f"{stats['headYawSd']:.2f}/{stats['headPitchSd']:.2f} ({stats['frames']} frames)"
            if stats
            else "--"
        )
        print(f"  {point.name:>6} | {error_text:>20} | {spread_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit and report recorded calibration sessions")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("calibration_data"),
        help="Directory holding calibration_session_*.json files",
    )
    parser.add_argument(
        "--viewing-distance",
        type=float,
        default=None,
        metavar="CM",
        help="Refit at this eye-to-screen distance instead of the recorded one",
    )
    args = parser.parse_args()
    paths = sorted(args.data_dir.glob("calibration_session_*.json"))
    if not paths:
        print(f"No calibration sessions in {args.data_dir}")
        sys.exit(1)
    distance = args.viewing_distance * MM_PER_CM if args.viewing_distance else None
    for path in paths:
        report(path, distance)


if __name__ == "__main__":
    main()
