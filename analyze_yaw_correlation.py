#!/usr/bin/env python3
"""
Yaw Correlation Analysis Script
Analyzes captured yaw correlation data to validate eye yaw consistency across head positions.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent / "src"))
from facemesh_app.facemesh_dao import FaceMeshEvent


def load_yaw_correlation_data(
    data_dir: Path = Path("yaw_correlation"),
) -> Dict[str, Any]:
    """Load all yaw correlation data files."""
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} not found")
        sys.exit(1)

    combined_file = data_dir / "yaw_correlation_combined.json"
    if combined_file.exists():
        with combined_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    combined = {
        "timestamp": None,
        "captureCount": 0,
        "cameraInfo": {},
        "points": [],
    }

    for json_file in sorted(data_dir.glob("*.json")):
        if json_file.name == "yaw_correlation_combined.json":
            continue
        with json_file.open("r", encoding="utf-8") as f:
            combined["points"].append(json.load(f))

    combined["captureCount"] = len(combined["points"])
    return combined


def extract_head_angles(result: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """Extract yaw, pitch, roll from facial transformation matrix."""
    matrix = result.get("facial_transformation_matrix")
    if matrix is None or len(matrix) < 16:
        return None

    m44 = [
        [matrix[0], matrix[1], matrix[2], matrix[3]],
        [matrix[4], matrix[5], matrix[6], matrix[7]],
        [matrix[8], matrix[9], matrix[10], matrix[11]],
        [matrix[12], matrix[13], matrix[14], matrix[15]],
    ]

    face_forward = [-m44[0][2], -m44[1][2], -m44[2][2]]
    yaw = math.degrees(math.atan2(face_forward[0], -face_forward[2]))
    pitch = math.degrees(math.atan2(-face_forward[1], -face_forward[2]))
    roll = math.degrees(math.atan2(m44[0][1], m44[0][0]))
    return yaw, pitch, roll


def _create_mock_result(raw_result: Dict[str, Any]) -> Any:
    """Create a mock MediaPipe result object from raw JSON data."""
    from types import SimpleNamespace

    facial_transformation_matrix = raw_result.get("facial_transformation_matrix")
    if facial_transformation_matrix is None:
        return SimpleNamespace(
            facial_transformation_matrixes=None,
            face_landmarks=[],
            face_blendshapes=[],
        )

    face_landmarks_data = raw_result.get("face_landmarks", [])
    face_landmarks = []
    for lm_data in face_landmarks_data:
        if lm_data is None:
            face_landmarks.append(None)
        else:
            face_landmarks.append(
                SimpleNamespace(
                    x=lm_data.get("x"),
                    y=lm_data.get("y"),
                    z=lm_data.get("z"),
                )
            )

    class MockResult:
        def __init__(self):
            self.facial_transformation_matrixes = [facial_transformation_matrix]
            self.face_landmarks = [face_landmarks]
            self.face_blendshapes = []

    return MockResult()


def extract_eye_gaze(result: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Extract eye gaze angles through FaceMeshEvent to match runtime math."""
    event = FaceMeshEvent.from_landmarker_result(_create_mock_result(result))

    left_yaw = event.left_eye_gaze_yaw
    left_pitch = event.left_eye_gaze_pitch
    right_yaw = event.right_eye_gaze_yaw
    right_pitch = event.right_eye_gaze_pitch
    combined_yaw = event.combined_eye_gaze_yaw
    combined_pitch = event.combined_eye_gaze_pitch

    if left_yaw is None and right_yaw is None and combined_yaw is None:
        return None

    return {
        "left_yaw": left_yaw,
        "left_pitch": left_pitch,
        "right_yaw": right_yaw,
        "right_pitch": right_pitch,
        "combined_yaw": combined_yaw,
        "combined_pitch": combined_pitch,
    }


def print_table(data: Dict[str, Any]) -> None:
    """Print a formatted table of yaw correlation data."""
    points = data.get("points", [])
    if not points:
        print("No data found")
        return

    print("\n" + "=" * 120)
    print("YAW CORRELATION DATA ANALYSIS")
    print("=" * 120)

    camera_info = data.get("cameraInfo", {})
    if camera_info:
        print(
            f"\nCamera: backend={camera_info.get('backend')} index={camera_info.get('index')} "
            f"{camera_info.get('width')}x{camera_info.get('height')} @ {camera_info.get('fps'):.1f}fps"
        )

    print(f"\nCaptured {len(points)} points")
    print(
        f"\n{'Name':<28} {'Head Pos':<10} {'Eye Pos':<10} "
        f"{'Head Yaw (deg)':<16} {'Eye Yaw (deg)':<16} {'Left Eye (deg)':<14} {'Right Eye (deg)':<14}"
    )
    print("-" * 120)

    for point in points:
        name = point.get("name", "")
        head_pos = point.get("headPosition", "").upper()
        eye_pos = point.get("eyePosition", "").upper()
        result = point.get("rawResult", {})

        head_angles = extract_head_angles(result)
        head_yaw_str = f"{head_angles[0]:+7.2f}" if head_angles else "n/a"

        eye_gaze = extract_eye_gaze(result)
        if eye_gaze:
            eye_yaw_str = f"{eye_gaze['combined_yaw']:+8.2f}"
            left_yaw_str = (
                f"{eye_gaze['left_yaw']:+8.2f}"
                if eye_gaze["left_yaw"] is not None
                else "n/a"
            )
            right_yaw_str = (
                f"{eye_gaze['right_yaw']:+8.2f}"
                if eye_gaze["right_yaw"] is not None
                else "n/a"
            )
        else:
            eye_yaw_str = "n/a"
            left_yaw_str = "n/a"
            right_yaw_str = "n/a"

        print(
            f"{name:<28} {head_pos:<10} {eye_pos:<10} "
            f"{head_yaw_str:<16} {eye_yaw_str:<16} {left_yaw_str:<14} {right_yaw_str:<14}"
        )

    print("=" * 120)


def analyze_yaw_consistency(data: Dict[str, Any]) -> bool:
    """Analyze monotonicity and yaw response variation across head positions."""
    points = data.get("points", [])
    if len(points) < 9:
        print("\n[ERROR] Not enough data for yaw consistency analysis (need 9 points)")
        return False

    print("\n" + "=" * 120)
    print("YAW CONSISTENCY ANALYSIS")
    print("=" * 120)

    head_groups: Dict[str, Dict[str, Dict[str, Any]]] = {
        "left": {},
        "center": {},
        "right": {},
    }
    for point in points:
        head_pos = point.get("headPosition")
        eye_pos = point.get("eyePosition")
        if head_pos in head_groups and eye_pos:
            head_groups[head_pos][eye_pos] = point

    all_tests_passed = True

    print("\n--- TEST 1: HEAD YAW VALIDATION ---\n")
    head_yaw_values: Dict[str, float] = {}
    for head_pos in ["left", "center", "right"]:
        sample_point = None
        for eye_pos in ["left", "center", "right"]:
            if eye_pos in head_groups[head_pos]:
                sample_point = head_groups[head_pos][eye_pos]
                break

        if sample_point:
            head_angles = extract_head_angles(sample_point.get("rawResult", {}))
            if head_angles:
                yaw, _, _ = head_angles
                head_yaw_values[head_pos] = yaw
                print(f"Head {head_pos.upper():6s}: yaw = {yaw:+7.2f} deg")
            else:
                print(f"[FAIL] Head {head_pos.upper()}: could not extract yaw")
                all_tests_passed = False

    if len(head_yaw_values) == 3:
        left_yaw = head_yaw_values["left"]
        right_yaw = head_yaw_values["right"]
        center_yaw = head_yaw_values["center"]
        print(
            f"\nHead yaw range: {left_yaw:+.2f} deg (left) to {right_yaw:+.2f} deg (right)"
        )
        if abs(left_yaw - center_yaw) < 5:
            print(
                f"[WARN] Head LEFT and CENTER positions too close (diff={abs(left_yaw - center_yaw):.2f} deg)"
            )
        if abs(center_yaw - right_yaw) < 5:
            print(
                f"[WARN] Head CENTER and RIGHT positions too close (diff={abs(center_yaw - right_yaw):.2f} deg)"
            )

    print("\n--- TEST 2: EYE YAW MONOTONICITY BY HEAD POSITION ---\n")
    yaw_deltas_by_head: Dict[str, Dict[str, float]] = {}

    for head_pos in ["left", "center", "right"]:
        print(f"\nHEAD POSITION: {head_pos.upper()}")
        print("-" * 80)

        head_yaw = head_yaw_values.get(head_pos)
        if head_yaw is not None:
            print(f"  Head yaw: {head_yaw:+7.2f} deg")
        print()

        eye_yaw_values: Dict[str, float] = {}
        eye_yaw_lr: Dict[str, Dict[str, Optional[float]]] = {}
        for eye_pos in ["left", "center", "right"]:
            if eye_pos not in head_groups[head_pos]:
                continue
            point = head_groups[head_pos][eye_pos]
            eye_gaze = extract_eye_gaze(point.get("rawResult", {}))
            if not eye_gaze or eye_gaze["combined_yaw"] is None:
                continue

            combined = float(eye_gaze["combined_yaw"])
            eye_yaw_values[eye_pos] = combined

            left_yaw = eye_gaze["left_yaw"]
            right_yaw = eye_gaze["right_yaw"]
            eye_yaw_lr[eye_pos] = {
                "left": float(left_yaw) if left_yaw is not None else None,
                "right": float(right_yaw) if right_yaw is not None else None,
            }
            left_str = f"{left_yaw:+7.2f}" if left_yaw is not None else "   n/a "
            right_str = f"{right_yaw:+7.2f}" if right_yaw is not None else "   n/a "
            print(
                f"  Eye {eye_pos.upper():6s}: yaw = {combined:+8.2f} deg "
                f"(L:{left_str} R:{right_str})"
            )

        if len(eye_yaw_values) == 3:
            left_val = eye_yaw_values["left"]
            center_val = eye_yaw_values["center"]
            right_val = eye_yaw_values["right"]

            print("\n  Checking eye yaw ordering (expected: left < center < right):")
            if left_val < center_val < right_val:
                print("  [OK] Eye yaw correctly ordered: left < center < right")
            else:
                print("  [FAIL] Eye yaw not correctly ordered")
                print(
                    f"       left={left_val:+.2f} deg, center={center_val:+.2f} deg, right={right_val:+.2f} deg"
                )
                right_center_margin = right_val - center_val
                center_left_margin = center_val - left_val
                span = right_val - left_val
                print("  Explainability:")
                print(
                    f"       center->right margin = {right_center_margin:+.2f} deg (expected > 0)"
                )
                print(
                    f"       left->center margin  = {center_left_margin:+.2f} deg (expected > 0)"
                )
                print(f"       total span (right-left) = {span:+.2f} deg")

                left_left = eye_yaw_lr.get("left", {}).get("left")
                left_center = eye_yaw_lr.get("center", {}).get("left")
                left_right = eye_yaw_lr.get("right", {}).get("left")
                right_left = eye_yaw_lr.get("left", {}).get("right")
                right_center = eye_yaw_lr.get("center", {}).get("right")
                right_right = eye_yaw_lr.get("right", {}).get("right")

                if None not in (left_left, left_center, left_right):
                    left_right_center = left_right - left_center
                    print(
                        f"       left-eye right-center = {left_right_center:+.2f} deg "
                        f"({'OK' if left_left < left_center < left_right else 'FAIL'})"
                    )
                if None not in (right_left, right_center, right_right):
                    right_right_center = right_right - right_center
                    print(
                        f"       right-eye right-center = {right_right_center:+.2f} deg "
                        f"({'OK' if right_left < right_center < right_right else 'FAIL'})"
                    )

                if (
                    None not in (left_left, right_left)
                    and None not in (left_center, right_center)
                    and None not in (left_right, right_right)
                ):
                    l_asym = abs(left_left - right_left)
                    c_asym = abs(left_center - right_center)
                    r_asym = abs(left_right - right_right)
                    print(
                        f"       inter-eye |L-R|     = left:{l_asym:.2f} center:{c_asym:.2f} right:{r_asym:.2f} deg"
                    )
                all_tests_passed = False

            right_minus_center = right_val - center_val
            left_minus_center = left_val - center_val
            center_minus_left = center_val - left_val
            right_minus_left = right_val - left_val

            yaw_deltas_by_head[head_pos] = {
                "right_minus_center": right_minus_center,
                "left_minus_center": left_minus_center,
                "center_minus_left": center_minus_left,
                "right_minus_left": right_minus_left,
            }

            print("\n  Delta from CENTER:")
            print(f"  RIGHT - CENTER = {right_minus_center:+8.2f} deg")
            print(f"  LEFT - CENTER  = {left_minus_center:+8.2f} deg")
            print(f"  CENTER - LEFT  = {center_minus_left:+8.2f} deg")

    print("\n--- TEST 3: YAW DELTA VARIATION ACROSS HEAD POSITIONS ---\n")
    if len(yaw_deltas_by_head) == 3:
        right_deltas = [
            yaw_deltas_by_head[h]["right_minus_center"]
            for h in ["left", "center", "right"]
        ]
        left_deltas = [
            yaw_deltas_by_head[h]["center_minus_left"]
            for h in ["left", "center", "right"]
        ]
        spans = [
            yaw_deltas_by_head[h]["right_minus_left"]
            for h in ["left", "center", "right"]
        ]

        for head_pos in ["left", "center", "right"]:
            d = yaw_deltas_by_head[head_pos]
            print(
                f"Head {head_pos.upper():6s}: "
                f"RIGHT-CENTER={d['right_minus_center']:+7.2f} deg  "
                f"CENTER-LEFT={d['center_minus_left']:+7.2f} deg  "
                f"RIGHT-LEFT={d['right_minus_left']:+7.2f} deg"
            )

        print()
        print(
            f"RIGHT-CENTER variation : min={min(right_deltas):+.2f} deg max={max(right_deltas):+.2f} deg "
            f"range={(max(right_deltas) - min(right_deltas)):.2f} deg"
        )
        print(
            f"CENTER-LEFT variation  : min={min(left_deltas):+.2f} deg max={max(left_deltas):+.2f} deg "
            f"range={(max(left_deltas) - min(left_deltas)):.2f} deg"
        )
        print(
            f"RIGHT-LEFT variation   : min={min(spans):+.2f} deg max={max(spans):+.2f} deg "
            f"range={(max(spans) - min(spans)):.2f} deg"
        )
    else:
        print("[WARN] Could not compute deltas for all head positions")

    print("\n--- TEST 4: FACEMESHDAO ANGLE VALIDATION ---\n")
    all_dao_ok = True
    for head_pos in ["left", "center", "right"]:
        print(f"\nHead position: {head_pos.upper()}")
        print("-" * 80)
        for eye_pos in ["left", "center", "right"]:
            if eye_pos not in head_groups[head_pos]:
                continue
            point = head_groups[head_pos][eye_pos]
            event = FaceMeshEvent.from_landmarker_result(
                _create_mock_result(point.get("rawResult", {}))
            )
            left_yaw = event.left_eye_gaze_yaw
            right_yaw = event.right_eye_gaze_yaw
            combined_yaw = event.combined_eye_gaze_yaw
            if combined_yaw is not None:
                left_str = f"{left_yaw:+7.2f}" if left_yaw is not None else "   n/a "
                right_str = f"{right_yaw:+7.2f}" if right_yaw is not None else "   n/a "
                print(
                    f"  Eye {eye_pos.upper():6s}: FaceMeshDao yaw = {combined_yaw:+8.2f} deg "
                    f"(L:{left_str} R:{right_str})"
                )
            else:
                print(
                    f"  Eye {eye_pos.upper():6s}: Could not extract yaw from FaceMeshDao"
                )
                all_dao_ok = False

    if not all_dao_ok:
        print("\n[WARN] FaceMeshDao extraction failed for some points")
        all_tests_passed = False

    print("\n" + "=" * 120)
    if all_tests_passed:
        print("[OK] ALL TESTS PASSED - Eye yaw is consistent across head positions")
    else:
        print("[FAIL] SOME TESTS FAILED - Eye yaw consistency issues detected")
    print("=" * 120 + "\n")

    return all_tests_passed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze yaw correlation capture data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="yaw_correlation",
        help="Directory containing yaw correlation data (default: yaw_correlation)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data = load_yaw_correlation_data(data_dir)

    print_table(data)
    success = analyze_yaw_consistency(data)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
