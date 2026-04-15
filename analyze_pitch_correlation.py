#!/usr/bin/env python3
"""
Pitch Correlation Analysis Script
Analyzes captured pitch correlation data to validate eye pitch consistency across head positions.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent / "src"))
from facemesh_app.facemesh_dao import FaceMeshEvent


def load_pitch_correlation_data(
    data_dir: Path = Path("pitch_correlation"),
) -> Dict[str, Any]:
    """Load all pitch correlation data files."""
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} not found")
        sys.exit(1)

    combined_file = data_dir / "pitch_correlation_combined.json"
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
        if json_file.name == "pitch_correlation_combined.json":
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

    if left_pitch is None and right_pitch is None and combined_pitch is None:
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
    """Print a formatted table of pitch correlation data."""
    points = data.get("points", [])
    if not points:
        print("No data found")
        return

    print("\n" + "=" * 120)
    print("PITCH CORRELATION DATA ANALYSIS")
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
        f"{'Head Pitch (deg)':<16} {'Eye Pitch (deg)':<16} {'Left Eye (deg)':<14} {'Right Eye (deg)':<14}"
    )
    print("-" * 120)

    for point in points:
        name = point.get("name", "")
        head_pos = point.get("headPosition", "").upper()
        eye_pos = point.get("eyePosition", "").upper()
        result = point.get("rawResult", {})

        head_angles = extract_head_angles(result)
        head_pitch_str = f"{head_angles[1]:+7.2f}" if head_angles else "n/a"

        eye_gaze = extract_eye_gaze(result)
        if eye_gaze:
            eye_pitch_str = f"{eye_gaze['combined_pitch']:+8.2f}"
            left_pitch_str = (
                f"{eye_gaze['left_pitch']:+8.2f}"
                if eye_gaze["left_pitch"] is not None
                else "n/a"
            )
            right_pitch_str = (
                f"{eye_gaze['right_pitch']:+8.2f}"
                if eye_gaze["right_pitch"] is not None
                else "n/a"
            )
        else:
            eye_pitch_str = "n/a"
            left_pitch_str = "n/a"
            right_pitch_str = "n/a"

        print(
            f"{name:<28} {head_pos:<10} {eye_pos:<10} "
            f"{head_pitch_str:<16} {eye_pitch_str:<16} {left_pitch_str:<14} {right_pitch_str:<14}"
        )

    print("=" * 120)


def analyze_pitch_consistency(data: Dict[str, Any]) -> bool:
    """Analyze monotonicity and pitch response variation across head positions."""
    points = data.get("points", [])
    if len(points) < 9:
        print("\n[ERROR] Not enough data for pitch consistency analysis (need 9 points)")
        return False

    print("\n" + "=" * 120)
    print("PITCH CONSISTENCY ANALYSIS")
    print("=" * 120)

    head_groups: Dict[str, Dict[str, Dict[str, Any]]] = {
        "down": {},
        "center": {},
        "up": {},
    }
    for point in points:
        head_pos = point.get("headPosition")
        eye_pos = point.get("eyePosition")
        if head_pos in head_groups and eye_pos:
            head_groups[head_pos][eye_pos] = point

    all_tests_passed = True

    print("\n--- TEST 1: HEAD PITCH VALIDATION ---\n")
    head_pitch_values: Dict[str, float] = {}
    for head_pos in ["down", "center", "up"]:
        sample_point = None
        for eye_pos in ["down", "center", "up"]:
            if eye_pos in head_groups[head_pos]:
                sample_point = head_groups[head_pos][eye_pos]
                break

        if sample_point:
            head_angles = extract_head_angles(sample_point.get("rawResult", {}))
            if head_angles:
                _, pitch, _ = head_angles
                head_pitch_values[head_pos] = pitch
                print(f"Head {head_pos.upper():6s}: pitch = {pitch:+7.2f} deg")
            else:
                print(f"[FAIL] Head {head_pos.upper()}: could not extract pitch")
                all_tests_passed = False

    if len(head_pitch_values) == 3:
        down_pitch = head_pitch_values["down"]
        up_pitch = head_pitch_values["up"]
        center_pitch = head_pitch_values["center"]
        print(f"\nHead pitch range: {down_pitch:+.2f} deg (down) to {up_pitch:+.2f} deg (up)")
        if abs(down_pitch - center_pitch) < 5:
            print(
                f"[WARN] Head DOWN and CENTER positions too close (diff={abs(down_pitch - center_pitch):.2f} deg)"
            )
        if abs(center_pitch - up_pitch) < 5:
            print(
                f"[WARN] Head CENTER and UP positions too close (diff={abs(center_pitch - up_pitch):.2f} deg)"
            )

    print("\n--- TEST 2: EYE PITCH MONOTONICITY BY HEAD POSITION ---\n")
    pitch_deltas_by_head: Dict[str, Dict[str, float]] = {}

    for head_pos in ["down", "center", "up"]:
        print(f"\nHEAD POSITION: {head_pos.upper()}")
        print("-" * 80)

        head_pitch = head_pitch_values.get(head_pos)
        if head_pitch is not None:
            print(f"  Head pitch: {head_pitch:+7.2f} deg")
        print()

        eye_pitch_values: Dict[str, float] = {}
        eye_pitch_lr: Dict[str, Dict[str, Optional[float]]] = {}
        for eye_pos in ["down", "center", "up"]:
            if eye_pos not in head_groups[head_pos]:
                continue
            point = head_groups[head_pos][eye_pos]
            eye_gaze = extract_eye_gaze(point.get("rawResult", {}))
            if not eye_gaze or eye_gaze["combined_pitch"] is None:
                continue

            combined = float(eye_gaze["combined_pitch"])
            eye_pitch_values[eye_pos] = combined

            left_pitch = eye_gaze["left_pitch"]
            right_pitch = eye_gaze["right_pitch"]
            eye_pitch_lr[eye_pos] = {
                "left": float(left_pitch) if left_pitch is not None else None,
                "right": float(right_pitch) if right_pitch is not None else None,
            }
            left_str = f"{left_pitch:+7.2f}" if left_pitch is not None else "   n/a "
            right_str = f"{right_pitch:+7.2f}" if right_pitch is not None else "   n/a "
            print(
                f"  Eye {eye_pos.upper():6s}: pitch = {combined:+8.2f} deg "
                f"(L:{left_str} R:{right_str})"
            )

        if len(eye_pitch_values) == 3:
            down_val = eye_pitch_values["down"]
            center_val = eye_pitch_values["center"]
            up_val = eye_pitch_values["up"]

            print("\n  Checking eye pitch ordering (expected: down < center < up):")
            if down_val < center_val < up_val:
                print("  [OK] Eye pitch correctly ordered: down < center < up")
            else:
                print("  [FAIL] Eye pitch not correctly ordered")
                print(
                    f"       down={down_val:+.2f} deg, center={center_val:+.2f} deg, up={up_val:+.2f} deg"
                )
                up_center_margin = up_val - center_val
                center_down_margin = center_val - down_val
                span = up_val - down_val
                print("  Explainability:")
                print(f"       center->up margin   = {up_center_margin:+.2f} deg (expected > 0)")
                print(f"       down->center margin = {center_down_margin:+.2f} deg (expected > 0)")
                print(f"       total span (up-down)= {span:+.2f} deg")

                left_down = eye_pitch_lr.get("down", {}).get("left")
                left_center = eye_pitch_lr.get("center", {}).get("left")
                left_up = eye_pitch_lr.get("up", {}).get("left")
                right_down = eye_pitch_lr.get("down", {}).get("right")
                right_center = eye_pitch_lr.get("center", {}).get("right")
                right_up = eye_pitch_lr.get("up", {}).get("right")

                if None not in (left_down, left_center, left_up):
                    left_up_center = left_up - left_center
                    print(
                        f"       left-eye up-center  = {left_up_center:+.2f} deg "
                        f"({'OK' if left_down < left_center < left_up else 'FAIL'})"
                    )
                if None not in (right_down, right_center, right_up):
                    right_up_center = right_up - right_center
                    print(
                        f"       right-eye up-center = {right_up_center:+.2f} deg "
                        f"({'OK' if right_down < right_center < right_up else 'FAIL'})"
                    )

                if (
                    None not in (left_down, right_down)
                    and None not in (left_center, right_center)
                    and None not in (left_up, right_up)
                ):
                    d_asym = abs(left_down - right_down)
                    c_asym = abs(left_center - right_center)
                    u_asym = abs(left_up - right_up)
                    print(
                        f"       inter-eye |L-R|     = down:{d_asym:.2f} center:{c_asym:.2f} up:{u_asym:.2f} deg"
                    )
                all_tests_passed = False

            up_minus_center = up_val - center_val
            down_minus_center = down_val - center_val
            center_minus_down = center_val - down_val
            up_minus_down = up_val - down_val

            pitch_deltas_by_head[head_pos] = {
                "up_minus_center": up_minus_center,
                "down_minus_center": down_minus_center,
                "center_minus_down": center_minus_down,
                "up_minus_down": up_minus_down,
            }

            print("\n  Delta from CENTER:")
            print(f"  UP - CENTER   = {up_minus_center:+8.2f} deg")
            print(f"  DOWN - CENTER = {down_minus_center:+8.2f} deg")
            print(f"  CENTER - DOWN = {center_minus_down:+8.2f} deg")

    print("\n--- TEST 3: PITCH DELTA VARIATION ACROSS HEAD POSITIONS ---\n")
    if len(pitch_deltas_by_head) == 3:
        up_deltas = [pitch_deltas_by_head[h]["up_minus_center"] for h in ["down", "center", "up"]]
        down_deltas = [
            pitch_deltas_by_head[h]["center_minus_down"] for h in ["down", "center", "up"]
        ]
        spans = [pitch_deltas_by_head[h]["up_minus_down"] for h in ["down", "center", "up"]]

        for head_pos in ["down", "center", "up"]:
            d = pitch_deltas_by_head[head_pos]
            print(
                f"Head {head_pos.upper():6s}: "
                f"UP-CENTER={d['up_minus_center']:+7.2f} deg  "
                f"CENTER-DOWN={d['center_minus_down']:+7.2f} deg  "
                f"UP-DOWN={d['up_minus_down']:+7.2f} deg"
            )

        print()
        print(
            f"UP-CENTER variation   : min={min(up_deltas):+.2f} deg max={max(up_deltas):+.2f} deg "
            f"range={(max(up_deltas) - min(up_deltas)):.2f} deg"
        )
        print(
            f"CENTER-DOWN variation : min={min(down_deltas):+.2f} deg max={max(down_deltas):+.2f} deg "
            f"range={(max(down_deltas) - min(down_deltas)):.2f} deg"
        )
        print(
            f"UP-DOWN variation     : min={min(spans):+.2f} deg max={max(spans):+.2f} deg "
            f"range={(max(spans) - min(spans)):.2f} deg"
        )
    else:
        print("[WARN] Could not compute deltas for all head positions")

    print("\n--- TEST 4: FACEMESHDAO ANGLE VALIDATION ---\n")
    all_dao_ok = True
    for head_pos in ["down", "center", "up"]:
        print(f"\nHead position: {head_pos.upper()}")
        print("-" * 80)
        for eye_pos in ["down", "center", "up"]:
            if eye_pos not in head_groups[head_pos]:
                continue
            point = head_groups[head_pos][eye_pos]
            event = FaceMeshEvent.from_landmarker_result(
                _create_mock_result(point.get("rawResult", {}))
            )
            left_pitch = event.left_eye_gaze_pitch
            right_pitch = event.right_eye_gaze_pitch
            combined_pitch = event.combined_eye_gaze_pitch
            if combined_pitch is not None:
                left_str = f"{left_pitch:+7.2f}" if left_pitch is not None else "   n/a "
                right_str = f"{right_pitch:+7.2f}" if right_pitch is not None else "   n/a "
                print(
                    f"  Eye {eye_pos.upper():6s}: FaceMeshDao pitch = {combined_pitch:+8.2f} deg "
                    f"(L:{left_str} R:{right_str})"
                )
            else:
                print(f"  Eye {eye_pos.upper():6s}: Could not extract pitch from FaceMeshDao")
                all_dao_ok = False

    if not all_dao_ok:
        print("\n[WARN] FaceMeshDao extraction failed for some points")
        all_tests_passed = False

    print("\n" + "=" * 120)
    if all_tests_passed:
        print("[OK] ALL TESTS PASSED - Eye pitch is consistent across head positions")
    else:
        print("[FAIL] SOME TESTS FAILED - Eye pitch consistency issues detected")
    print("=" * 120 + "\n")

    return all_tests_passed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze pitch correlation capture data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="pitch_correlation",
        help="Directory containing pitch correlation data (default: pitch_correlation)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data = load_pitch_correlation_data(data_dir)

    print_table(data)
    success = analyze_pitch_consistency(data)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
