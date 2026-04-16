#!/usr/bin/env python3
"""
Analyze harmonization capture data against one explicit test case contract.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent / "src"))
from facemesh_app.facemesh_dao import FaceMeshEvent
from facemesh_app.harmonization_contract import (
    HARMONIZATION_PROMPTS,
    HARMONIZATION_SCHEMA_VERSION,
    HARMONIZATION_TEST_CASE,
)


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isfinite(out):
        return out
    return fallback


def fmt(value: Optional[float], width: int = 9, precision: int = 2) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    if not math.isfinite(value):
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{precision}f}"


def load_harmonization_data(data_dir: Path) -> Dict[str, Any]:
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    combined_file = data_dir / "harmonization_combined.json"
    if combined_file.exists():
        with combined_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    points: List[Dict[str, Any]] = []
    for json_file in sorted(data_dir.glob("*.json")):
        if json_file.name == "harmonization_combined.json":
            continue
        with json_file.open("r", encoding="utf-8") as handle:
            points.append(json.load(handle))
    return {
        "schemaVersion": HARMONIZATION_SCHEMA_VERSION,
        "suite": "harmonization",
        "timestamp": None,
        "captureCount": len(points),
        "cameraInfo": {},
        "prompts": HARMONIZATION_PROMPTS,
        "points": points,
        "testCase": HARMONIZATION_TEST_CASE,
    }


def _create_mock_result(raw_result: Dict[str, Any]) -> Any:
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


def extract_measurements(raw_result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    event = FaceMeshEvent.from_landmarker_result(_create_mock_result(raw_result))
    return {
        "head_yaw": event.head_yaw,
        "head_pitch": event.head_pitch,
        "head_roll": event.roll,
        "translation_x": event.x,
        "translation_y": event.y,
        "translation_z": event.raw_transform_z,
        "eye_yaw": event.combined_eye_gaze_yaw,
        "eye_pitch": event.combined_eye_gaze_pitch,
        "left_eye_yaw": event.left_eye_gaze_yaw,
        "left_eye_pitch": event.left_eye_gaze_pitch,
        "right_eye_yaw": event.right_eye_gaze_yaw,
        "right_eye_pitch": event.right_eye_gaze_pitch,
    }


def build_measurement_map(points: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    measurement_map: Dict[str, Dict[str, Optional[float]]] = {}
    for point in points:
        name = str(point.get("name") or "")
        if not name:
            continue
        raw_result = point.get("rawResult") or {}
        measurement_map[name] = extract_measurements(raw_result)
    return measurement_map


def hydrate_points_from_prompts(
    points: List[Dict[str, Any]],
    prompts: List[Dict[str, Any]],
    force_prompt_fields: bool = False,
) -> List[Dict[str, Any]]:
    prompt_by_name: Dict[str, Dict[str, Any]] = {}
    for prompt in prompts:
        name = str(prompt.get("name") or "")
        if name:
            prompt_by_name[name] = prompt

    hydrated_points: List[Dict[str, Any]] = []
    for point in points:
        point_name = str(point.get("name") or "")
        prompt = prompt_by_name.get(point_name) or {}
        hydrated: Dict[str, Any] = dict(point)
        if force_prompt_fields or not hydrated.get("movementType"):
            hydrated["movementType"] = prompt.get("type", "")
        if force_prompt_fields or not hydrated.get("movementAxis"):
            hydrated["movementAxis"] = prompt.get("axis", "")
        if force_prompt_fields or not hydrated.get("movementDirection"):
            hydrated["movementDirection"] = prompt.get("direction", "")
        hydrated_points.append(hydrated)
    return hydrated_points


def print_table(points: List[Dict[str, Any]], measurement_map: Dict[str, Dict[str, Optional[float]]]) -> None:
    print("\n" + "=" * 162)
    print("HARMONIZATION CAPTURE TABLE")
    print("=" * 162)
    print(
        f"{'Point':<14} {'Type':<8} {'Axis':<7} {'Dir':<9} "
        f"{'HeadYaw':>9} {'HeadPitch':>10} {'HeadRoll':>9} {'TransX':>9} {'TransY':>9} {'TransZ':>9} "
        f"{'EyeYaw':>9} {'EyePitch':>10} {'LeftEyeYaw':>11} {'RightEyeYaw':>12}"
    )
    print("-" * 162)
    for point in points:
        name = str(point.get("name") or "")
        m = measurement_map.get(name) or {}
        print(
            f"{name:<14} "
            f"{str(point.get('movementType') or ''):<8} "
            f"{str(point.get('movementAxis') or ''):<7} "
            f"{str(point.get('movementDirection') or ''):<9} "
            f"{fmt(m.get('head_yaw'))} "
            f"{fmt(m.get('head_pitch'), width=10)} "
            f"{fmt(m.get('head_roll'))} "
            f"{fmt(m.get('translation_x'))} "
            f"{fmt(m.get('translation_y'))} "
            f"{fmt(m.get('translation_z'))} "
            f"{fmt(m.get('eye_yaw'))} "
            f"{fmt(m.get('eye_pitch'), width=10)} "
            f"{fmt(m.get('left_eye_yaw'), width=11)} "
            f"{fmt(m.get('right_eye_yaw'), width=12)}"
        )


def resolve_operand(
    operand: Dict[str, Any],
    measurement_map: Dict[str, Dict[str, Optional[float]]],
) -> Tuple[Optional[float], str]:
    if "value" in operand:
        value = safe_float(operand.get("value"), float("nan"))
        if math.isfinite(value):
            return float(value), str(value)
        return None, "invalid value operand"
    point_name = str(operand.get("point") or "")
    metric = str(operand.get("metric") or "")
    if not point_name or not metric:
        return None, "invalid operand"
    if point_name not in measurement_map:
        return None, f"missing point '{point_name}'"
    value = measurement_map[point_name].get(metric)
    if value is None or not math.isfinite(value):
        return None, f"missing metric '{metric}' for point '{point_name}'"
    return float(value), f"{point_name}.{metric}"


def evaluate_test_case(
    test_case: Dict[str, Any],
    measurement_map: Dict[str, Dict[str, Optional[float]]],
) -> Tuple[List[Dict[str, Any]], bool]:
    rows: List[Dict[str, Any]] = []
    all_passed = True
    for comparison in test_case.get("comparisons") or []:
        name = str(comparison.get("name") or "")
        op = str(comparison.get("op") or "")
        min_delta = safe_float(comparison.get("minDelta"), 0.0)
        lhs_val, lhs_label = resolve_operand(comparison.get("lhs") or {}, measurement_map)
        rhs_val, rhs_label = resolve_operand(comparison.get("rhs") or {}, measurement_map)

        if lhs_val is None:
            rows.append(
                {
                    "name": name,
                    "passed": False,
                    "lhs": lhs_label,
                    "rhs": rhs_label,
                    "delta": None,
                    "op": op,
                    "minDelta": min_delta,
                    "reason": lhs_label,
                }
            )
            all_passed = False
            continue
        if rhs_val is None:
            rows.append(
                {
                    "name": name,
                    "passed": False,
                    "lhs": lhs_label,
                    "rhs": rhs_label,
                    "delta": None,
                    "op": op,
                    "minDelta": min_delta,
                    "reason": rhs_label,
                }
            )
            all_passed = False
            continue

        delta = lhs_val - rhs_val
        passed = False
        if op == "gt":
            passed = delta > min_delta
        elif op == "lt":
            passed = delta < -min_delta
        else:
            passed = False
        if not passed:
            all_passed = False
        rows.append(
            {
                "name": name,
                "passed": passed,
                "lhs": f"{lhs_label}={lhs_val:+.3f}",
                "rhs": f"{rhs_label}={rhs_val:+.3f}",
                "delta": delta,
                "op": op,
                "minDelta": min_delta,
                "reason": "",
            }
        )

    return rows, all_passed


def print_test_case_result(test_case: Dict[str, Any], rows: List[Dict[str, Any]], passed: bool) -> None:
    print("\n" + "=" * 120)
    print(f"TEST CASE: {test_case.get('id', 'harmonization')}")
    description = str(test_case.get("description") or "")
    if description:
        print(description)
    print("-" * 120)
    print(
        f"{'Check':<34} {'Status':<8} {'LHS':<34} {'RHS':<34} {'Delta':>8} {'Rule':>10}"
    )
    print("-" * 120)
    for row in rows:
        status = "PASS" if bool(row.get("passed")) else "FAIL"
        delta = row.get("delta")
        delta_text = "n/a" if delta is None else f"{delta:+.3f}"
        rule = f"{row.get('op')} {safe_float(row.get('minDelta'), 0.0):.1f}"
        lhs = str(row.get("lhs") or "")
        rhs = str(row.get("rhs") or "")
        reason = str(row.get("reason") or "")
        if reason:
            rhs = reason
        print(
            f"{str(row.get('name') or ''):<34} "
            f"{status:<8} "
            f"{lhs:<34} "
            f"{rhs:<34} "
            f"{delta_text:>8} "
            f"{rule:>10}"
        )
    print("-" * 120)
    if passed:
        print("[OK] Harmonization test case passed")
    else:
        print("[FAIL] Harmonization test case failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze harmonization capture against one test-case contract"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="harmonization_data",
        help="Directory containing harmonization capture output",
    )
    args = parser.parse_args()

    payload = load_harmonization_data(Path(args.data_dir))
    payload_schema_version = int(safe_float(payload.get("schemaVersion"), 0.0))
    force_prompt_fields = False
    if payload_schema_version < HARMONIZATION_SCHEMA_VERSION:
        prompts = HARMONIZATION_PROMPTS
        test_case = HARMONIZATION_TEST_CASE
        force_prompt_fields = True
    else:
        prompts = payload.get("prompts") or HARMONIZATION_PROMPTS
        test_case = payload.get("testCase") or HARMONIZATION_TEST_CASE

    points = hydrate_points_from_prompts(
        payload.get("points") or [],
        prompts,
        force_prompt_fields=force_prompt_fields,
    )
    if not points:
        print("Error: no harmonization points found")
        sys.exit(1)

    measurement_map = build_measurement_map(points)
    print_table(points, measurement_map)

    rows, passed = evaluate_test_case(test_case, measurement_map)
    print_test_case_result(test_case, rows, passed)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
