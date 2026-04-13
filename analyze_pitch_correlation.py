#!/usr/bin/env python3
"""
Pitch Correlation Analysis Script
Analyzes captured pitch correlation data to validate eye pitch consistency across head positions.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import math

# Import FaceMeshDao to test its angle extraction
sys.path.insert(0, str(Path(__file__).parent / "src"))
from facemesh_app.facemesh_dao import FaceMeshEvent


def load_pitch_correlation_data(data_dir: Path = Path("pitch_correlation")) -> Dict[str, Any]:
    """Load all pitch correlation data files."""
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} not found")
        sys.exit(1)
    
    # Try to load combined file first
    combined_file = data_dir / "pitch_correlation_combined.json"
    if combined_file.exists():
        with combined_file.open('r') as f:
            return json.load(f)
    
    # Otherwise load individual files
    combined = {
        "timestamp": None,
        "captureCount": 0,
        "cameraInfo": {},
        "points": [],
    }
    
    for json_file in sorted(data_dir.glob("*.json")):
        if json_file.name == "pitch_correlation_combined.json":
            continue
        
        with json_file.open('r') as f:
            point_data = json.load(f)
            combined["points"].append(point_data)
    
    combined["captureCount"] = len(combined["points"])
    return combined


def extract_head_angles(result: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """Extract yaw, pitch, roll from facial transformation matrix."""
    matrix = result.get("facial_transformation_matrix")
    if matrix is None or len(matrix) < 16:
        return None
    
    # Convert flat array to 4x4 matrix
    m44 = [
        [matrix[0], matrix[1], matrix[2], matrix[3]],
        [matrix[4], matrix[5], matrix[6], matrix[7]],
        [matrix[8], matrix[9], matrix[10], matrix[11]],
        [matrix[12], matrix[13], matrix[14], matrix[15]],
    ]
    
    # Extract face forward vector (third column, inverted to face camera)
    face_forward = [
        -m44[0][2],
        -m44[1][2],
        -m44[2][2],
    ]
    
    # Calculate yaw (horizontal rotation)
    yaw = math.degrees(math.atan2(face_forward[0], -face_forward[2]))
    
    # Calculate pitch (vertical rotation)
    pitch = math.degrees(math.atan2(-face_forward[1], -face_forward[2]))
    
    # Calculate roll (rotation around forward axis)
    roll = math.degrees(math.atan2(m44[0][1], m44[0][0]))
    
    return yaw, pitch, roll


def extract_eye_gaze(result: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Extract eye gaze angles from landmarks."""
    landmarks = result.get("face_landmarks")
    if landmarks is None or len(landmarks) < 478:
        return None
    
    def get_lm(idx: int) -> Optional[Tuple[float, float]]:
        """Get landmark coordinates at given index."""
        if idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        if lm is None:
            return None
        x = lm.get("x")
        y = lm.get("y")
        if x is None or y is None:
            return None
        return (float(x), float(y))
    
    # Left eye key points
    left_inner = get_lm(133)  # Left eye inner corner
    left_outer = get_lm(33)   # Left eye outer corner
    left_upper = get_lm(159)  # Left eye upper lid
    left_lower = get_lm(145)  # Left eye lower lid
    left_iris = get_lm(468)   # Left iris center
    
    # Right eye key points
    right_inner = get_lm(362)  # Right eye inner corner
    right_outer = get_lm(263)  # Right eye outer corner
    right_upper = get_lm(386)  # Right eye upper lid
    right_lower = get_lm(374)  # Right eye lower lid
    right_iris = get_lm(473)   # Right iris center
    
    def calc_gaze(iris, inner, outer, upper, lower) -> Optional[Dict[str, float]]:
        """Calculate gaze angles from iris and eye boundary positions."""
        if iris is None or inner is None or outer is None or upper is None or lower is None:
            return None
        
        # Eye center (average of inner/outer for x, upper/lower for y)
        eye_center_x = (inner[0] + outer[0]) / 2
        eye_center_y = (upper[1] + lower[1]) / 2
        
        # Eye dimensions
        eye_width = abs(outer[0] - inner[0])
        eye_height = abs(upper[1] - lower[1])
        
        if eye_width <= 1e-9 or eye_height <= 1e-9:
            return None
        
        # Normalized iris position relative to eye center
        dx = (iris[0] - eye_center_x) / eye_width
        dy = (iris[1] - eye_center_y) / eye_height
        
        # Convert to angles using atan2
        yaw = math.degrees(math.atan2(dx, 1.0))
        pitch = math.degrees(math.atan2(dy, 1.0))
        
        return {"yaw": yaw, "pitch": pitch, "dx": dx, "dy": dy}
    
    left_gaze = calc_gaze(left_iris, left_inner, left_outer, left_upper, left_lower)
    right_gaze = calc_gaze(right_iris, right_inner, right_outer, right_upper, right_lower)
    
    if left_gaze is None and right_gaze is None:
        return None
    
    # Combine both eyes if available
    combined_yaw = 0.0
    combined_pitch = 0.0
    count = 0
    
    if left_gaze is not None:
        combined_yaw += left_gaze["yaw"]
        combined_pitch += left_gaze["pitch"]
        count += 1
    
    if right_gaze is not None:
        combined_yaw += right_gaze["yaw"]
        combined_pitch += right_gaze["pitch"]
        count += 1
    
    if count > 0:
        combined_yaw /= count
        combined_pitch /= count
    
    return {
        "left_yaw": left_gaze["yaw"] if left_gaze else None,
        "left_pitch": left_gaze["pitch"] if left_gaze else None,
        "right_yaw": right_gaze["yaw"] if right_gaze else None,
        "right_pitch": right_gaze["pitch"] if right_gaze else None,
        "combined_yaw": combined_yaw,
        "combined_pitch": combined_pitch,
    }


def print_table(data: Dict[str, Any]):
    """Print a formatted table of pitch correlation data."""
    points = data.get("points", [])
    if not points:
        print("No data found")
        return
    
    print("\n" + "="*120)
    print("PITCH CORRELATION DATA ANALYSIS")
    print("="*120)
    
    # Print camera info
    camera_info = data.get("cameraInfo", {})
    if camera_info:
        print(f"\nCamera: backend={camera_info.get('backend')} index={camera_info.get('index')} "
              f"{camera_info.get('width')}x{camera_info.get('height')} @ {camera_info.get('fps'):.1f}fps")
    
    print(f"\nCaptured {len(points)} points")
    print(f"\n{'Name':<28} {'Head Pos':<10} {'Eye Pos':<10} "
          f"{'Head Pitch (°)':<16} {'Eye Pitch (°)':<16} {'Left Eye (°)':<14} {'Right Eye (°)':<14}")
    print("-"*120)
    
    for point in points:
        name = point.get("name", "")
        head_pos = point.get("headPosition", "").upper()
        eye_pos = point.get("eyePosition", "").upper()
        result = point.get("rawResult", {})
        
        # Extract head angles
        head_angles = extract_head_angles(result)
        if head_angles:
            _, head_pitch, _ = head_angles
            head_pitch_str = f"{head_pitch:+7.2f}"
        else:
            head_pitch_str = "n/a"
        
        # Extract eye gaze
        eye_gaze = extract_eye_gaze(result)
        if eye_gaze:
            eye_pitch_str = f"{eye_gaze['combined_pitch']:+8.2f}"
            left_pitch_str = f"{eye_gaze['left_pitch']:+8.2f}" if eye_gaze['left_pitch'] is not None else "n/a"
            right_pitch_str = f"{eye_gaze['right_pitch']:+8.2f}" if eye_gaze['right_pitch'] is not None else "n/a"
        else:
            eye_pitch_str = "n/a"
            left_pitch_str = "n/a"
            right_pitch_str = "n/a"
        
        print(f"{name:<28} {head_pos:<10} {eye_pos:<10} "
              f"{head_pitch_str:<16} {eye_pitch_str:<16} {left_pitch_str:<14} {right_pitch_str:<14}")
    
    print("="*120)


def analyze_pitch_consistency(data: Dict[str, Any]) -> bool:
    """Analyze pitch consistency across head positions.
    
    Returns True if all tests pass, False otherwise.
    """
    points = data.get("points", [])
    if len(points) < 9:
        print("\n[ERROR] Not enough data for pitch consistency analysis (need 9 points)")
        return False
    
    print("\n" + "="*120)
    print("PITCH CONSISTENCY ANALYSIS")
    print("="*120)
    
    # Group points by head position
    head_groups = {
        "down": {},
        "center": {},
        "up": {}
    }
    
    for point in points:
        head_pos = point.get("headPosition")
        eye_pos = point.get("eyePosition")
        name = point.get("name", "")
        
        if head_pos in head_groups and eye_pos:
            head_groups[head_pos][eye_pos] = point
    
    all_tests_passed = True
    
    # Test 1: Verify head pitch values make sense
    print("\n--- TEST 1: HEAD PITCH VALIDATION ---\n")
    
    head_pitch_values = {}
    for head_pos in ["down", "center", "up"]:
        # Use any eye position for head pitch (they should be consistent)
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
                print(f"Head {head_pos.upper():6s}: pitch = {pitch:+7.2f}°")
            else:
                print(f"[FAIL] Head {head_pos.upper()}: could not extract pitch")
                all_tests_passed = False
    
    # Validate head pitch ordering (down < center < up typically, but depends on convention)
    if len(head_pitch_values) == 3:
        down_pitch = head_pitch_values.get("down")
        center_pitch = head_pitch_values.get("center")
        up_pitch = head_pitch_values.get("up")
        
        if down_pitch is not None and center_pitch is not None and up_pitch is not None:
            print(f"\nHead pitch range: {down_pitch:+.2f}° (down) to {up_pitch:+.2f}° (up)")
            
            # Check if head positions are distinct
            if abs(down_pitch - center_pitch) < 5:
                print(f"[WARN] Head DOWN and CENTER positions too close (diff={abs(down_pitch - center_pitch):.2f}°)")
            if abs(center_pitch - up_pitch) < 5:
                print(f"[WARN] Head CENTER and UP positions too close (diff={abs(center_pitch - up_pitch):.2f}°)")
    
    # Test 2: Eye pitch validation for each head position
    print("\n--- TEST 2: EYE PITCH VALIDATION BY HEAD POSITION ---\n")
    
    for head_pos in ["down", "center", "up"]:
        print(f"\nHEAD POSITION: {head_pos.upper()}")
        print("-" * 80)
        
        # Get head pitch for this head position (should be consistent)
        head_pitch = None
        for eye_pos in ["down", "center", "up"]:
            if eye_pos in head_groups[head_pos]:
                point = head_groups[head_pos][eye_pos]
                head_angles = extract_head_angles(point.get("rawResult", {}))
                if head_angles:
                    _, head_pitch, _ = head_angles
                    break
        
        if head_pitch is not None:
            print(f"  Head pitch: {head_pitch:+7.2f}°")
        print()
        
        # Collect eye pitch values for different eye positions
        eye_pitch_values = {}
        for eye_pos in ["down", "center", "up"]:
            if eye_pos in head_groups[head_pos]:
                point = head_groups[head_pos][eye_pos]
                eye_gaze = extract_eye_gaze(point.get("rawResult", {}))
                
                if eye_gaze:
                    eye_pitch = eye_gaze['combined_pitch']
                    eye_pitch_values[eye_pos] = eye_pitch
                    print(f"  Eye {eye_pos.upper():6s}: pitch = {eye_pitch:+8.2f}° "
                          f"(L:{eye_gaze['left_pitch']:+7.2f}° R:{eye_gaze['right_pitch']:+7.2f}°)")
        
        # Check that eye positions show expected ordering: down < center < up
        if len(eye_pitch_values) == 3:
            down_val = eye_pitch_values.get("down")
            center_val = eye_pitch_values.get("center")
            up_val = eye_pitch_values.get("up")
            
            if all(v is not None for v in [down_val, center_val, up_val]):
                print(f"\n  Checking eye pitch ordering (expected: down < center < up):")
                
                # Verify ordering
                if down_val < center_val < up_val:
                    print(f"  [OK] Eye pitch correctly ordered: down < center < up")
                else:
                    print(f"  [FAIL] Eye pitch not correctly ordered")
                    print(f"       down={down_val:+.2f}°, center={center_val:+.2f}°, up={up_val:+.2f}°")
                    all_tests_passed = False
                
                # Verify eye-up is positive
                print(f"\n  Checking up-positive convention:")
                if up_val > 0:
                    print(f"  [OK] Eye UP pitch is positive ({up_val:+.2f}°)")
                else:
                    print(f"  [FAIL] Eye UP pitch is not positive ({up_val:+.2f}°)")
                    all_tests_passed = False
                
                # Verify eye-down is negative
                print(f"\n  Checking down-negative convention:")
                if down_val < 0:
                    print(f"  [OK] Eye DOWN pitch is negative ({down_val:+.2f}°)")
                else:
                    print(f"  [FAIL] Eye DOWN pitch is not negative ({down_val:+.2f}°)")
                    all_tests_passed = False
    
    # Test 3: FaceMeshDao angle validation
    print("\n--- TEST 3: FACEMESHDAO ANGLE VALIDATION ---\n")
    
    # Test eye pitch for each head position
    all_dao_ok = True
    for head_pos in ["down", "center", "up"]:
        print(f"\nHead position: {head_pos.upper()}")
        print("-" * 80)
        
        for eye_pos in ["down", "center", "up"]:
            if eye_pos in head_groups[head_pos]:
                point = head_groups[head_pos][eye_pos]
                event = FaceMeshEvent.from_landmarker_result(
                    _create_mock_result(point.get("rawResult", {}))
                )
                
                dao_eye_pitch = event.left_eye_gaze_pitch
                if dao_eye_pitch is not None:
                    print(f"  Eye {eye_pos.upper():6s}: FaceMeshDao pitch = {dao_eye_pitch:+8.2f}°")
                else:
                    print(f"  Eye {eye_pos.upper():6s}: Could not extract pitch from FaceMeshDao")
                    all_dao_ok = False
    
    if not all_dao_ok:
        print(f"\n[WARN] FaceMeshDao extraction failed for some points")
        all_tests_passed = False
    
    # Summary
    print("\n" + "="*120)
    if all_tests_passed:
        print("[OK] ALL TESTS PASSED - Eye pitch is consistent across head positions")
    else:
        print("[FAIL] SOME TESTS FAILED - Eye pitch consistency issues detected")
    print("="*120 + "\n")
    
    return all_tests_passed


def _create_mock_result(raw_result: Dict[str, Any]) -> Any:
    """Create a mock MediaPipe result object from raw JSON data."""
    from types import SimpleNamespace
    
    # Extract facial transformation matrix
    facial_transformation_matrix = raw_result.get("facial_transformation_matrix")
    if facial_transformation_matrix is None:
        return SimpleNamespace(
            facial_transformation_matrix=None,
            face_landmarks=[]
        )
    
    # Extract face landmarks
    face_landmarks_data = raw_result.get("face_landmarks", [])
    face_landmarks = []
    for lm_data in face_landmarks_data:
        if lm_data is None:
            face_landmarks.append(None)
        else:
            face_landmarks.append(SimpleNamespace(
                x=lm_data.get("x"),
                y=lm_data.get("y"),
                z=lm_data.get("z")
            ))
    
    # Create mock result object
    class MockResult:
        def __init__(self):
            self.facial_transformation_matrixes = [facial_transformation_matrix]
            self.face_landmarks = [face_landmarks]
            self.face_blendshapes = []
    
    return MockResult()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze pitch correlation capture data"
    )
    parser.add_argument(
        "--data-dir", type=str, default="pitch_correlation",
        help="Directory containing pitch correlation data (default: pitch_correlation)"
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data = load_pitch_correlation_data(data_dir)
    
    print_table(data)
    success = analyze_pitch_consistency(data)
    
    # Exit with non-zero status if tests failed
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
