#!/usr/bin/env python3
"""
Harmonization Analysis Script
Analyzes captured harmonization data to understand coordinate systems.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import math

# Import FaceMeshDao to test its angle extraction
sys.path.insert(0, str(Path(__file__).parent / "src"))
from facemesh_app.facemesh_dao import FaceMeshEvent


def load_harmonization_data(data_dir: Path = Path("harmonization_data")) -> Dict[str, Any]:
    """Load all harmonization data files."""
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} not found")
        sys.exit(1)
    
    # Try to load combined file first
    combined_file = data_dir / "harmonization_combined.json"
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
        if json_file.name == "harmonization_combined.json":
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
        # Note: x=0 is left side of image in MediaPipe coordinates
        dx = (iris[0] - eye_center_x) / eye_width
        dy = (iris[1] - eye_center_y) / eye_height
        
        # Convert to angles using atan2
        # dx > 0 means iris is to the right of eye center (looking right in original)
        # But since video is mirrored, we need to invert for display coordinates
        # In mirrored view: positive dx = looking left (from user's perspective)
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
    """Print a formatted table of harmonization data."""
    points = data.get("points", [])
    if not points:
        print("No data found")
        return
    
    print("\n" + "="*100)
    print("HARMONIZATION DATA ANALYSIS")
    print("="*100)
    
    # Print camera info
    camera_info = data.get("cameraInfo", {})
    if camera_info:
        print(f"\nCamera: backend={camera_info.get('backend')} index={camera_info.get('index')} "
              f"{camera_info.get('width')}x{camera_info.get('height')} @ {camera_info.get('fps'):.1f}fps")
    
    print(f"\nCaptured {len(points)} points")
    print(f"\n{'Name':<15} {'Type':<8} {'Yaw (°)':<12} {'Pitch (°)':<12} {'Roll (°)':<12} "
          f"{'Eye Yaw (°)':<14} {'Eye Pitch (°)':<14}")
    print("-"*100)
    
    for point in points:
        name = point.get("name", "")
        movement_type = point.get("movementType", "")
        result = point.get("rawResult", {})
        
        # Extract head angles
        head_angles = extract_head_angles(result)
        if head_angles:
            yaw, pitch, roll = head_angles
            yaw_str = f"{yaw:+7.2f}"
            pitch_str = f"{pitch:+7.2f}"
            roll_str = f"{roll:+7.2f}"
        else:
            yaw_str = "n/a"
            pitch_str = "n/a"
            roll_str = "n/a"
        
        # Extract eye gaze
        eye_gaze = extract_eye_gaze(result)
        if eye_gaze:
            eye_yaw_str = f"{eye_gaze['combined_yaw']:+8.2f}"
            eye_pitch_str = f"{eye_gaze['combined_pitch']:+8.2f}"
        else:
            eye_yaw_str = "n/a"
            eye_pitch_str = "n/a"
        
        print(f"{name:<15} {movement_type:<8} {yaw_str:<12} {pitch_str:<12} {roll_str:<12} "
              f"{eye_yaw_str:<14} {eye_pitch_str:<14}")
    
    print("="*100)


def analyze_coordinate_system(data: Dict[str, Any]):
    """Analyze the coordinate system and print observations."""
    points = data.get("points", [])
    if len(points) < 8:
        print("\nNot enough data for coordinate system analysis (need 8 points)")
        return
    
    print("\n" + "="*100)
    print("COORDINATE SYSTEM ANALYSIS")
    print("="*100)
    
    # Group by movement type
    head_points = {p["name"]: p for p in points if p["movementType"] == "head"}
    eye_points = {p["name"]: p for p in points if p["movementType"] == "eye"}
    
    print("\n--- HEAD MOVEMENT ANALYSIS ---\n")
    
    # Analyze yaw (left/right)
    if "head-left" in head_points and "head-right" in head_points:
        left_yaw = extract_head_angles(head_points["head-left"]["rawResult"])
        right_yaw = extract_head_angles(head_points["head-right"]["rawResult"])
        
        if left_yaw and right_yaw:
            print(f"Head Left:  yaw={left_yaw[0]:+.2f}°, pitch={left_yaw[1]:+.2f}°")
            print(f"Head Right: yaw={right_yaw[0]:+.2f}°, pitch={right_yaw[1]:+.2f}°")
            
            yaw_diff = right_yaw[0] - left_yaw[0]
            print(f"\nYaw difference: {yaw_diff:+.2f}°")
            
            if yaw_diff > 0:
                print("[OK] Yaw increases when turning head RIGHT")
            else:
                print("[OK] Yaw increases when turning head LEFT")
            
            # Check if pitch is consistent
            if abs(left_yaw[1] - right_yaw[1]) < 5:
                print("[OK] Pitch is consistent between left/right movements")
            else:
                print(f"[WARN] Pitch varies by {abs(left_yaw[1] - right_yaw[1]):.2f}° between left/right")
    
    # Analyze pitch (up/down)
    if "head-up" in head_points and "head-down" in head_points:
        up_yaw = extract_head_angles(head_points["head-up"]["rawResult"])
        down_yaw = extract_head_angles(head_points["head-down"]["rawResult"])
        
        if up_yaw and down_yaw:
            print(f"\nHead Up:   yaw={up_yaw[0]:+.2f}°, pitch={up_yaw[1]:+.2f}°")
            print(f"Head Down: yaw={down_yaw[0]:+.2f}°, pitch={down_yaw[1]:+.2f}°")
            
            pitch_diff = down_yaw[1] - up_yaw[1]
            print(f"\nPitch difference: {pitch_diff:+.2f}°")
            
            if pitch_diff > 0:
                print("[OK] Pitch increases when tilting head DOWN")
            else:
                print("[OK] Pitch increases when tilting head UP")
            
            # Check if yaw is consistent
            if abs(up_yaw[0] - down_yaw[0]) < 5:
                print("[OK] Yaw is consistent between up/down movements")
            else:
                print(f"[WARN] Yaw varies by {abs(up_yaw[0] - down_yaw[0]):.2f}° between up/down")
    
    print("\n--- EYE GAZE ANALYSIS ---\n")
    
    # Analyze eye yaw (left/right)
    if "eye-left" in eye_points and "eye-right" in eye_points:
        left_gaze = extract_eye_gaze(eye_points["eye-left"]["rawResult"])
        right_gaze = extract_eye_gaze(eye_points["eye-right"]["rawResult"])
        
        if left_gaze and right_gaze:
            print(f"Eye Left:  combined_yaw={left_gaze['combined_yaw']:+.2f}°, combined_pitch={left_gaze['combined_pitch']:+.2f}°")
            print(f"Eye Right: combined_yaw={right_gaze['combined_yaw']:+.2f}°, combined_pitch={right_gaze['combined_pitch']:+.2f}°")
            
            yaw_diff = right_gaze["combined_yaw"] - left_gaze["combined_yaw"]
            print(f"\nEye yaw difference: {yaw_diff:+.2f}°")
            
            if yaw_diff > 0:
                print("[OK] Eye yaw increases when looking RIGHT")
            else:
                print("[OK] Eye yaw increases when looking LEFT")
    
    # Analyze eye pitch (up/down)
    if "eye-up" in eye_points and "eye-down" in eye_points:
        up_gaze = extract_eye_gaze(eye_points["eye-up"]["rawResult"])
        down_gaze = extract_eye_gaze(eye_points["eye-down"]["rawResult"])
        
        if up_gaze and down_gaze:
            print(f"\nEye Up:   combined_yaw={up_gaze['combined_yaw']:+.2f}°, combined_pitch={up_gaze['combined_pitch']:+.2f}°")
            print(f"Eye Down: combined_yaw={down_gaze['combined_yaw']:+.2f}°, combined_pitch={down_gaze['combined_pitch']:+.2f}°")
            
            pitch_diff = down_gaze["combined_pitch"] - up_gaze["combined_pitch"]
            print(f"\nEye pitch difference: {pitch_diff:+.2f}°")
            
            if pitch_diff > 0:
                print("[OK] Eye pitch increases when looking DOWN")
            else:
                print("[OK] Eye pitch increases when looking UP")

    print("\n--- FACEMESHDAO ANGLE VALIDATION ---\n")
    
    # Test FaceMeshDao angle extraction
    all_ok = True
    
    # Check head yaw convention
    if "head-left" in head_points and "head-right" in head_points:
        left_event = FaceMeshEvent.from_landmarker_result(
            _create_mock_result(head_points["head-left"]["rawResult"])
        )
        right_event = FaceMeshEvent.from_landmarker_result(
            _create_mock_result(head_points["head-right"]["rawResult"])
        )
        
        left_dao_yaw = left_event.head_yaw
        right_dao_yaw = right_event.head_yaw
        
        if left_dao_yaw is not None and right_dao_yaw is not None:
            print(f"FaceMeshDao Head Left:  yaw={left_dao_yaw:+.2f}°")
            print(f"FaceMeshDao Head Right: yaw={right_dao_yaw:+.2f}°")
            
            yaw_diff = right_dao_yaw - left_dao_yaw
            print(f"FaceMeshDao Yaw difference: {yaw_diff:+.2f}°")
            
            if yaw_diff > 0:
                print("[OK] FaceMeshDao: Yaw increases when turning head RIGHT")
            else:
                print("[FAIL] FaceMeshDao: Yaw does NOT follow right-positive convention")
                all_ok = False
        else:
            print("[WARN] FaceMeshDao: Could not extract head yaw angles")
    
    # Check head pitch convention
    if "head-up" in head_points and "head-down" in head_points:
        up_event = FaceMeshEvent.from_landmarker_result(
            _create_mock_result(head_points["head-up"]["rawResult"])
        )
        down_event = FaceMeshEvent.from_landmarker_result(
            _create_mock_result(head_points["head-down"]["rawResult"])
        )
        
        up_dao_pitch = up_event.head_pitch
        down_dao_pitch = down_event.head_pitch
        
        if up_dao_pitch is not None and down_dao_pitch is not None:
            print(f"\nFaceMeshDao Head Up:   pitch={up_dao_pitch:+.2f}°")
            print(f"FaceMeshDao Head Down: pitch={down_dao_pitch:+.2f}°")
            
            pitch_diff = down_dao_pitch - up_dao_pitch
            print(f"FaceMeshDao Pitch difference: {pitch_diff:+.2f}°")
            
            if pitch_diff < 0:
                print("[OK] FaceMeshDao: Pitch increases when tilting head UP")
            else:
                print("[FAIL] FaceMeshDao: Pitch does NOT follow up-positive convention")
                all_ok = False
        else:
            print("[WARN] FaceMeshDao: Could not extract head pitch angles")
    
    # Check eye yaw convention
    if "eye-left" in eye_points and "eye-right" in eye_points:
        left_event = FaceMeshEvent.from_landmarker_result(
            _create_mock_result(eye_points["eye-left"]["rawResult"])
        )
        right_event = FaceMeshEvent.from_landmarker_result(
            _create_mock_result(eye_points["eye-right"]["rawResult"])
        )
        
        left_dao_yaw = left_event.left_eye_gaze_yaw
        right_dao_yaw = right_event.left_eye_gaze_yaw
        
        if left_dao_yaw is not None and right_dao_yaw is not None:
            print(f"\nFaceMeshDao Eye Left:  left_eye_yaw={left_dao_yaw:+.2f}°")
            print(f"FaceMeshDao Eye Right: left_eye_yaw={right_dao_yaw:+.2f}°")
            
            yaw_diff = right_dao_yaw - left_dao_yaw
            print(f"FaceMeshDao Eye yaw difference: {yaw_diff:+.2f}°")
            
            if yaw_diff > 0:
                print("[OK] FaceMeshDao: Eye yaw increases when looking RIGHT")
            else:
                print("[FAIL] FaceMeshDao: Eye yaw does NOT follow right-positive convention")
                print(f"  -> Looking LEFT gives {left_dao_yaw:+.2f}°, looking RIGHT gives {right_dao_yaw:+.2f}°")
                print(f"  -> Expected: looking RIGHT should be MORE positive than LEFT")
                all_ok = False
        else:
            print("[WARN] FaceMeshDao: Could not extract eye yaw angles")
    
    # Check eye pitch convention
    if "eye-up" in eye_points and "eye-down" in eye_points:
        up_event = FaceMeshEvent.from_landmarker_result(
            _create_mock_result(eye_points["eye-up"]["rawResult"])
        )
        down_event = FaceMeshEvent.from_landmarker_result(
            _create_mock_result(eye_points["eye-down"]["rawResult"])
        )
        
        up_dao_pitch = up_event.left_eye_gaze_pitch
        down_dao_pitch = down_event.left_eye_gaze_pitch
        
        if up_dao_pitch is not None and down_dao_pitch is not None:
            print(f"\nFaceMeshDao Eye Up:   left_eye_pitch={up_dao_pitch:+.2f}°")
            print(f"FaceMeshDao Eye Down: left_eye_pitch={down_dao_pitch:+.2f}°")
            
            pitch_diff = down_dao_pitch - up_dao_pitch
            print(f"FaceMeshDao Eye pitch difference: {pitch_diff:+.2f}°")
            
            if pitch_diff < 0:
                print("[OK] FaceMeshDao: Eye pitch increases when looking UP")
            else:
                print("[FAIL] FaceMeshDao: Eye pitch does NOT follow up-positive convention")
                print(f"  -> Looking UP gives {up_dao_pitch:+.2f}°, looking DOWN gives {down_dao_pitch:+.2f}°")
                print(f"  -> Expected: looking UP should be MORE positive than DOWN")
                all_ok = False
        else:
            print("[WARN] FaceMeshDao: Could not extract eye pitch angles")
    
    print("\n" + "="*100)
    if all_ok:
        print("[OK] FaceMeshDao ANGLE CONVENTION VALIDATION: ALL TESTS PASSED")
    else:
        print("[FAIL] FaceMeshDao ANGLE CONVENTION VALIDATION: SOME TESTS FAILED")
    print("="*100 + "\n")


def _create_mock_result(raw_result: Dict[str, Any]) -> Any:
    """Create a mock MediaPipe result object from raw JSON data.
    
    This creates a minimal mock object that FaceMeshEvent can work with.
    The mock has the same structure as a real MediaPipe FaceLandmarker result.
    """
    from types import SimpleNamespace
    
    # Extract facial transformation matrix
    facial_transformation_matrix = raw_result.get("facial_transformation_matrix")
    if facial_transformation_matrix is None:
        # Return minimal mock
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
            # FaceMeshDao expects facial_transformation_matrixes (plural)
            self.facial_transformation_matrixes = [facial_transformation_matrix]
            self.face_landmarks = [face_landmarks]  # Wrap in list for multi-face support
            self.face_blendshapes = []
    
    return MockResult()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze harmonization capture data"
    )
    parser.add_argument(
        "--data-dir", type=str, default="harmonization_data",
        help="Directory containing harmonization data (default: harmonization_data)"
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data = load_harmonization_data(data_dir)
    
    print_table(data)
    analyze_coordinate_system(data)


if __name__ == "__main__":
    main()
