#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test to verify the calibration loading bug fix in main.py
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from facemesh_app.facemesh_dao import (
    CalibrationMatrix, CalibrationPoint,
    compute_calibration_matrix, save_calibration, load_calibration
)

def test_calibration_loading():
    """Test that load_calibration returns a tuple and can be unpacked correctly."""
    print("Testing calibration loading and unpacking...")
    
    # Test 1: Load calibration that doesn't exist (should return defaults)
    print("\nTest 1: Load non-existent calibration")
    calibration, calib_points = load_calibration(profile="test-nonexistent")
    
    # Verify calibration is a CalibrationMatrix object, not a tuple
    assert isinstance(calibration, CalibrationMatrix), f"Expected CalibrationMatrix, got {type(calibration)}"
    assert isinstance(calib_points, list), f"Expected list, got {type(calib_points)}"
    assert calibration.sample_count == 0, "Default calibration should have 0 samples"
    
    print(f"  [OK] Calibration type: {type(calibration).__name__}")
    print(f"  [OK] Calibration points type: {type(calib_points).__name__}")
    print(f"  [OK] Sample count: {calibration.sample_count}")
    
    # Test 2: Verify we can access calibration attributes directly
    print("\nTest 2: Access calibration attributes")
    try:
        yaw = calibration.center_yaw
        pitch = calibration.center_pitch
        samples = calibration.sample_count
        print(f"  [OK] center_yaw: {yaw}")
        print(f"  [OK] center_pitch: {pitch}")
        print(f"  [OK] sample_count: {samples}")
    except AttributeError as e:
        print(f"  [FAIL] Failed to access calibration attributes: {e}")
        raise
    
    # Test 3: Save and load a calibration
    print("\nTest 3: Save and load calibration")
    
    # Create a calibration with proper 9-point pattern
    point_names = ["C", "TL", "TC", "TR", "R", "BR", "BC", "BL", "L"]
    calib_points = []
    
    # Center point
    calib_points.append(CalibrationPoint(
        name="C",
        screen_x=0.5,
        screen_y=0.5,
        raw_eye_yaw=0.0,
        raw_eye_pitch=0.0,
        raw_left_eye_yaw=0.0,
        raw_left_eye_pitch=0.0,
        raw_right_eye_yaw=0.0,
        raw_right_eye_pitch=0.0,
        sample_count=10
    ))
    
    # Other 8 points around the screen
    positions = [
        (0.0, 0.0, -0.1, -0.1),   # TL
        (0.5, 0.0, 0.0, -0.1),    # TC
        (1.0, 0.0, 0.1, -0.1),    # TR
        (1.0, 0.5, 0.1, 0.0),     # R
        (1.0, 1.0, 0.1, 0.1),     # BR
        (0.5, 1.0, 0.0, 0.1),     # BC
        (0.0, 1.0, -0.1, 0.1),    # BL
        (0.0, 0.5, -0.1, 0.0),    # L
    ]
    
    for i, (x, y, yaw, pitch) in enumerate(positions):
        calib_points.append(CalibrationPoint(
            name=point_names[i+1],
            screen_x=x,
            screen_y=y,
            raw_eye_yaw=yaw,
            raw_eye_pitch=pitch,
            raw_left_eye_yaw=yaw,
            raw_left_eye_pitch=pitch,
            raw_right_eye_yaw=yaw,
            raw_right_eye_pitch=pitch,
            sample_count=10
        ))
    
    calib_matrix = compute_calibration_matrix(calib_points)
    save_calibration(calib_matrix, calib_points, profile="test-fix")
    
    # Load it back
    loaded_calib, loaded_points = load_calibration(profile="test-fix")
    
    # Verify loaded calibration is a CalibrationMatrix object
    assert isinstance(loaded_calib, CalibrationMatrix), f"Expected CalibrationMatrix, got {type(loaded_calib)}"
    assert loaded_calib.sample_count > 0, "Loaded calibration should have samples"
    
    # Verify we can access attributes
    yaw = loaded_calib.center_yaw
    pitch = loaded_calib.center_pitch
    samples = loaded_calib.sample_count
    
    print(f"  [OK] Loaded calibration type: {type(loaded_calib).__name__}")
    print(f"  [OK] center_yaw: {yaw:.4f}")
    print(f"  [OK] center_pitch: {pitch:.4f}")
    print(f"  [OK] sample_count: {samples}")
    
    # Cleanup
    test_file = Path("calibration-test-fix.json")
    if test_file.exists():
        test_file.unlink()
        print(f"  [OK] Cleaned up test file")
    
    print("\n[SUCCESS] All tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_calibration_loading()
        sys.exit(0)
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
