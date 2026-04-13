"""Test script to verify calibration support in FaceMeshEvent."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from facemesh_app.facemesh_dao import FaceMeshEvent, CalibrationMatrix, CalibrationPoint, compute_calibration_matrix


def test_apply_calibration():
    """Test the _apply_calibration method with known values."""
    print("Testing _apply_calibration method...")
    
    # Create a simple calibration matrix
    # center_yaw = 5.0, center_pitch = 3.0
    # Identity matrix (no transformation)
    calib = CalibrationMatrix(
        center_yaw=5.0,
        center_pitch=3.0,
        matrix_yaw_yaw=1.0,
        matrix_yaw_pitch=0.0,
        matrix_pitch_yaw=0.0,
        matrix_pitch_pitch=1.0,
        sample_count=100,
        timestamp_ms=0
    )
    
    # Create a FaceMeshEvent
    event = FaceMeshEvent(calibration=calib)
    
    # Test case 1: Raw values at center (should return center values)
    raw_yaw = 5.0
    raw_pitch = 3.0
    calibrated_yaw, calibrated_pitch = event._apply_calibration(raw_yaw, raw_pitch, calib)
    print(f"  Test 1 - Center: raw=({raw_yaw}, {raw_pitch}), calibrated=({calibrated_yaw}, {calibrated_pitch})")
    assert abs(calibrated_yaw - 5.0) < 1e-9, f"Expected 5.0, got {calibrated_yaw}"
    assert abs(calibrated_pitch - 3.0) < 1e-9, f"Expected 3.0, got {calibrated_pitch}"
    print("  [PASS] Test 1 passed")
    
    # Test case 2: Raw values offset from center (should maintain offset with identity matrix)
    raw_yaw = 7.0
    raw_pitch = 5.0
    calibrated_yaw, calibrated_pitch = event._apply_calibration(raw_yaw, raw_pitch, calib)
    print(f"  Test 2 - Offset: raw=({raw_yaw}, {raw_pitch}), calibrated=({calibrated_yaw}, {calibrated_pitch})")
    assert abs(calibrated_yaw - 7.0) < 1e-9, f"Expected 7.0, got {calibrated_yaw}"
    assert abs(calibrated_pitch - 5.0) < 1e-9, f"Expected 5.0, got {calibrated_pitch}"
    print("  [PASS] Test 2 passed")
    
    # Test case 3: Non-identity matrix with cross-talk
    calib_cross = CalibrationMatrix(
        center_yaw=0.0,
        center_pitch=0.0,
        matrix_yaw_yaw=2.0,    # Scale yaw by 2
        matrix_yaw_pitch=0.5,   # Add 0.5 * pitch to yaw
        matrix_pitch_yaw=0.3,   # Add 0.3 * yaw to pitch
        matrix_pitch_pitch=1.5, # Scale pitch by 1.5
        sample_count=100,
        timestamp_ms=0
    )
    
    raw_yaw = 10.0
    raw_pitch = 5.0
    calibrated_yaw, calibrated_pitch = event._apply_calibration(raw_yaw, raw_pitch, calib_cross)
    expected_yaw = 2.0 * 10.0 + 0.5 * 5.0  # = 20.0 + 2.5 = 22.5
    expected_pitch = 0.3 * 10.0 + 1.5 * 5.0  # = 3.0 + 7.5 = 10.5
    print(f"  Test 3 - Cross-talk: raw=({raw_yaw}, {raw_pitch}), calibrated=({calibrated_yaw}, {calibrated_pitch})")
    assert abs(calibrated_yaw - expected_yaw) < 1e-9, f"Expected {expected_yaw}, got {calibrated_yaw}"
    assert abs(calibrated_pitch - expected_pitch) < 1e-9, f"Expected {expected_pitch}, got {calibrated_pitch}"
    print("  [PASS] Test 3 passed")
    
    print("[PASS] All _apply_calibration tests passed!\n")


def test_calibrated_properties_without_raw_values():
    """Test that calibrated properties return None when raw values are None."""
    print("Testing calibrated properties without raw values...")
    
    calib = CalibrationMatrix(
        center_yaw=5.0,
        center_pitch=3.0,
        matrix_yaw_yaw=1.0,
        matrix_yaw_pitch=0.0,
        matrix_pitch_yaw=0.0,
        matrix_pitch_pitch=1.0,
        sample_count=100,
        timestamp_ms=0
    )
    
    # Create event without result (no raw gaze values)
    event = FaceMeshEvent(calibration=calib)
    
    # All calibrated properties should return None when raw values are None
    assert event.calibrated_left_eye_gaze_yaw is None, "Expected None for left_eye_gaze_yaw without raw value"
    assert event.calibrated_right_eye_gaze_yaw is None, "Expected None for right_eye_gaze_yaw without raw value"
    assert event.calibrated_left_eye_gaze_pitch is None, "Expected None for left_eye_gaze_pitch without raw value"
    assert event.calibrated_right_eye_gaze_pitch is None, "Expected None for right_eye_gaze_pitch without raw value"
    assert event.calibrated_combined_eye_gaze_yaw is None, "Expected None for combined_yaw without raw values"
    assert event.calibrated_combined_eye_gaze_pitch is None, "Expected None for combined_pitch without raw values"
    
    print("[PASS] All properties correctly return None without raw values\n")


def test_calibrated_properties_without_calibration():
    """Test that calibrated properties return None when calibration is None."""
    print("Testing calibrated properties without calibration...")
    
    # Create event without calibration
    event = FaceMeshEvent(calibration=None)
    
    # All calibrated properties should return None when calibration is None
    assert event.calibrated_left_eye_gaze_yaw is None, "Expected None without calibration"
    assert event.calibrated_right_eye_gaze_yaw is None, "Expected None without calibration"
    assert event.calibrated_left_eye_gaze_pitch is None, "Expected None without calibration"
    assert event.calibrated_right_eye_gaze_pitch is None, "Expected None without calibration"
    assert event.calibrated_combined_eye_gaze_yaw is None, "Expected None without calibration"
    assert event.calibrated_combined_eye_gaze_pitch is None, "Expected None without calibration"
    
    print("[PASS] All properties correctly return None without calibration\n")


def test_raw_properties_unchanged():
    """Test that raw properties remain unchanged (backward compatibility)."""
    print("Testing raw properties remain unchanged...")
    
    calib = CalibrationMatrix(
        center_yaw=5.0,
        center_pitch=3.0,
        matrix_yaw_yaw=1.0,
        matrix_yaw_pitch=0.0,
        matrix_pitch_yaw=0.0,
        matrix_pitch_pitch=1.0,
        sample_count=100,
        timestamp_ms=0
    )
    
    # Create event with calibration
    event = FaceMeshEvent(calibration=calib)
    
    # Raw properties should still exist and return None without result
    assert hasattr(event, 'left_eye_gaze_yaw'), "Raw property left_eye_gaze_yaw should exist"
    assert hasattr(event, 'right_eye_gaze_yaw'), "Raw property right_eye_gaze_yaw should exist"
    assert hasattr(event, 'left_eye_gaze_pitch'), "Raw property left_eye_gaze_pitch should exist"
    assert hasattr(event, 'right_eye_gaze_pitch'), "Raw property right_eye_gaze_pitch should exist"
    
    # They should return None without result
    assert event.left_eye_gaze_yaw is None
    assert event.right_eye_gaze_yaw is None
    assert event.left_eye_gaze_pitch is None
    assert event.right_eye_gaze_pitch is None
    
    print("[PASS] Raw properties remain unchanged and accessible\n")


def test_compute_calibration_matrix():
    """Test the compute_calibration_matrix function."""
    print("Testing compute_calibration_matrix function...")
    
    # Test case 1: Normal case with 9 points including center
    print("  Test 1 - Normal case with 9 points...")
    
    # Create center point
    center = CalibrationPoint(
        name="C",
        screen_x=0.5,
        screen_y=0.5,
        raw_eye_yaw=5.0,
        raw_eye_pitch=3.0,
        raw_left_eye_yaw=4.8,
        raw_left_eye_pitch=2.9,
        raw_right_eye_yaw=5.2,
        raw_right_eye_pitch=3.1,
        sample_count=100
    )
    
    # Create 8 edge points with known deviations
    # We'll create points where the transformation should be straightforward
    edge_points = [
        # Top-left
        CalibrationPoint(
            name="TL",
            screen_x=0.0,
            screen_y=0.0,
            raw_eye_yaw=0.0,  # 5.0 deviation in yaw
            raw_eye_pitch=0.0,  # 3.0 deviation in pitch
            raw_left_eye_yaw=-0.2,
            raw_left_eye_pitch=-0.1,
            raw_right_eye_yaw=0.2,
            raw_right_eye_pitch=0.1,
            sample_count=50
        ),
        # Top-center
        CalibrationPoint(
            name="TC",
            screen_x=0.5,
            screen_y=0.0,
            raw_eye_yaw=5.0,
            raw_eye_pitch=0.0,
            raw_left_eye_yaw=4.8,
            raw_left_eye_pitch=-0.1,
            raw_right_eye_yaw=5.2,
            raw_right_eye_pitch=0.1,
            sample_count=50
        ),
        # Top-right
        CalibrationPoint(
            name="TR",
            screen_x=1.0,
            screen_y=0.0,
            raw_eye_yaw=10.0,
            raw_eye_pitch=0.0,
            raw_left_eye_yaw=9.8,
            raw_left_eye_pitch=-0.1,
            raw_right_eye_yaw=10.2,
            raw_right_eye_pitch=0.1,
            sample_count=50
        ),
        # Right
        CalibrationPoint(
            name="R",
            screen_x=1.0,
            screen_y=0.5,
            raw_eye_yaw=10.0,
            raw_eye_pitch=3.0,
            raw_left_eye_yaw=9.8,
            raw_left_eye_pitch=2.9,
            raw_right_eye_yaw=10.2,
            raw_right_eye_pitch=3.1,
            sample_count=50
        ),
        # Bottom-right
        CalibrationPoint(
            name="BR",
            screen_x=1.0,
            screen_y=1.0,
            raw_eye_yaw=10.0,
            raw_eye_pitch=6.0,
            raw_left_eye_yaw=9.8,
            raw_left_eye_pitch=5.9,
            raw_right_eye_yaw=10.2,
            raw_right_eye_pitch=6.1,
            sample_count=50
        ),
        # Bottom-center
        CalibrationPoint(
            name="BC",
            screen_x=0.5,
            screen_y=1.0,
            raw_eye_yaw=5.0,
            raw_eye_pitch=6.0,
            raw_left_eye_yaw=4.8,
            raw_left_eye_pitch=5.9,
            raw_right_eye_yaw=5.2,
            raw_right_eye_pitch=6.1,
            sample_count=50
        ),
        # Bottom-left
        CalibrationPoint(
            name="BL",
            screen_x=0.0,
            screen_y=1.0,
            raw_eye_yaw=0.0,
            raw_eye_pitch=6.0,
            raw_left_eye_yaw=-0.2,
            raw_left_eye_pitch=5.9,
            raw_right_eye_yaw=0.2,
            raw_right_eye_pitch=6.1,
            sample_count=50
        ),
        # Left
        CalibrationPoint(
            name="L",
            screen_x=0.0,
            screen_y=0.5,
            raw_eye_yaw=0.0,
            raw_eye_pitch=3.0,
            raw_left_eye_yaw=-0.2,
            raw_left_eye_pitch=2.9,
            raw_right_eye_yaw=0.2,
            raw_right_eye_pitch=3.1,
            sample_count=50
        )
    ]
    
    # Compute calibration matrix
    calib_matrix = compute_calibration_matrix([center] + edge_points)
    
    print(f"    Center values: yaw={calib_matrix.center_yaw}, pitch={calib_matrix.center_pitch}")
    print(f"    Matrix coefficients:")
    print(f"      yaw_yaw: {calib_matrix.matrix_yaw_yaw}")
    print(f"      yaw_pitch: {calib_matrix.matrix_yaw_pitch}")
    print(f"      pitch_yaw: {calib_matrix.matrix_pitch_yaw}")
    print(f"      pitch_pitch: {calib_matrix.matrix_pitch_pitch}")
    print(f"    Sample count: {calib_matrix.sample_count}")
    print(f"    Timestamp: {calib_matrix.timestamp_ms}")
    
    # Verify center values
    assert abs(calib_matrix.center_yaw - 5.0) < 1e-9, f"Expected center_yaw=5.0, got {calib_matrix.center_yaw}"
    assert abs(calib_matrix.center_pitch - 3.0) < 1e-9, f"Expected center_pitch=3.0, got {calib_matrix.center_pitch}"
    
    # Verify sample count
    assert calib_matrix.sample_count == 100 + 8 * 50, f"Expected sample_count=500, got {calib_matrix.sample_count}"
    
    # Verify timestamp is reasonable (within last 10 seconds)
    import time
    current_time = int(time.time() * 1000)
    assert current_time - calib_matrix.timestamp_ms < 10000, "Timestamp should be recent"
    
    print("  [PASS] Test 1 passed")
    
    # Test case 2: Error case - fewer than 9 points
    print("  Test 2 - Error case with fewer than 9 points...")
    try:
        compute_calibration_matrix([center] + edge_points[:7])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"    Correctly raised ValueError: {e}")
        assert "at least 9 points" in str(e), "Error message should mention point count"
    print("  [PASS] Test 2 passed")
    
    # Test case 3: Error case - missing center point
    print("  Test 3 - Error case with missing center point...")
    # Create 9 points but none named "C"
    no_center_points = []
    for i, name in enumerate(["TL", "TC", "TR", "R", "BR", "BC", "BL", "L", "M"]):
        no_center_points.append(CalibrationPoint(
            name=name,
            screen_x=0.5,
            screen_y=0.5,
            raw_eye_yaw=5.0 + i,
            raw_eye_pitch=3.0 + i,
            raw_left_eye_yaw=4.8 + i,
            raw_left_eye_pitch=2.9 + i,
            raw_right_eye_yaw=5.2 + i,
            raw_right_eye_pitch=3.1 + i,
            sample_count=10
        ))
    
    try:
        compute_calibration_matrix(no_center_points)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"    Correctly raised ValueError: {e}")
        assert "center point" in str(e).lower(), "Error message should mention center point"
    print("  [PASS] Test 3 passed")
    
    # Test case 4: Matrix computation with realistic data
    print("  Test 4 - Matrix with realistic transformation...")
    
    # Create points with a known transformation
    # Let's say: screen_x = 0.2 * raw_yaw_dev + 0.1 * raw_pitch_dev
    #           screen_y = 0.05 * raw_yaw_dev + 0.3 * raw_pitch_dev
    
    realistic_points = [
        CalibrationPoint(
            name="C",
            screen_x=0.5,
            screen_y=0.5,
            raw_eye_yaw=0.0,
            raw_eye_pitch=0.0,
            raw_left_eye_yaw=0.0,
            raw_left_eye_pitch=0.0,
            raw_right_eye_yaw=0.0,
            raw_right_eye_pitch=0.0,
            sample_count=100
        )
    ]
    
    # Add 8 edge points following the transformation
    for name, sx, sy in [
        ("TL", 0.0, 0.0), ("TC", 0.5, 0.0), ("TR", 1.0, 0.0),
        ("R", 1.0, 0.5), ("BR", 1.0, 1.0), ("BC", 0.5, 1.0),
        ("BL", 0.0, 1.0), ("L", 0.0, 0.5)
    ]:
        # Reverse the transformation to get raw gaze values
        # sx_dev = 0.2 * ry_dev + 0.1 * rp_dev
        # sy_dev = 0.05 * ry_dev + 0.3 * rp_dev
        # We need to solve this linear system
        
        sx_dev = sx - 0.5
        sy_dev = sy - 0.5
        
        # Solve: [0.2 0.1][ry_dev] = [sx_dev]
        #         [0.05 0.3][rp_dev]   [sy_dev]
        # Determinant = 0.2*0.3 - 0.05*0.1 = 0.06 - 0.005 = 0.055
        
        det = 0.055
        ry_dev = (0.3 * sx_dev - 0.1 * sy_dev) / det
        rp_dev = (-0.05 * sx_dev + 0.2 * sy_dev) / det
        
        realistic_points.append(CalibrationPoint(
            name=name,
            screen_x=sx,
            screen_y=sy,
            raw_eye_yaw=ry_dev,
            raw_eye_pitch=rp_dev,
            raw_left_eye_yaw=ry_dev * 0.98,
            raw_left_eye_pitch=rp_dev * 0.97,
            raw_right_eye_yaw=ry_dev * 1.02,
            raw_right_eye_pitch=rp_dev * 1.03,
            sample_count=50
        ))
    
    calib_real = compute_calibration_matrix(realistic_points)
    
    # Verify the computed coefficients match expected values
    # Note: Allow some tolerance due to floating point arithmetic
    print(f"    Matrix coefficients:")
    print(f"      yaw_yaw: {calib_real.matrix_yaw_yaw} (expected ~0.2)")
    print(f"      yaw_pitch: {calib_real.matrix_yaw_pitch} (expected ~0.1)")
    print(f"      pitch_yaw: {calib_real.matrix_pitch_yaw} (expected ~0.05)")
    print(f"      pitch_pitch: {calib_real.matrix_pitch_pitch} (expected ~0.3)")
    
    assert abs(calib_real.matrix_yaw_yaw - 0.2) < 0.01, f"Expected ~0.2, got {calib_real.matrix_yaw_yaw}"
    assert abs(calib_real.matrix_yaw_pitch - 0.1) < 0.01, f"Expected ~0.1, got {calib_real.matrix_yaw_pitch}"
    assert abs(calib_real.matrix_pitch_yaw - 0.05) < 0.01, f"Expected ~0.05, got {calib_real.matrix_pitch_yaw}"
    assert abs(calib_real.matrix_pitch_pitch - 0.3) < 0.01, f"Expected ~0.3, got {calib_real.matrix_pitch_pitch}"
    
    print("  [PASS] Test 4 passed")
    
    print("[PASS] All compute_calibration_matrix tests passed!\n")


def main():
    """Run all calibration tests."""
    print("=" * 60)
    print("Testing FaceMeshEvent Calibration Support")
    print("=" * 60)
    print()
    
    test_apply_calibration()
    test_calibrated_properties_without_raw_values()
    test_calibrated_properties_without_calibration()
    test_raw_properties_unchanged()
    test_compute_calibration_matrix()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
