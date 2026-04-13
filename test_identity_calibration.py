"""
Test identity calibration behavior to ensure app works without calibration data.

This test verifies that:
1. CalibrationMatrix defaults create an identity transformation
2. When calibration is None, calibrated properties use identity transform
3. Identity transform produces same values as raw values
"""

import math
from src.facemesh_app.facemesh_dao import FaceMeshEvent, CalibrationMatrix


def test_calibration_matrix_defaults():
    """Test that CalibrationMatrix creates identity transformation with defaults."""
    calib = CalibrationMatrix()
    
    assert calib.center_yaw == 0.0, "center_yaw should default to 0.0"
    assert calib.center_pitch == 0.0, "center_pitch should default to 0.0"
    assert calib.matrix_yaw_yaw == 1.0, "matrix_yaw_yaw should default to 1.0"
    assert calib.matrix_yaw_pitch == 0.0, "matrix_yaw_pitch should default to 0.0"
    assert calib.matrix_pitch_yaw == 0.0, "matrix_pitch_yaw should default to 0.0"
    assert calib.matrix_pitch_pitch == 1.0, "matrix_pitch_pitch should default to 1.0"
    assert calib.sample_count == 0, "sample_count should default to 0"
    assert calib.timestamp_ms == 0, "timestamp_ms should default to 0"
    
    print("[PASS] CalibrationMatrix default values create identity transformation")


def test_identity_transform_preserves_values():
    """Test that identity transform preserves raw values."""
    calib = CalibrationMatrix()
    
    # Test with various raw values
    test_cases = [
        (0.0, 0.0),
        (5.0, -3.0),
        (-10.5, 7.2),
        (45.0, -30.0),
    ]
    
    for raw_yaw, raw_pitch in test_cases:
        # Apply calibration
        event = FaceMeshEvent()
        calibrated_yaw, calibrated_pitch = event._apply_calibration(
            raw_yaw, raw_pitch, calib
        )
        
        # Identity transform should preserve values
        assert abs(calibrated_yaw - raw_yaw) < 1e-9, \
            f"Identity transform should preserve yaw: {raw_yaw} != {calibrated_yaw}"
        assert abs(calibrated_pitch - raw_pitch) < 1e-9, \
            f"Identity transform should preserve pitch: {raw_pitch} != {calibrated_pitch}"
    
    print("[PASS] Identity transformation preserves raw values")


def test_calibrated_properties_without_calibration():
    """Test that calibrated properties work when calibration is None."""
    # Create event without calibration (calibration=None is default)
    event = FaceMeshEvent()
    
    # Simulate having raw gaze values by setting up a mock result
    # Since we can't easily create a full MediaPipe result, we'll test the logic
    # by directly testing the calibration application
    
    # Test the property logic with simulated raw values
    raw_yaw_test = 5.0
    raw_pitch_test = -3.0
    
    # Manually set what the raw properties would return
    class MockFaceMeshEvent(FaceMeshEvent):
        @property
        def left_eye_gaze_yaw(self):
            return raw_yaw_test
        
        @property
        def left_eye_gaze_pitch(self):
            return raw_pitch_test
        
        @property
        def right_eye_gaze_yaw(self):
            return raw_yaw_test * 0.9  # Slightly different
        
        @property
        def right_eye_gaze_pitch(self):
            return raw_pitch_test * 0.9
    
    mock_event = MockFaceMeshEvent()
    
    # Verify calibration is None
    assert mock_event.calibration is None, "Calibration should be None by default"
    
    # Get calibrated values - should use identity transform
    calib_left_yaw = mock_event.calibrated_left_eye_gaze_yaw
    calib_left_pitch = mock_event.calibrated_left_eye_gaze_pitch
    calib_right_yaw = mock_event.calibrated_right_eye_gaze_yaw
    calib_right_pitch = mock_event.calibrated_right_eye_gaze_pitch
    calib_combined_yaw = mock_event.calibrated_combined_eye_gaze_yaw
    calib_combined_pitch = mock_event.calibrated_combined_eye_gaze_pitch
    
    # Verify calibrated values match raw values (identity transform)
    assert calib_left_yaw is not None, "calibrated_left_eye_gaze_yaw should not be None"
    assert calib_left_pitch is not None, "calibrated_left_eye_gaze_pitch should not be None"
    assert calib_right_yaw is not None, "calibrated_right_eye_gaze_yaw should not be None"
    assert calib_right_pitch is not None, "calibrated_right_eye_gaze_pitch should not be None"
    assert calib_combined_yaw is not None, "calibrated_combined_eye_gaze_yaw should not be None"
    assert calib_combined_pitch is not None, "calibrated_combined_eye_gaze_pitch should not be None"
    
    # Identity transform should preserve raw values
    assert abs(calib_left_yaw - raw_yaw_test) < 1e-9, \
        f"Left calibrated yaw should match raw: {raw_yaw_test} != {calib_left_yaw}"
    assert abs(calib_left_pitch - raw_pitch_test) < 1e-9, \
        f"Left calibrated pitch should match raw: {raw_pitch_test} != {calib_left_pitch}"
    
    print("[PASS] Calibrated properties work without calibration (use identity transform)")


def test_with_calibration_vs_without():
    """Test that app works identically with or without calibration when using identity."""
    # Create identity calibration
    identity_calib = CalibrationMatrix()
    
    class MockFaceMeshEvent(FaceMeshEvent):
        def __init__(self, calibration=None):
            super().__init__(calibration=calibration)
            self._raw_yaw = 10.0
            self._raw_pitch = -5.0
        
        @property
        def left_eye_gaze_yaw(self):
            return self._raw_yaw
        
        @property
        def left_eye_gaze_pitch(self):
            return self._raw_pitch
        
        @property
        def right_eye_gaze_yaw(self):
            return self._raw_yaw
        
        @property
        def right_eye_gaze_pitch(self):
            return self._raw_pitch
    
    # Event without calibration
    event_no_calib = MockFaceMeshEvent(calibration=None)
    
    # Event with identity calibration
    event_with_calib = MockFaceMeshEvent(calibration=identity_calib)
    
    # Both should produce identical results
    assert abs(event_no_calib.calibrated_left_eye_gaze_yaw - 
               event_with_calib.calibrated_left_eye_gaze_yaw) < 1e-9
    assert abs(event_no_calib.calibrated_left_eye_gaze_pitch - 
               event_with_calib.calibrated_left_eye_gaze_pitch) < 1e-9
    assert abs(event_no_calib.calibrated_right_eye_gaze_yaw - 
               event_with_calib.calibrated_right_eye_gaze_yaw) < 1e-9
    assert abs(event_no_calib.calibrated_right_eye_gaze_pitch - 
               event_with_calib.calibrated_right_eye_gaze_pitch) < 1e-9
    assert abs(event_no_calib.calibrated_combined_eye_gaze_yaw - 
               event_with_calib.calibrated_combined_eye_gaze_yaw) < 1e-9
    assert abs(event_no_calib.calibrated_combined_eye_gaze_pitch - 
               event_with_calib.calibrated_combined_eye_gaze_pitch) < 1e-9
    
    print("[PASS] App works identically with or without identity calibration")


def test_non_identity_calibration_affects_values():
    """Test that non-identity calibration actually transforms values."""
    # Create a non-identity calibration
    calib = CalibrationMatrix(
        center_yaw=2.0,
        center_pitch=1.0,
        matrix_yaw_yaw=1.5,
        matrix_yaw_pitch=0.2,
        matrix_pitch_yaw=0.1,
        matrix_pitch_pitch=1.3,
        sample_count=100,
        timestamp_ms=1234567890
    )
    
    class MockFaceMeshEvent(FaceMeshEvent):
        def __init__(self, calibration):
            super().__init__(calibration=calibration)
            self._raw_yaw = 10.0
            self._raw_pitch = -5.0
        
        @property
        def left_eye_gaze_yaw(self):
            return self._raw_yaw
        
        @property
        def left_eye_gaze_pitch(self):
            return self._raw_pitch
        
        @property
        def right_eye_gaze_yaw(self):
            return self._raw_yaw
        
        @property
        def right_eye_gaze_pitch(self):
            return self._raw_pitch
    
    event = MockFaceMeshEvent(calibration=calib)
    
    # Get calibrated values
    calib_yaw = event.calibrated_left_eye_gaze_yaw
    calib_pitch = event.calibrated_left_eye_gaze_pitch
    
    # Should be different from raw values
    raw_yaw = 10.0
    raw_pitch = -5.0
    
    assert abs(calib_yaw - raw_yaw) > 0.1, \
        "Non-identity calibration should change yaw value"
    assert abs(calib_pitch - raw_pitch) > 0.1, \
        "Non-identity calibration should change pitch value"
    
    print("[PASS] Non-identity calibration correctly transforms values")


if __name__ == "__main__":
    print("Testing identity calibration behavior...\n")
    
    test_calibration_matrix_defaults()
    test_identity_transform_preserves_values()
    test_calibrated_properties_without_calibration()
    test_with_calibration_vs_without()
    test_non_identity_calibration_affects_values()
    
    print("\n[PASS] All identity calibration tests passed!")
    print("\nThe app now works seamlessly with or without calibration data.")
    print("When no calibration is present, calibrated properties use identity")
    print("transformation, effectively passing through raw values unchanged.")
