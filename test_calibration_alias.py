#!/usr/bin/env python3
"""
Test script to verify --calibration alias works correctly for --calibrate
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from facemesh_app.main import parse_args, normalize_runtime_args

def test_calibrate_flag():
    """Test --calibrate flag"""
    sys.argv = ['test', '--calibrate']
    args = parse_args()
    normalize_runtime_args(args)
    
    assert args.calibrate == True, "calibrate should be True"
    assert args.calibration == False, "calibration should be False"
    assert args.should_calibrate == True, "should_calibrate should be True"
    print("✓ Test 1 passed: --calibrate flag works correctly")

def test_calibration_flag():
    """Test --calibration flag (alias)"""
    sys.argv = ['test', '--calibration']
    args = parse_args()
    normalize_runtime_args(args)
    
    assert args.calibrate == False, "calibrate should be False"
    assert args.calibration == True, "calibration should be True"
    assert args.should_calibrate == True, "should_calibrate should be True"
    print("✓ Test 2 passed: --calibration flag works correctly")

def test_both_flags():
    """Test both flags together"""
    sys.argv = ['test', '--calibrate', '--calibration']
    args = parse_args()
    normalize_runtime_args(args)
    
    assert args.calibrate == True, "calibrate should be True"
    assert args.calibration == True, "calibration should be True"
    assert args.should_calibrate == True, "should_calibrate should be True"
    print("✓ Test 3 passed: both flags together work correctly")

def test_no_flags():
    """Test no calibration flags"""
    sys.argv = ['test']
    args = parse_args()
    normalize_runtime_args(args)
    
    assert args.calibrate == False, "calibrate should be False"
    assert args.calibration == False, "calibration should be False"
    assert args.should_calibrate == False, "should_calibrate should be False"
    print("✓ Test 4 passed: no calibration flags work correctly")

def test_with_profile():
    """Test calibration with profile name"""
    sys.argv = ['test', '--calibration', '--calibration-profile', 'lorenzo']
    args = parse_args()
    normalize_runtime_args(args)
    
    assert args.calibration == True, "calibration should be True"
    assert args.calibration_profile == 'lorenzo', "calibration_profile should be 'lorenzo'"
    assert args.should_calibrate == True, "should_calibrate should be True"
    print("✓ Test 5 passed: --calibration with --calibration-profile works correctly")

def test_calibrate_with_profile():
    """Test calibrate with profile name (the correct way)"""
    sys.argv = ['test', '--calibrate', '--calibration-profile', 'lorenzo']
    args = parse_args()
    normalize_runtime_args(args)
    
    assert args.calibrate == True, "calibrate should be True"
    assert args.calibration_profile == 'lorenzo', "calibration_profile should be 'lorenzo'"
    assert args.should_calibrate == True, "should_calibrate should be True"
    print("✓ Test 6 passed: --calibrate with --calibration-profile works correctly")

if __name__ == '__main__':
    print("Testing calibration argument aliases...")
    print()
    
    try:
        test_calibrate_flag()
        test_calibration_flag()
        test_both_flags()
        test_no_flags()
        test_with_profile()
        test_calibrate_with_profile()
        
        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()
        print("Users can now use:")
        print("  --calibrate                    (primary flag)")
        print("  --calibration                  (alias, for convenience)")
        print("  --calibration-profile <name>   (profile name)")
        print()
        print("Example valid commands:")
        print("  --calibrate --calibration-profile lorenzo")
        print("  --calibration --calibration-profile lorenzo")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
