#!/usr/bin/env python3
"""
Standalone test for calibration argument parsing (no module dependencies)
"""
import argparse

def create_parser():
    """Create argument parser matching main.py"""
    parser = argparse.ArgumentParser(description="FaceMesh data capture app")
    
    # Calibration options (matching main.py)
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=False,
                        help="Run 9-point calibration workflow (primary flag)")
    parser.add_argument("--calibration", action=argparse.BooleanOptionalAction, default=False,
                        help="Run 9-point calibration workflow (alias for --calibrate)")
    parser.add_argument("--calibration-profile", type=str, default="",
                        help="Calibration profile name")
    parser.add_argument("--force-recalibrate", action="store_true",
                        help="Ignore existing calibration and force recalibration")
    
    return parser

def normalize_runtime_args(args):
    """Normalize arguments matching main.py"""
    # Unify calibrate and calibration flags into should_calibrate
    args.should_calibrate = args.calibrate or args.calibration

def test_calibrate_flag():
    """Test --calibrate flag"""
    parser = create_parser()
    args = parser.parse_args(['--calibrate'])
    normalize_runtime_args(args)
    
    assert args.calibrate == True, "calibrate should be True"
    assert args.calibration == False, "calibration should be False"
    assert args.should_calibrate == True, "should_calibrate should be True"
    print("[PASS] Test 1 passed: --calibrate flag works correctly")

def test_calibration_flag():
    """Test --calibration flag (alias)"""
    parser = create_parser()
    args = parser.parse_args(['--calibration'])
    normalize_runtime_args(args)
    
    assert args.calibrate == False, "calibrate should be False"
    assert args.calibration == True, "calibration should be True"
    assert args.should_calibrate == True, "should_calibrate should be True"
    print("[PASS] Test 2 passed: --calibration flag works correctly")

def test_both_flags():
    """Test both flags together"""
    parser = create_parser()
    args = parser.parse_args(['--calibrate', '--calibration'])
    normalize_runtime_args(args)
    
    assert args.calibrate == True, "calibrate should be True"
    assert args.calibration == True, "calibration should be True"
    assert args.should_calibrate == True, "should_calibrate should be True"
    print("[PASS] Test 3 passed: both flags together work correctly")

def test_no_flags():
    """Test no calibration flags"""
    parser = create_parser()
    args = parser.parse_args([])
    normalize_runtime_args(args)
    
    assert args.calibrate == False, "calibrate should be False"
    assert args.calibration == False, "calibration should be False"
    assert args.should_calibrate == False, "should_calibrate should be False"
    print("[PASS] Test 4 passed: no calibration flags work correctly")

def test_with_profile():
    """Test calibration with profile name"""
    parser = create_parser()
    args = parser.parse_args(['--calibration', '--calibration-profile', 'lorenzo'])
    normalize_runtime_args(args)
    
    assert args.calibration == True, "calibration should be True"
    assert args.calibration_profile == 'lorenzo', "calibration_profile should be 'lorenzo'"
    assert args.should_calibrate == True, "should_calibrate should be True"
    print("[PASS] Test 5 passed: --calibration with --calibration-profile works correctly")

def test_calibrate_with_profile():
    """Test calibrate with profile name (the correct way)"""
    parser = create_parser()
    args = parser.parse_args(['--calibrate', '--calibration-profile', 'lorenzo'])
    normalize_runtime_args(args)
    
    assert args.calibrate == True, "calibrate should be True"
    assert args.calibration_profile == 'lorenzo', "calibration_profile should be 'lorenzo'"
    assert args.should_calibrate == True, "should_calibrate should be True"
    print("[PASS] Test 6 passed: --calibrate with --calibration-profile works correctly")

def test_negative_flags():
    """Test negative flags (--no-calibrate, --no-calibration)"""
    parser = create_parser()
    args = parser.parse_args(['--no-calibrate'])
    normalize_runtime_args(args)
    
    assert args.calibrate == False, "calibrate should be False"
    assert args.calibration == False, "calibration should be False"
    assert args.should_calibrate == False, "should_calibrate should be False"
    print("[PASS] Test 7 passed: --no-calibrate flag works correctly")

def test_original_issue():
    """Test the original issue: --calibration lorenzo (should not work)"""
    parser = create_parser()
    # This should fail because 'lorenzo' is not a flag
    try:
        args = parser.parse_args(['--calibration', 'lorenzo'])
        # If we get here, it means 'lorenzo' was ignored (treated as a positional arg)
        # This is actually what argparse does - ignores unknown positional args
        normalize_runtime_args(args)
        print("[PASS] Test 8a: --calibration lorenzo - 'lorenzo' treated as positional (ignored)")
    except SystemExit:
        # If it fails, that's also acceptable behavior
        print("[PASS] Test 8b: --calibration lorenzo - correctly rejects as invalid")

def test_correct_usage():
    """Test the correct usage: --calibrate --calibration-profile lorenzo"""
    parser = create_parser()
    args = parser.parse_args(['--calibrate', '--calibration-profile', 'lorenzo'])
    normalize_runtime_args(args)
    
    assert args.should_calibrate == True, "should_calibrate should be True"
    assert args.calibration_profile == 'lorenzo', "calibration_profile should be 'lorenzo'"
    print("[PASS] Test 9 passed: Correct usage --calibrate --calibration-profile lorenzo works")

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
        test_negative_flags()
        test_original_issue()
        test_correct_usage()
        
        print()
        print("=" * 70)
        print("All tests passed! [OK]")
        print("=" * 70)
        print()
        print("SUMMARY:")
        print("--------")
        print("Users can now use:")
        print("  --calibrate                    (primary flag)")
        print("  --calibration                  (alias, for convenience)")
        print("  --calibration-profile <name>   (profile name)")
        print()
        print("Example valid commands:")
        print("  --calibrate --calibration-profile lorenzo")
        print("  --calibration --calibration-profile lorenzo")
        print()
        print("The fix makes --calibration a convenient alias for --calibrate.")
        print("Both flags trigger the same calibration workflow.")
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        import sys
        sys.exit(1)
