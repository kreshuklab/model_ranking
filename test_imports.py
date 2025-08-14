#!/usr/bin/env python3
"""Test script to verify the new import structure works correctly."""


def test_transferability_metrics():
    """Test that transferability metrics can be imported directly from model_ranking."""
    try:
        # Test importing transferability metrics directly from model_ranking
        from model_ranking import (
            h_score,
            bhattacharyya_coefficient,
            log_expected_empirical_prediction,
            log_maximum_evidence,
            NCTI_Score,
        )

        print("✓ Transferability metrics imported successfully from model_ranking")
        return True
    except ImportError as e:
        print(f"✗ Failed to import transferability metrics: {e}")
        return False


def test_yaml_generators():
    """Test that YAML generator functions can be imported directly."""
    try:
        from model_ranking import generate_run_yamls

        print("✓ YAML generators imported successfully from model_ranking")
        return True
    except ImportError as e:
        print(f"✗ Failed to import YAML generators: {e}")
        return False


def test_subdirectory_import():
    """Test that the old subdirectory import structure still works."""
    try:
        from model_ranking.transferability_metrics import h_score as h_score_sub

        print("✓ Subdirectory imports still work")
        return True
    except ImportError as e:
        print(f"✗ Subdirectory imports failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing new import structure...")
    print("=" * 50)

    tests = [
        test_transferability_metrics,
        test_yaml_generators,
        test_subdirectory_import,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("🎉 All tests passed! The restructuring was successful.")
    else:
        print("❌ Some tests failed. Please check the import structure.")
