#!/usr/bin/env python3
"""
Runner for all minimal pipeline tests.
Combines configuration, pipeline, and database tests.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import test functions directly
def import_test_functions():
    """Import test functions dynamically."""
    import importlib.util
    
    # Import config tests
    config_spec = importlib.util.spec_from_file_location(
        "test_config", Path(__file__).parent / "test_config.py"
    )
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    
    # Import minimal tests
    minimal_spec = importlib.util.spec_from_file_location(
        "test_minimal", Path(__file__).parent / "test_minimal.py"
    )
    minimal_module = importlib.util.module_from_spec(minimal_spec)
    minimal_spec.loader.exec_module(minimal_module)
    
    # Import qdrant tests
    qdrant_spec = importlib.util.spec_from_file_location(
        "test_qdrant", Path(__file__).parent / "test_qdrant.py"
    )
    qdrant_module = importlib.util.module_from_spec(qdrant_spec)
    qdrant_spec.loader.exec_module(qdrant_module)
    
    return (
        config_module.run_config_validation_tests,
        minimal_module.run_minimal_pipeline_tests,
        qdrant_module.run_qdrant_tests,
        qdrant_module.wait_for_qdrant
    )


def run_all_pipeline_tests() -> bool:
    """Run all minimal pipeline tests."""
    print("🧪 Complete Minimal Pipeline Test Suite")
    print("=" * 50)
    print("🎯 Using only Google Gemini embeddings (no local models)")
    print("=" * 50)
    
    # Import test functions
    (run_config_validation_tests, 
     run_minimal_pipeline_tests, 
     run_qdrant_tests, 
     wait_for_qdrant) = import_test_functions()
    
    # Test categories
    test_suites = [
        ("Configuration Validation", run_config_validation_tests),
        ("Minimal Pipeline Tests", run_minimal_pipeline_tests),
    ]
    
    # Add Qdrant tests if enabled
    if os.getenv('CI_RUN_DB_TESTS'):
        print("🗄️ Database tests enabled")
        if wait_for_qdrant(60):
            test_suites.append(("Qdrant Connectivity", run_qdrant_tests))
        else:
            print("⚠️ Qdrant not available, skipping database tests")
    else:
        print("⚠️ Database tests disabled (CI_RUN_DB_TESTS not set)")
    
    # Run test suites
    passed_suites = 0
    failed_suites = []
    
    for suite_name, test_func in test_suites:
        print(f"\n🚀 Running {suite_name}")
        print("=" * 50)
        
        if test_func():
            passed_suites += 1
            print(f"✅ {suite_name} PASSED")
        else:
            failed_suites.append(suite_name)
            print(f"❌ {suite_name} FAILED")
    
    # Final summary
    total_suites = len(test_suites)
    print("\n" + "=" * 50)
    print("📊 FINAL PIPELINE TEST RESULTS")
    print("=" * 50)
    
    if passed_suites == total_suites:
        print("🎉 ALL PIPELINE TESTS PASSED!")
        print("✅ Pipeline is ready for production")
        print("✅ Google Gemini embeddings properly configured")
        print("✅ No local models required")
        return True
    else:
        print(f"❌ {total_suites - passed_suites} of {total_suites} test suites failed")
        print("Failed test suites:")
        for suite in failed_suites:
            print(f"  • {suite}")
        print("\n🔧 Please fix the issues above")
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run minimal pipeline tests")
    parser.add_argument("--with-db", action="store_true", 
                       help="Enable database tests")
    parser.add_argument("--wait", action="store_true",
                       help="Wait for Qdrant if database tests enabled")
    
    args = parser.parse_args()
    
    if args.with_db:
        os.environ['CI_RUN_DB_TESTS'] = '1'
    
    if args.wait and os.getenv('CI_RUN_DB_TESTS'):
        print("⏳ Waiting for Qdrant...")
        _, _, _, wait_for_qdrant = import_test_functions()
        wait_for_qdrant(60)
    
    success = run_all_pipeline_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
