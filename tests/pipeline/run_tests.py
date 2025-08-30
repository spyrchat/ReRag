"""
Comprehensive Test Runner for Minimal Pipeline Tests

Runs all minimal pipeline tests with proper error handling and reporting.
Designed to be CI-friendly and avoid local model dependencies.
"""

import pytest
import sys
import os
from pathlib import Path
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """
    Run all minimal pipeline tests with comprehensive reporting.
    """
    print("🧪 Comprehensive Pipeline Test Suite")
    print("=" * 60)
    print("🎯 Testing core pipeline functionality without local models")
    
    # Check API key availability
    has_api_key = bool(os.getenv("GOOGLE_API_KEY"))
    has_qdrant = check_qdrant_availability()
    
    print(f"🔑 Google API Key: {'✅ Available' if has_api_key else '❌ Not set'}")
    print(f"🗄️  Qdrant Service: {'✅ Running' if has_qdrant else '❌ Not available'}")
    print("=" * 60)
    
    # Test files to run
    test_files = [
        ("test_minimal_pipeline.py", "Minimal Pipeline Tests", False, False),
        ("test_components.py", "Component Integration Tests", False, False), 
        ("test_qdrant_connectivity.py", "Qdrant Connectivity Tests", True, False)
    ]
    
    # Add end-to-end tests if API key is available
    if has_api_key and has_qdrant:
        test_files.append(("test_end_to_end.py", "End-to-End Pipeline Tests", True, True))
        print("� Full test suite - including end-to-end tests")
    elif has_api_key:
        print("⚠️  API key available but Qdrant not running - skipping end-to-end tests")
    elif has_qdrant:
        print("⚠️  Qdrant running but no API key - skipping end-to-end tests")
    else:
        print("⚠️  Running minimal test suite only")
    
    results = {}
    
    for test_file, description, requires_qdrant, requires_api in test_files:
        print(f"\n🚀 Running {description}")
        print("-" * 40)
        
        # Skip if requirements not met
        if requires_qdrant and not has_qdrant:
            print(f"⏭️  Skipping {test_file} - Qdrant not available")
            results[test_file] = "SKIPPED_NO_QDRANT"
            continue
            
        if requires_api and not has_api_key:
            print(f"⏭️  Skipping {test_file} - API key not available")
            results[test_file] = "SKIPPED_NO_API"
            continue
        
        test_path = Path(__file__).parent / test_file
        
        if not test_path.exists():
            print(f"❌ Test file not found: {test_file}")
            results[test_file] = "FILE_NOT_FOUND"
            continue
        
        try:
            # Build pytest command
            pytest_args = [
                str(test_path),
                "-v",
                "--tb=short"
            ]
            
            # Add markers for end-to-end tests
            if requires_api:
                pytest_args.extend(["-m", "requires_api"])
            
            # Run pytest on specific file
            exit_code = pytest.main(pytest_args)
            
            if exit_code == 0:
                print(f"✅ {description} passed")
                results[test_file] = "PASSED"
            else:
                print(f"❌ {description} failed")
                results[test_file] = "FAILED"
                
        except Exception as e:
            print(f"💥 Error running {test_file}: {e}")
            results[test_file] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result == "PASSED")
    skipped = sum(1 for result in results.values() if result.startswith("SKIPPED"))
    failed = sum(1 for result in results.values() if result == "FAILED" or result.startswith("ERROR"))
    total = len(results)
    
    for test_file, result in results.items():
        if result == "PASSED":
            status_emoji = "✅"
        elif result.startswith("SKIPPED"):
            status_emoji = "⏭️ "
        else:
            status_emoji = "❌"
        print(f"{status_emoji} {test_file}: {result}")
    
    print(f"\n🎯 Results: {passed} passed, {skipped} skipped, {failed} failed ({total} total)")
    
    if failed == 0:
        if passed > 0:
            print("🎉 ALL AVAILABLE TESTS PASSED!")
            if has_api_key and has_qdrant:
                print("✅ Complete pipeline validation successful")
            else:
                print("✅ Available components validated successfully")
                if not has_api_key:
                    print("💡 Set GOOGLE_API_KEY for end-to-end tests")
                if not has_qdrant:
                    print("💡 Start Qdrant for database tests")
        else:
            print("⚠️  No tests were executed")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


def check_qdrant_availability():
    """Check if Qdrant is available."""
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=3)
        return response.status_code == 200
    except:
        return False


def check_dependencies():
    """
    Check if required dependencies are available.
    """
    print("🔍 Checking dependencies...")
    
    required_modules = [
        "pytest",
        "yaml", 
        "requests",
        "langchain_core"
    ]
    
    missing = []
    
    for module in required_modules:
        try:
            spec = importlib.util.find_spec(module)
            if spec is None:
                missing.append(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("💡 Install with: pip install " + " ".join(missing))
        return False
    else:
        print("✅ All dependencies available")
        return True


def check_environment():
    """
    Check environment setup for tests.
    """
    print("🌍 Checking environment...")
    
    # Check if we're in the right directory
    if not Path("config.yml").exists():
        print("❌ config.yml not found. Make sure you're in the project root directory.")
        return False
    
    # Check CI Google config exists
    ci_config = Path("pipelines/configs/retrieval/ci_google_gemini.yml")
    if not ci_config.exists():
        print(f"❌ CI Google config not found: {ci_config}")
        return False
    
    print("✅ Environment ready")
    return True


def main():
    """
    Main test runner entry point.
    """
    print("🎬 Starting Minimal Pipeline Test Suite")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python: {sys.version}")
    
    # Pre-flight checks
    if not check_dependencies():
        sys.exit(1)
    
    if not check_environment():
        sys.exit(1)
    
    # Run tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
