#!/usr/bin/env python3
"""
Local End-to-End Test Setup

Script to test the end-to-end pipeline locally before pushing to GitHub.
Requires GOOGLE_API_KEY environment variable and Qdrant running locally.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable not set")
        print("💡 Set it with: export GOOGLE_API_KEY=your_api_key")
        return False
    else:
        print("✅ GOOGLE_API_KEY is set")
    
    # Check Qdrant
    try:
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is running on localhost:6333")
        else:
            print(f"❌ Qdrant responded with status {response.status_code}")
            return False
    except requests.ConnectionError:
        print("❌ Qdrant is not running on localhost:6333")
        print("💡 Start it with: docker run -p 6333:6333 qdrant/qdrant")
        return False
    except requests.Timeout:
        print("❌ Qdrant connection timeout")
        return False
    
    # Check Python packages
    required_packages = [
        "langchain_google_genai",
        "qdrant_client", 
        "pytest",
        "requests"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"💡 Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def run_test_levels():
    """Run tests in progressive levels."""
    print("\n🧪 Running Progressive Test Levels")
    print("=" * 50)
    
    # Level 1: Minimal tests (no external dependencies)
    print("\n📋 Level 1: Minimal Tests (No External Dependencies)")
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/pipeline/test_minimal_pipeline.py",
        "tests/pipeline/test_components.py",
        "-v", "--tb=short"
    ])
    
    if result.returncode != 0:
        print("❌ Level 1 tests failed - fix basic issues first")
        return False
    else:
        print("✅ Level 1 tests passed")
    
    # Level 2: Integration tests (Qdrant only)
    print("\n📋 Level 2: Integration Tests (Qdrant Only)")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/pipeline/test_qdrant_connectivity.py", 
        "-v", "--tb=short"
    ])
    
    if result.returncode != 0:
        print("❌ Level 2 tests failed - check Qdrant setup")
        return False
    else:
        print("✅ Level 2 tests passed")
    
    # Level 3: End-to-end tests (API + Qdrant)
    print("\n📋 Level 3: End-to-End Tests (API + Qdrant)")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/pipeline/test_end_to_end.py",
        "-v", "--tb=short", "-m", "requires_api"
    ])
    
    if result.returncode != 0:
        print("❌ Level 3 tests failed - check API key and Qdrant data")
        return False
    else:
        print("✅ Level 3 tests passed")
    
    return True

def main():
    """Main test runner."""
    print("🚀 Local End-to-End Pipeline Test Setup")
    print("=" * 50)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    # Run progressive tests
    if run_test_levels():
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your pipeline is ready for GitHub CI/CD")
        print("\n💡 Next steps:")
        print("1. Commit your changes")
        print("2. Push to GitHub")
        print("3. Check GitHub Actions for automated testing")
    else:
        print("\n❌ Some tests failed. Please fix the issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()
