#!/usr/bin/env python3
"""
Setup Verification Script
=========================

Verifies that all dependencies are correctly installed and
the environment is ready to run the Numerai project.

Usage:
    python verify_setup.py

Author: ML Student
Date: 2024
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_imports():
    """Check required libraries."""
    print("\nChecking required libraries...")

    required_libs = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'lightgbm': 'lightgbm',
        'numerapi': 'numerapi',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy',
        'tqdm': 'tqdm'
    }

    all_good = True
    for import_name, package_name in required_libs.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name:20s} (version {version})")
        except ImportError:
            print(f"✗ {package_name:20s} (not installed)")
            all_good = False

    return all_good


def check_files():
    """Check required project files."""
    print("\nChecking project files...")

    # Get the project root (parent of scripts/)
    project_root = Path(__file__).parent.parent

    required_files = [
        'notebooks/numerai_project.ipynb',
        'numerai_utils.py',
        'scripts/quick_start.py',
        'requirements.txt',
        'README.md',
        'docs/PROJECT_SUMMARY.md'
    ]

    all_good = True
    for filename in required_files:
        path = project_root / filename
        if path.exists():
            size = path.stat().st_size
            print(f"✓ {filename:35s} ({size:,} bytes)")
        else:
            print(f"✗ {filename:35s} (missing)")
            all_good = False

    return all_good


def check_jupyter():
    """Check Jupyter installation."""
    print("\nChecking Jupyter...")
    try:
        import notebook
        version = getattr(notebook, '__version__', 'unknown')
        print(f"✓ Jupyter Notebook installed (version {version})")
        return True
    except ImportError:
        print("✗ Jupyter Notebook not installed")
        print("  Install with: pip install jupyter notebook")
        return False


def check_disk_space():
    """Check available disk space."""
    print("\nChecking disk space...")
    try:
        import shutil
        stats = shutil.disk_usage('.')
        free_gb = stats.free / (1024**3)

        if free_gb >= 5:
            print(f"✓ Available disk space: {free_gb:.1f} GB (sufficient)")
            return True
        else:
            print(f"⚠ Available disk space: {free_gb:.1f} GB (may be insufficient)")
            print("  Recommended: 5+ GB for data download and processing")
            return True  # Warning, not error
    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")
        return True


def test_utility_module():
    """Test numerai_utils module."""
    print("\nTesting utility module...")
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        from numerai_utils import (
            NumeraiMetrics,
            NumeraiDataLoader,
            NumeraiEnsemble,
            SubmissionFormatter
        )
        print("✓ All utility classes imported successfully")
        return True
    except Exception as e:
        print(f"✗ Error importing utility module: {e}")
        return False


def print_summary(results):
    """Print summary of checks."""
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {check}")

    print("=" * 70)

    if all_passed:
        print("\n🎉 SUCCESS! Your environment is ready.")
        print("\nNext steps:")
        print("  1. Run: jupyter notebook notebooks/numerai_project.ipynb")
        print("  2. Or run: python scripts/quick_start.py")
        print("\n")
        return 0
    else:
        print("\n❌ FAILED! Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Upgrade Python to 3.8+")
        print("  - Check file permissions")
        print("\n")
        return 1


def main():
    """Main verification function."""
    print("=" * 70)
    print("NUMERAI PROJECT - SETUP VERIFICATION")
    print("=" * 70)
    print()

    results = {
        'Python Version': check_python_version(),
        'Required Libraries': check_imports(),
        'Project Files': check_files(),
        'Jupyter Notebook': check_jupyter(),
        'Disk Space': check_disk_space(),
        'Utility Module': test_utility_module()
    }

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
