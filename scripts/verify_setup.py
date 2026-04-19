"""
Setup Verification Script

Checks all components are properly configured.

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_imports():
    """Verify imports."""
    logger.info("Checking imports...")
    try:
        from ml_platform.registry import ModelRegistry
        from ml_platform.config import load_config
        from serving.api import app
        from storage.model_store import ModelStore
        from monitoring.drift_detector import DriftDetector
        logger.info("   ✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"   ❌ Import failed: {e}")
        return False


def check_config():
    """Verify config."""
    logger.info("Checking configuration...")
    try:
        from ml_platform.config import load_config, get_config
        load_config()
        name = get_config("platform.name")
        logger.info(f"   ✅ Config loaded: {name}")
        return True
    except Exception as e:
        logger.error(f"   ❌ Config failed: {e}")
        return False


def check_storage():
    """Verify storage."""
    logger.info("Checking storage...")
    try:
        from ml_platform.config import get_storage_paths
        models, meta, audit = get_storage_paths()
        
        for p in [models, meta, audit]:
            Path(p).mkdir(parents=True, exist_ok=True)
            if not p.exists():
                logger.error(f"   ❌ Failed: {p}")
                return False
        
        logger.info("   ✅ Storage ready")
        return True
    except Exception as e:
        logger.error(f"   ❌ Storage failed: {e}")
        return False


def check_deps():
    """Verify dependencies."""
    logger.info("Checking dependencies...")
    required = [
        'fastapi', 'uvicorn', 'pydantic', 'numpy', 'pandas',
        'scikit-learn', 'joblib', 'evidently', 'pytest'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.error(f"   ❌ Missing: {', '.join(missing)}")
        logger.error("      Run: pip install -r requirements.txt")
        return False
    
    logger.info("   ✅ All dependencies installed")
    return True


def check_old_platform():
    """Check for old platform/ folder."""
    logger.info("Checking for old platform/ folder...")
    old = Path("platform")
    if old.exists():
        logger.error("   ❌ Old 'platform/' folder exists!")
        logger.error("      Delete it: rmdir /s /q platform")
        return False
    logger.info("   ✅ No conflicts")
    return True


def main():
    """Run all checks."""
    logger.info("=" * 70)
    logger.info("ML RELIABILITY PLATFORM - SETUP VERIFICATION")
    logger.info("=" * 70)
    logger.info("")
    
    checks = [
        ("Old Platform Folder", check_old_platform),
        ("Dependencies", check_deps),
        ("Imports", check_imports),
        ("Configuration", check_config),
        ("Storage Paths", check_storage),
    ]
    
    results = []
    for name, func in checks:
        result = func()
        results.append((name, result))
        logger.info("")
    
    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    passed = all(r for _, r in results)
    
    logger.info("=" * 70)
    if passed:
        logger.info("✅ ALL CHECKS PASSED")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Run demo:  python scripts/demo_workflow.py")
        logger.info("2. Run tests: pytest tests/ -v")
        logger.info("3. Start API: python run.py")
        return 0
    else:
        logger.error("❌ SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
