"""
Tests automated continuous retraining pipeline by invoking the Retrainer
class manually, without using the HTTP API.
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_platform.config import load_config
from lifecycle.trainer import AutoRetrainer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python train_new_version.py <model_name>")
        sys.exit(1)
        
    model_name = sys.argv[1]
    
    logger.info("=" * 60)
    logger.info(f"🚀 Triggering background retraining for {model_name}")
    logger.info("=" * 60)
    
    load_config()
    retrainer = AutoRetrainer()
    
    try:
        result = retrainer.run_retraining_pipeline(model_name=model_name)
        logger.info(f"🎯 Outcome: {result['model_name']} {result['new_version']} registered.")
        logger.info(f"   Accuracy: {result['accuracy']*100:.1f}% (+{result['improvement']*100:.1f}%)")
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")

if __name__ == "__main__":
    main()
