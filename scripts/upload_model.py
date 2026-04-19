"""
CLI utility to formally upload a model artifact to the ML Reliability Platform.
Posts the model binary + metadata to /models/upload
"""
import argparse
import json
import logging
from pathlib import Path
import httpx
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Upload a model artifact (joblib/pickle) to the platform.")
    parser.add_argument("--file", required=True, help="Path to the model file (.joblib)")
    parser.add_argument("--name", required=True, help="Logical model name (e.g., fraud_detector)")
    parser.add_argument("--version", required=True, help="Version string (e.g., v5)")
    parser.add_argument("--type", default="classification", help="Model type")
    parser.add_argument("--features", required=True, help="Comma separated features: age,loc,amount")
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="Platform host")
    
    args = parser.parse_args()
    file_path = Path(args.file)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
        
    features_list = [f.strip() for f in args.features.split(",") if f.strip()]
    metadata = {
        "model_name": args.name,
        "version": args.version,
        "model_type": args.type,
        "features": json.dumps(features_list),
        "target": "target",
        "accuracy": "0.85" # default placeholder
    }
    
    logger.info(f"📤 Uploading {args.name} {args.version} [{file_path.name}]...")
    try:
        with open(file_path, "rb") as f:
            files = {"model_file": (file_path.name, f, "application/octet-stream")}
            r = httpx.post(f"{args.host}/models/upload", data=metadata, files=files, timeout=30.0)
            
        if r.status_code == 200:
            data = r.json()
            logger.info("✅ Upload successful!")
            logger.info(f"ID: {data.get('model_id')}")
            logger.info(f"Status: {data.get('status_assigned')}")
        else:
            logger.error(f"❌ Upload failed ({r.status_code}): {r.text}")
    except Exception as e:
        logger.error(f"❌ Connection error: {e}")

if __name__ == "__main__":
    main()
