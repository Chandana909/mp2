"""
ML Reliability Platform - Main Entry Point

Starts the FastAPI application with proper configuration,
logging, and error handling.

Usage:
    python run.py                    # Default: localhost:8000
    python run.py --host 0.0.0.0     # Expose on all interfaces
    python run.py --port 8080        # Custom port
    python run.py --reload           # Development mode
    python run.py --workers 4        # Production with 4 workers
"""

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

from ml_platform.config import load_config, get_config


def setup_logging():
    """Configure logging for the platform."""
    log_level = get_config("serving.log_level") or "info"
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/platform.log"),
        ],
    )
    
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)


def ensure_directories():
    """Create required storage directories if they don't exist."""
    from ml_platform.config import get_storage_paths
    
    try:
        models_path, metadata_path, audit_path = get_storage_paths()
        
        for path in [models_path, metadata_path, audit_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        logging.info(f"✅ Storage directories ready")
        logging.info(f"   Models: {models_path}")
        logging.info(f"   Metadata: {metadata_path}")
        logging.info(f"   Audit: {audit_path}")
        
    except Exception as e:
        logging.error(f"❌ Failed to create directories: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ML Reliability Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind (default: from config or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (default: from config or 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        load_config()
    except Exception as e:
        print(f"❌ ERROR: Failed to load configuration: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Ensure storage directories
    try:
        ensure_directories()
    except Exception as e:
        logger.error(f"❌ Failed to setup directories: {e}")
        sys.exit(1)
    
    # Get server config
    host = args.host or get_config("serving.host") or "0.0.0.0"
    port = args.port or get_config("serving.port") or 8000
    workers = args.workers or get_config("serving.workers") or 1
    log_level = get_config("serving.log_level") or "info"
    
    # Log startup
    logger.info("=" * 60)
    logger.info("🚀 ML RELIABILITY PLATFORM STARTING")
    logger.info("=" * 60)
    logger.info(f"Host:     {host}")
    logger.info(f"Port:     {port}")
    logger.info(f"Workers:  {workers if not args.reload else '1 (reload)'}")
    logger.info(f"Reload:   {args.reload}")
    logger.info("=" * 60)
    logger.info(f"📡 API Docs: http://{host}:{port}/docs")
    logger.info(f"❤️  Health:  http://{host}:{port}/health")
    logger.info("=" * 60)
    
    # Start server
    try:
        uvicorn.run(
            "serving.api:app",
            host=host,
            port=port,
            reload=args.reload,
            workers=1 if args.reload else workers,
            log_level=log_level,
        )
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down...")
    except Exception as e:
        logger.error(f"❌ Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
