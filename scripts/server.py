#!/usr/bin/env python3
"""
Startup script for Voice Authentication System API server
Provides easy server startup with configuration validation
"""

import sys
import subprocess
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings, ensure_directories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import torch
        import torchaudio
        import librosa
        import speechbrain
        logger.info("✓ All dependencies installed")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        logger.error("Please run: pip install -r requirements.txt")
        return False


def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        if result.returncode == 0:
            logger.info("✓ FFmpeg is installed")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    logger.warning("✗ FFmpeg not found. Audio processing may fail.")
    logger.warning("Install FFmpeg: https://ffmpeg.org/download.html")
    return False


def main():
    """Main startup function"""
    print("=" * 60)
    print(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    print("=" * 60)
    print()
    
    # Check dependencies
    logger.info("Checking system requirements...")
    if not check_dependencies():
        sys.exit(1)
    
    check_ffmpeg()
    
    # Ensure directories exist
    logger.info("Creating necessary directories...")
    ensure_directories()
    
    # Print configuration
    print()
    print("Configuration:")
    print(f"  Host: {settings.API_HOST}")
    print(f"  Port: {settings.API_PORT}")
    print(f"  Debug: {settings.DEBUG}")
    print(f"  Database: {settings.DATABASE_URL}")
    print(f"  Required Samples: {settings.REQUIRED_SAMPLES}")
    print(f"  Similarity Threshold: {settings.SIMILARITY_THRESHOLD}")
    print()
    
    # Start server
    logger.info("Starting API server...")
    print("=" * 60)
    print(f"API Documentation: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print(f"Health Check: http://{settings.API_HOST}:{settings.API_PORT}/health")
    print("=" * 60)
    print()
    
    try:
        import uvicorn
        uvicorn.run(
            "api.app:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=settings.DEBUG,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()