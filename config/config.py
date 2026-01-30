"""
Configuration settings for Voice Authentication System
Centralized configuration management for easy deployment
"""

from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    All values can be overridden via .env file or environment variables
    """
    
    # Application Info
    APP_NAME: str = "Voice Authentication System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Database Settings
    DATABASE_URL: str = "sqlite+aiosqlite:///./voice_auth.db"
    
    # Voice Authentication Settings
    SAMPLE_RATE: int = 16000  # Hz - Standard for speech processing
    MIN_AUDIO_LENGTH: float = 1.0  # seconds - Minimum audio length for registration
    MAX_AUDIO_LENGTH: float = 10.0  # seconds - Maximum audio length
    REQUIRED_SAMPLES: int = 3  # Number of samples required for registration
    SIMILARITY_THRESHOLD: float = 0.75  # Cosine similarity threshold (0-1, higher = stricter)
    
    # Audio Processing
    NOISE_REDUCTION: bool = True
    NORMALIZE_AUDIO: bool = True
    
    # File Storage
    AUDIO_UPLOAD_DIR: Path = Path("./audio_uploads")
    VOICEPRINT_DIR: Path = Path("./voiceprints")
    
    # Model Settings
    SPEAKER_MODEL: str = "speechbrain/spkrec-ecapa-voxceleb"
    
    # Security
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_AUDIO_FORMATS: list = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def ensure_directories():
    """
    Create necessary directories if they don't exist
    Called during application startup
    """
    settings.AUDIO_UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
    settings.VOICEPRINT_DIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    # Print current configuration for verification
    print(f"Configuration for {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Database: {settings.DATABASE_URL}")
    print(f"Sample Rate: {settings.SAMPLE_RATE} Hz")
    print(f"Similarity Threshold: {settings.SIMILARITY_THRESHOLD}")