"""
Core package for Voice Authentication System
Exports the main engine and service
"""

from core.voice_auth import VoiceAuthEngine, voice_auth_engine
from core.service import VoiceAuthService, voice_auth_service

__all__ = [
    "VoiceAuthEngine",
    "voice_auth_engine",
    "VoiceAuthService",
    "voice_auth_service"
]