"""
Database package for Voice Authentication System
Exports models and manager for easy importing

"""

from database.models import Base, User, VoiceSample, AuthenticationLog
from database.manager import DatabaseManager, db_manager

__all__ = [
    "Base",
    "User",
    "VoiceSample", 
    "AuthenticationLog",
    "DatabaseManager",
    "db_manager"
]