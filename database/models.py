"""
Database Models for Voice Authentication System
Defines the schema for users, voiceprints, and authentication logs
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid


Base = declarative_base()


class User(Base):
    """
    User model - stores basic user information
    Each user can have multiple voice samples for better accuracy
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationship to voice samples
    voice_samples = relationship("VoiceSample", back_populates="user", cascade="all, delete-orphan")
    auth_logs = relationship("AuthenticationLog", back_populates="user", cascade="all, delete-orphan")
    
    # Registration status
    is_registered = Column(Boolean, default=False)  # True when enough samples collected
    required_samples = Column(Integer, default=3)
    
    def __repr__(self):
        return f"<User(username='{self.username}', uuid='{self.uuid}', registered={self.is_registered})>"


class VoiceSample(Base):
    """
    VoiceSample model - stores voice embeddings and metadata
    Multiple samples per user improve authentication accuracy
    """
    __tablename__ = "voice_samples"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Audio metadata
    passphrase = Column(String(500), nullable=False)  # What the user said
    audio_duration = Column(Float, nullable=False)  # Duration in seconds
    sample_rate = Column(Integer, nullable=False)  # Sample rate in Hz
    
    # Voiceprint storage (stored as numpy array path)
    embedding_path = Column(String(500), nullable=False)  # Path to .npy file
    embedding_dimension = Column(Integer, nullable=False)  # Typically 192 for ECAPA-TDNN
    
    # Quality metrics
    signal_to_noise_ratio = Column(Float, nullable=True)  # SNR in dB
    confidence_score = Column(Float, nullable=True)  # Model confidence (0-1)
    
    # Timestamps
    recorded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="voice_samples")
    
    def __repr__(self):
        return f"<VoiceSample(user_id={self.user_id}, passphrase='{self.passphrase[:20]}...', duration={self.audio_duration}s)>"


class AuthenticationLog(Base):
    """
    AuthenticationLog model - tracks all authentication attempts
    Useful for security monitoring and analytics
    """
    __tablename__ = "authentication_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Null if failed authentication
    
    # Authentication details
    username_attempted = Column(String(100), nullable=False)
    passphrase = Column(String(500), nullable=False)
    success = Column(Boolean, nullable=False)
    similarity_score = Column(Float, nullable=True)  # Cosine similarity (0-1)
    
    # Additional metadata
    ip_address = Column(String(45), nullable=True)  # Support IPv6
    user_agent = Column(Text, nullable=True)
    audio_duration = Column(Float, nullable=True)
    
    # Failure reason if applicable
    failure_reason = Column(String(500), nullable=True)
    
    # Timestamp
    attempted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="auth_logs")
    
    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"<AuthLog({status}, user='{self.username_attempted}', score={self.similarity_score})>"


# Database utility functions
def get_table_names():
    """Return list of all table names in the database"""
    return [table.name for table in Base.metadata.sorted_tables]


def get_model_by_tablename(tablename: str):
    """Get SQLAlchemy model class by table name"""
    for mapper in Base.registry.mappers:
        model = mapper.class_
        if model.__tablename__ == tablename:
            return model
    return None