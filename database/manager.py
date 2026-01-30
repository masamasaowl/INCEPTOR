"""
Database Manager for Voice Authentication System
Handles all database operations with async support
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy import and_, or_, func
from typing import Optional, List
import logging

from database.models import Base, User, VoiceSample, AuthenticationLog
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Async database manager for all database operations
    Provides high-level methods for user, voice sample, and auth log management
    """
    
    def __init__(self, database_url: str = None):
        """
        Initialize database manager with connection
        
        Args:
            database_url: Database connection string (uses config default if not provided)
        """
        self.database_url = database_url or settings.DATABASE_URL
        self.engine = create_async_engine(
            self.database_url,
            echo=settings.DEBUG,  # Log SQL queries in debug mode
            future=True
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        logger.info(f"Database manager initialized with: {self.database_url}")
    
    async def initialize_database(self):
        """
        Create all tables in the database
        Should be called once during application startup
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    
    async def drop_all_tables(self):
        """
        Drop all tables (USE WITH CAUTION - for testing only)
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.warning("All database tables dropped")
    
    # ==================== USER OPERATIONS ====================
    
    async def create_user(self, username: str, email: Optional[str] = None) -> User:
        """
        Create a new user in the database
        
        Args:
            username: Unique username
            email: Optional email address
            
        Returns:
            User object
            
        Raises:
            ValueError: If username already exists
        """
        async with self.async_session() as session:
            # Check if user already exists
            existing_user = await self.get_user_by_username(username)
            if existing_user:
                raise ValueError(f"User '{username}' already exists")
            
            user = User(username=username, email=email)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            
            logger.info(f"User created: {username} (UUID: {user.uuid})")
            return user
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()
    
    async def get_user_by_uuid(self, user_uuid: str) -> Optional[User]:
        """Get user by UUID"""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.uuid == user_uuid)
            )
            return result.scalar_one_or_none()
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
    
    async def list_all_users(self) -> List[User]:
        """Get all users in the system"""
        async with self.async_session() as session:
            result = await session.execute(select(User))
            return result.scalars().all()
    
    async def update_user_registration_status(self, user_id: int, is_registered: bool):
        """Update user's registration status"""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            if user:
                user.is_registered = is_registered
                await session.commit()
                logger.info(f"User {user.username} registration status updated: {is_registered}")
    
    async def delete_user(self, username: str) -> bool:
        """
        Delete user and all associated data (cascade delete)
        
        Args:
            username: Username to delete
            
        Returns:
            True if deleted, False if user not found
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.username == username)
            )
            user = result.scalar_one_or_none()
            if user:
                await session.delete(user)
                await session.commit()
                logger.info(f"User deleted: {username}")
                return True
            return False
    
    # ==================== VOICE SAMPLE OPERATIONS ====================
    
    async def add_voice_sample(
        self,
        user_id: int,
        passphrase: str,
        audio_duration: float,
        sample_rate: int,
        embedding_path: str,
        embedding_dimension: int,
        snr: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> VoiceSample:
        """
        Add a voice sample for a user
        
        Args:
            user_id: User's database ID
            passphrase: What the user said
            audio_duration: Audio length in seconds
            sample_rate: Sample rate in Hz
            embedding_path: Path to saved embedding (.npy file)
            embedding_dimension: Dimension of the embedding vector
            snr: Signal-to-noise ratio (optional)
            confidence: Model confidence score (optional)
            
        Returns:
            VoiceSample object
        """
        async with self.async_session() as session:
            voice_sample = VoiceSample(
                user_id=user_id,
                passphrase=passphrase,
                audio_duration=audio_duration,
                sample_rate=sample_rate,
                embedding_path=embedding_path,
                embedding_dimension=embedding_dimension,
                signal_to_noise_ratio=snr,
                confidence_score=confidence
            )
            session.add(voice_sample)
            await session.commit()
            await session.refresh(voice_sample)
            
            logger.info(f"Voice sample added for user_id={user_id}, duration={audio_duration}s")
            return voice_sample
    
    async def get_user_voice_samples(self, user_id: int) -> List[VoiceSample]:
        """Get all voice samples for a user"""
        async with self.async_session() as session:
            result = await session.execute(
                select(VoiceSample).where(VoiceSample.user_id == user_id)
            )
            return result.scalars().all()
    
    async def count_user_voice_samples(self, user_id: int) -> int:
        """Count how many voice samples a user has"""
        async with self.async_session() as session:
            result = await session.execute(
                select(func.count(VoiceSample.id)).where(VoiceSample.user_id == user_id)
            )
            return result.scalar()
    
    # ==================== AUTHENTICATION LOG OPERATIONS ====================
    
    async def log_authentication_attempt(
        self,
        username_attempted: str,
        passphrase: str,
        success: bool,
        similarity_score: Optional[float] = None,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        audio_duration: Optional[float] = None,
        failure_reason: Optional[str] = None
    ) -> AuthenticationLog:
        """
        Log an authentication attempt (success or failure)
        
        Args:
            username_attempted: Username that was attempted
            passphrase: Passphrase used
            success: Whether authentication succeeded
            similarity_score: Cosine similarity score
            user_id: User ID if authentication succeeded
            ip_address: IP address of the request
            user_agent: User agent string
            audio_duration: Duration of the audio sample
            failure_reason: Reason for failure (if applicable)
            
        Returns:
            AuthenticationLog object
        """
        async with self.async_session() as session:
            auth_log = AuthenticationLog(
                username_attempted=username_attempted,
                passphrase=passphrase,
                success=success,
                similarity_score=similarity_score,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                audio_duration=audio_duration,
                failure_reason=failure_reason
            )
            session.add(auth_log)
            await session.commit()
            await session.refresh(auth_log)
            
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"Auth attempt logged: {username_attempted} - {status}")
            return auth_log
    
    async def get_user_auth_logs(self, user_id: int, limit: int = 50) -> List[AuthenticationLog]:
        """Get authentication logs for a user"""
        async with self.async_session() as session:
            result = await session.execute(
                select(AuthenticationLog)
                .where(AuthenticationLog.user_id == user_id)
                .order_by(AuthenticationLog.attempted_at.desc())
                .limit(limit)
            )
            return result.scalars().all()
    
    async def get_failed_auth_attempts(self, username: str, limit: int = 10) -> List[AuthenticationLog]:
        """Get recent failed authentication attempts for a username"""
        async with self.async_session() as session:
            result = await session.execute(
                select(AuthenticationLog)
                .where(
                    and_(
                        AuthenticationLog.username_attempted == username,
                        AuthenticationLog.success == False
                    )
                )
                .order_by(AuthenticationLog.attempted_at.desc())
                .limit(limit)
            )
            return result.scalars().all()
    
    async def close(self):
        """Close database connections"""
        await self.engine.dispose()
        logger.info("Database connections closed")


# Singleton instance for use across the application
db_manager = DatabaseManager()