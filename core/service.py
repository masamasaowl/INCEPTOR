"""
Voice Authentication Service
High-level service layer that orchestrates voice registration and authentication
Combines the voice engine with database operations
"""

import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import uuid
from datetime import datetime

from core.voice_auth import voice_auth_engine
from database.manager import db_manager
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceAuthService:
    """
    Voice Authentication Service
    
    Provides high-level methods for:
    - User registration with voice samples
    - Voice-based authentication
    - User management
    """
    
    def __init__(self):
        """Initialize the voice authentication service"""
        self.engine = voice_auth_engine
        self.db = db_manager
        logger.info("VoiceAuthService initialized")
    
    async def register_user(self, username: str, email: Optional[str] = None) -> Dict[str, Any]:
        """
        Register a new user (without voice samples yet)
        
        Args:
            username: Unique username
            email: Optional email address
            
        Returns:
            Dictionary with user information
            
        Raises:
            ValueError: If username already exists
        """
        try:
            user = await self.db.create_user(username=username, email=email)
            
            return {
                'success': True,
                'user_uuid': user.uuid,
                'username': user.username,
                'message': f"User '{username}' registered. Please provide {settings.REQUIRED_SAMPLES} voice samples.",
                'samples_required': settings.REQUIRED_SAMPLES,
                'samples_collected': 0
            }
        except ValueError as e:
            logger.error(f"Registration failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def add_voice_sample(
        self,
        username: str,
        audio_file_path: str,
        passphrase: str
    ) -> Dict[str, Any]:
        """
        Add a voice sample for a user during enrollment
        
        Args:
            username: Username of the user
            audio_file_path: Path to the audio file
            passphrase: What the user said in the recording
            
        Returns:
            Dictionary with enrollment status
        """
        try:
            # Get user from database
            user = await self.db.get_user_by_username(username)
            if not user:
                return {
                    'success': False,
                    'error': f"User '{username}' not found. Please register first."
                }
            
            # Extract voice embedding
            logger.info(f"Processing voice sample for user: {username}")
            embedding = self.engine.extract_embedding(audio_file_path)
            
            # Calculate audio quality metrics
            quality_metrics = self.engine.calculate_audio_quality_metrics(audio_file_path)
            
            # Generate unique filename for embedding
            embedding_filename = f"{user.uuid}_{uuid.uuid4().hex[:8]}.npy"
            embedding_path = settings.VOICEPRINT_DIR / embedding_filename
            
            # Save embedding to disk
            saved_path = self.engine.save_embedding(embedding, str(embedding_path))
            
            # Add voice sample to database
            await self.db.add_voice_sample(
                user_id=user.id,
                passphrase=passphrase,
                audio_duration=quality_metrics['duration'],
                sample_rate=quality_metrics['sample_rate'],
                embedding_path=saved_path,
                embedding_dimension=len(embedding),
                snr=quality_metrics['snr_db'],
                confidence=None  # Can add model confidence if needed
            )
            
            # Check how many samples user has now
            sample_count = await self.db.count_user_voice_samples(user.id)
            
            # If user has enough samples, mark as registered
            if sample_count >= settings.REQUIRED_SAMPLES:
                await self.db.update_user_registration_status(user.id, True)
                logger.info(f"User {username} fully registered with {sample_count} samples")
                
                return {
                    'success': True,
                    'message': f"Registration complete! User '{username}' can now authenticate.",
                    'samples_collected': sample_count,
                    'samples_required': settings.REQUIRED_SAMPLES,
                    'fully_registered': True,
                    'quality_metrics': quality_metrics
                }
            else:
                samples_remaining = settings.REQUIRED_SAMPLES - sample_count
                return {
                    'success': True,
                    'message': f"Voice sample added. {samples_remaining} more sample(s) needed.",
                    'samples_collected': sample_count,
                    'samples_required': settings.REQUIRED_SAMPLES,
                    'fully_registered': False,
                    'quality_metrics': quality_metrics
                }
                
        except FileNotFoundError as e:
            return {'success': False, 'error': f"Audio file not found: {e}"}
        except ValueError as e:
            return {'success': False, 'error': f"Invalid audio: {e}"}
        except Exception as e:
            logger.error(f"Error adding voice sample: {e}")
            return {'success': False, 'error': f"Failed to process voice sample: {str(e)}"}
    
    async def authenticate(
        self,
        username: str,
        audio_file_path: str,
        passphrase: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Authenticate a user using their voice
        
        Args:
            username: Username to authenticate
            audio_file_path: Path to the audio file
            passphrase: What the user said (should match enrollment)
            ip_address: Optional IP address for logging
            user_agent: Optional user agent for logging
            
        Returns:
            Dictionary with authentication result
        """
        try:
            # Get user from database
            user = await self.db.get_user_by_username(username)
            if not user:
                await self.db.log_authentication_attempt(
                    username_attempted=username,
                    passphrase=passphrase,
                    success=False,
                    failure_reason="User not found",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                return {
                    'success': False,
                    'authenticated': False,
                    'error': f"User '{username}' not found"
                }
            
            # Check if user is fully registered
            if not user.is_registered:
                sample_count = await self.db.count_user_voice_samples(user.id)
                await self.db.log_authentication_attempt(
                    username_attempted=username,
                    passphrase=passphrase,
                    success=False,
                    user_id=user.id,
                    failure_reason="User not fully registered",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                return {
                    'success': False,
                    'authenticated': False,
                    'error': f"User registration incomplete. {sample_count}/{settings.REQUIRED_SAMPLES} samples collected."
                }
            
            # Extract embedding from test audio
            logger.info(f"Authenticating user: {username}")
            test_embedding = self.engine.extract_embedding(audio_file_path)
            
            # Get audio quality metrics
            quality_metrics = self.engine.calculate_audio_quality_metrics(audio_file_path)
            
            # Load all reference embeddings for the user
            voice_samples = await self.db.get_user_voice_samples(user.id)
            reference_embeddings = [
                self.engine.load_embedding(sample.embedding_path)
                for sample in voice_samples
            ]
            
            # Verify speaker
            is_match, similarity_score = self.engine.verify_speaker(
                test_embedding=test_embedding,
                reference_embeddings=reference_embeddings
            )
            
            # Log authentication attempt
            await self.db.log_authentication_attempt(
                username_attempted=username,
                passphrase=passphrase,
                success=is_match,
                similarity_score=similarity_score,
                user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent,
                audio_duration=quality_metrics['duration'],
                failure_reason=None if is_match else "Voice does not match"
            )
            
            if is_match:
                logger.info(f"Authentication SUCCESS for {username} (score: {similarity_score:.3f})")
                return {
                    'success': True,
                    'authenticated': True,
                    'username': username,
                    'user_uuid': user.uuid,
                    'similarity_score': similarity_score,
                    'threshold': settings.SIMILARITY_THRESHOLD,
                    'message': "Authentication successful",
                    'quality_metrics': quality_metrics
                }
            else:
                logger.warning(f"Authentication FAILED for {username} (score: {similarity_score:.3f})")
                return {
                    'success': True,
                    'authenticated': False,
                    'similarity_score': similarity_score,
                    'threshold': settings.SIMILARITY_THRESHOLD,
                    'message': "Voice does not match. Authentication failed."
                }
                
        except FileNotFoundError as e:
            return {'success': False, 'authenticated': False, 'error': f"Audio file not found: {e}"}
        except ValueError as e:
            return {'success': False, 'authenticated': False, 'error': f"Invalid audio: {e}"}
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return {'success': False, 'authenticated': False, 'error': f"Authentication failed: {str(e)}"}
    
    async def get_user_info(self, username: str) -> Dict[str, Any]:
        """
        Get information about a user
        
        Args:
            username: Username to look up
            
        Returns:
            Dictionary with user information
        """
        user = await self.db.get_user_by_username(username)
        if not user:
            return {'success': False, 'error': f"User '{username}' not found"}
        
        sample_count = await self.db.count_user_voice_samples(user.id)
        
        return {
            'success': True,
            'username': user.username,
            'user_uuid': user.uuid,
            'email': user.email,
            'is_registered': user.is_registered,
            'samples_collected': sample_count,
            'samples_required': settings.REQUIRED_SAMPLES,
            'created_at': user.created_at.isoformat(),
            'is_active': user.is_active
        }
    
    async def list_users(self) -> Dict[str, Any]:
        """
        List all users in the system
        
        Returns:
            Dictionary with list of users
        """
        users = await self.db.list_all_users()
        
        user_list = []
        for user in users:
            sample_count = await self.db.count_user_voice_samples(user.id)
            user_list.append({
                'username': user.username,
                'user_uuid': user.uuid,
                'is_registered': user.is_registered,
                'samples_collected': sample_count,
                'created_at': user.created_at.isoformat()
            })
        
        return {
            'success': True,
            'total_users': len(user_list),
            'users': user_list
        }
    
    async def delete_user(self, username: str) -> Dict[str, Any]:
        """
        Delete a user and all their voice data
        
        Args:
            username: Username to delete
            
        Returns:
            Dictionary with deletion result
        """
        # Get user's voice samples to delete embedding files
        user = await self.db.get_user_by_username(username)
        if user:
            voice_samples = await self.db.get_user_voice_samples(user.id)
            
            # Delete embedding files from disk
            for sample in voice_samples:
                embedding_path = Path(sample.embedding_path)
                if embedding_path.exists():
                    embedding_path.unlink()
                    logger.info(f"Deleted embedding file: {embedding_path}")
        
        # Delete user from database (cascades to voice samples and logs)
        deleted = await self.db.delete_user(username)
        
        if deleted:
            return {
                'success': True,
                'message': f"User '{username}' and all associated data deleted"
            }
        else:
            return {
                'success': False,
                'error': f"User '{username}' not found"
            }
    
    async def get_authentication_history(self, username: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get authentication history for a user
        
        Args:
            username: Username to look up
            limit: Maximum number of records to return
            
        Returns:
            Dictionary with authentication history
        """
        user = await self.db.get_user_by_username(username)
        if not user:
            return {'success': False, 'error': f"User '{username}' not found"}
        
        auth_logs = await self.db.get_user_auth_logs(user.id, limit=limit)
        
        history = []
        for log in auth_logs:
            history.append({
                'timestamp': log.attempted_at.isoformat(),
                'success': log.success,
                'similarity_score': log.similarity_score,
                'passphrase': log.passphrase,
                'ip_address': log.ip_address,
                'failure_reason': log.failure_reason
            })
        
        return {
            'success': True,
            'username': username,
            'total_attempts': len(history),
            'history': history
        }


# Singleton instance for use across the application
voice_auth_service = VoiceAuthService()