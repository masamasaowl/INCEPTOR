"""
FastAPI Application for Voice Authentication System
RESTful API for voice registration and authentication
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from config import settings, ensure_directories
from core.service import voice_auth_service
from database.manager import db_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Voice Authentication System API - Speaker verification and authentication",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (configure based on your needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Pydantic Models ====================

class UserRegistrationRequest(BaseModel):
    """Request model for user registration"""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: Optional[str] = Field(None, description="Optional email address")


class UserRegistrationResponse(BaseModel):
    """Response model for user registration"""
    success: bool
    user_uuid: Optional[str] = None
    username: Optional[str] = None
    message: str
    samples_required: Optional[int] = None
    samples_collected: Optional[int] = None
    error: Optional[str] = None


class AuthenticationResponse(BaseModel):
    """Response model for authentication"""
    success: bool
    authenticated: bool
    username: Optional[str] = None
    user_uuid: Optional[str] = None
    similarity_score: Optional[float] = None
    threshold: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None


class UserInfoResponse(BaseModel):
    """Response model for user information"""
    success: bool
    username: Optional[str] = None
    user_uuid: Optional[str] = None
    email: Optional[str] = None
    is_registered: Optional[bool] = None
    samples_collected: Optional[int] = None
    samples_required: Optional[int] = None
    created_at: Optional[str] = None
    is_active: Optional[bool] = None
    error: Optional[str] = None


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """
    Initialize application on startup
    - Create necessary directories
    - Initialize database
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Ensure directories exist
    ensure_directories()
    logger.info("Directories created/verified")
    
    # Initialize database
    await db_manager.initialize_database()
    logger.info("Database initialized")
    
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on application shutdown
    """
    logger.info("Shutting down application...")
    await db_manager.close()
    logger.info("Application shutdown complete")


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """
    Root endpoint - API information
    """
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "register_user": "POST /api/v1/register",
            "add_voice_sample": "POST /api/v1/enroll",
            "authenticate": "POST /api/v1/authenticate",
            "user_info": "GET /api/v1/user/{username}",
            "list_users": "GET /api/v1/users",
            "delete_user": "DELETE /api/v1/user/{username}",
            "auth_history": "GET /api/v1/user/{username}/history"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected"
    }


@app.post(f"{settings.API_PREFIX}/register", response_model=UserRegistrationResponse)
async def register_user(request: UserRegistrationRequest):
    """
    Register a new user
    
    This creates a user account but doesn't register their voice yet.
    After registration, use the /enroll endpoint to add voice samples.
    """
    logger.info(f"Registration request for username: {request.username}")
    
    result = await voice_auth_service.register_user(
        username=request.username,
        email=request.email
    )
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result.get('error', 'Registration failed'))
    
    return result


@app.post(f"{settings.API_PREFIX}/enroll")
async def add_voice_sample(
    username: str = Form(..., description="Username"),
    passphrase: str = Form(..., description="What the user said in the recording"),
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, M4A)")
):
    """
    Add a voice sample for user enrollment
    
    Users must provide at least 3 voice samples (configured in settings) 
    before they can authenticate. Each sample should contain the same passphrase.
    
    Best practices:
    - Use clear, quiet environment
    - Speak naturally at normal pace
    - Use the same passphrase for all samples
    - Provide 3-5 samples for best accuracy
    """
    logger.info(f"Voice enrollment for username: {username}")
    
    # Validate file type
    file_ext = Path(audio_file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Allowed formats: {settings.ALLOWED_AUDIO_FORMATS}"
        )
    
    # Check file size
    audio_file.file.seek(0, 2)  # Seek to end
    file_size = audio_file.file.tell()
    audio_file.file.seek(0)  # Reset to start
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        shutil.copyfileobj(audio_file.file, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # Process voice sample
        result = await voice_auth_service.add_voice_sample(
            username=username,
            audio_file_path=tmp_path,
            passphrase=passphrase
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Enrollment failed'))
        
        return result
        
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)


@app.post(f"{settings.API_PREFIX}/authenticate", response_model=AuthenticationResponse)
async def authenticate(
    request: Request,
    username: str = Form(..., description="Username to authenticate"),
    passphrase: str = Form(..., description="Passphrase spoken in the recording"),
    audio_file: UploadFile = File(..., description="Audio file for authentication")
):
    """
    Authenticate a user using their voice
    
    The user must have completed enrollment (provided required voice samples).
    The system will compare the voice in the audio file against the user's
    registered voiceprint.
    
    Returns:
    - authenticated: True if voice matches, False otherwise
    - similarity_score: How similar the voices are (0-1)
    - threshold: The similarity threshold used for authentication
    """
    logger.info(f"Authentication attempt for username: {username}")
    
    # Validate file type
    file_ext = Path(audio_file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Allowed formats: {settings.ALLOWED_AUDIO_FORMATS}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        shutil.copyfileobj(audio_file.file, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # Get client info for logging
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        # Authenticate
        result = await voice_auth_service.authenticate(
            username=username,
            audio_file_path=tmp_path,
            passphrase=passphrase,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Authentication failed'))
        
        return result
        
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)


@app.get(f"{settings.API_PREFIX}/user/{{username}}", response_model=UserInfoResponse)
async def get_user_info(username: str):
    """
    Get information about a user
    
    Returns user details including registration status and number of voice samples.
    """
    result = await voice_auth_service.get_user_info(username)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result.get('error', 'User not found'))
    
    return result


@app.get(f"{settings.API_PREFIX}/users")
async def list_users():
    """
    List all users in the system
    
    Returns a list of all registered users with their basic information.
    """
    result = await voice_auth_service.list_users()
    return result


@app.delete(f"{settings.API_PREFIX}/user/{{username}}")
async def delete_user(username: str):
    """
    Delete a user and all their voice data
    
    This permanently deletes:
    - User account
    - All voice samples and embeddings
    - Authentication history
    
    This action cannot be undone.
    """
    result = await voice_auth_service.delete_user(username)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result.get('error', 'User not found'))
    
    return result


@app.get(f"{settings.API_PREFIX}/user/{{username}}/history")
async def get_authentication_history(username: str, limit: int = 20):
    """
    Get authentication history for a user
    
    Returns recent authentication attempts (both successful and failed).
    Useful for security monitoring and analytics.
    """
    result = await voice_auth_service.get_authentication_history(username, limit=limit)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result.get('error', 'User not found'))
    
    return result


@app.get(f"{settings.API_PREFIX}/settings")
async def get_settings():
    """
    Get current system settings
    
    Returns configuration parameters like similarity threshold,
    required samples, etc.
    """
    return {
        "sample_rate": settings.SAMPLE_RATE,
        "min_audio_length": settings.MIN_AUDIO_LENGTH,
        "max_audio_length": settings.MAX_AUDIO_LENGTH,
        "required_samples": settings.REQUIRED_SAMPLES,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "allowed_audio_formats": settings.ALLOWED_AUDIO_FORMATS,
        "max_upload_size_mb": settings.MAX_UPLOAD_SIZE / 1024 / 1024
    }


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )