"""
Voice Authentication API Server - Fixed & Integrated
FastAPI backend that connects frontend to voice processing core

This is like a restaurant:
- Frontend is the customer (orders food)
- Server is the waiter (takes orders, brings food)
- Voice processor is the kitchen (makes the food)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf
import os
import json
from datetime import datetime
import io
from typing import Optional

# Import our fixed voice processor
from voice_processor import VoiceProcessor, load_audio_from_bytes, calculate_adaptive_thresholds

app = FastAPI(title="Voice Authentication API", version="6.0 - Fixed")

# Enable CORS - allows frontend to talk to backend
# Like allowing phone calls from specific numbers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_DIR = "voice_data"
LOGS_DIR = os.path.join(DATA_DIR, "logs")
SAMPLE_RATE = 16000

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize voice processor
processor = VoiceProcessor(sample_rate=SAMPLE_RATE)


@app.get("/")
async def root():
    """API status check - like knocking on the door to see if anyone's home"""
    return {
        "status": "online",
        "message": "Voice Authentication API - Fixed Version",
        "version": "6.0",
        "features": "172 voice features • 4 metrics • Adaptive thresholds"
    }


@app.get("/api/users")
async def list_users():
    """
    List all registered users
    Like looking at a phone directory
    """
    users = []
    try:
        for file in os.listdir(DATA_DIR):
            if file.endswith('_profile.json'):
                username = file.replace('_profile.json', '')
                profile_path = os.path.join(DATA_DIR, file)
                
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                
                users.append({
                    'username': username,
                    'registered': profile.get('registered_at', 'Unknown'),
                    'feature_count': profile.get('feature_count', 0)
                })
        
        return {"users": users, "count": len(users)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/register")
async def register(
    username: str = Form(...),
    audio1: UploadFile = File(...),
    audio2: UploadFile = File(...),
    audio3: UploadFile = File(...)
):
    """
    Register a new user with 3 audio samples
    
    This is like taking multiple photos for a passport:
    - We need several samples to get a good average
    - More samples = more accurate identification
    - We calibrate thresholds based on these samples
    """
    try:
        # Check if user already exists
        profile_path = os.path.join(DATA_DIR, f"{username}_profile.json")
        if os.path.exists(profile_path):
            raise HTTPException(
                status_code=400, 
                detail=f"User '{username}' already exists. Please delete first or choose a different username."
            )
        
        # Process all 3 audio files
        audio_files = [audio1, audio2, audio3]
        feature_vectors = []
        frame_features_list = []
        
        print(f"\n🎤 Processing registration for {username}...")
        
        for i, audio_file in enumerate(audio_files, 1):
            # Read audio bytes from upload
            audio_bytes = await audio_file.read()
            
            # Convert to numpy array - this handles format conversion
            audio = load_audio_from_bytes(audio_bytes, target_sr=SAMPLE_RATE)
            
            # Save the audio file as WAV
            audio_path = os.path.join(DATA_DIR, f"{username}_sample_{i}.wav")
            sf.write(audio_path, audio, SAMPLE_RATE)
            print(f"   ✓ Saved sample {i}")
            
            # Extract features
            features, frame_feats = processor.extract_features(audio, SAMPLE_RATE)
            feature_vectors.append(features)
            frame_features_list.append(frame_feats)
            print(f"   ✓ Extracted {len(features)} features from sample {i}")
        
        # Calculate average features
        avg_features = np.mean(feature_vectors, axis=0)
        
        # Calculate adaptive thresholds based on consistency
        thresholds, stats = calculate_adaptive_thresholds(
            feature_vectors, 
            frame_features_list,
            processor
        )
        
        print(f"\n📊 Registration Quality:")
        print(f"   Cosine:         {stats['cosine_mean']*100:.1f}%")
        print(f"   Euclidean:      {stats['euclidean_mean']*100:.1f}%")
        print(f"   Bhattacharyya:  {stats['bhattacharyya_mean']*100:.1f}%")
        print(f"   KS-Test:        {stats['ks_test_mean']*100:.1f}%")
        
        print(f"\n🎯 Calibrated Thresholds:")
        print(f"   Cosine:         ≥{thresholds['cosine']*100:.1f}%")
        print(f"   Euclidean:      ≥{thresholds['euclidean']*100:.1f}%")
        print(f"   Bhattacharyya:  ≥{thresholds['bhattacharyya']*100:.1f}%")
        print(f"   KS-Test:        ≥{thresholds['ks_test']*100:.1f}%")
        
        # Save user profile
        profile = {
            'username': username,
            'features': avg_features.tolist(),
            'thresholds': thresholds,
            'registration_stats': stats,
            'registered_at': datetime.now().isoformat(),
            'sample_count': 3,
            'feature_count': len(avg_features)
        }
        
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        print(f"✅ Registration complete!\n")
        
        return {
            "success": True,
            "message": f"User '{username}' registered successfully",
            "username": username,
            "consistency": stats,
            "thresholds": thresholds,
            "feature_count": len(avg_features)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/api/authenticate")
async def authenticate(
    username: str = Form(...),
    audio: UploadFile = File(...)
):
    """
    Authenticate a user
    
    This is like showing your ID at airport security:
    - We compare what you just gave us with what's on file
    - We use 4 different checks to be thorough
    - Need to pass 3 out of 4 checks to get through
    """
    try:
        # Load user profile
        profile_path = os.path.join(DATA_DIR, f"{username}_profile.json")
        if not os.path.exists(profile_path):
            raise HTTPException(
                status_code=404, 
                detail=f"User '{username}' not found. Please register first."
            )
        
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        stored_features = np.array(profile['features'])
        thresholds = profile['thresholds']
        
        print(f"\n🔐 Authenticating {username}...")
        
        # Process uploaded audio
        audio_bytes = await audio.read()
        test_audio = load_audio_from_bytes(audio_bytes, target_sr=SAMPLE_RATE)
        
        # Extract features from test audio
        test_features, test_frame_features = processor.extract_features(test_audio, SAMPLE_RATE)
        print(f"   ✓ Extracted features from test audio")
        
        # Load reference frame features from first registration sample
        ref_audio_path = os.path.join(DATA_DIR, f"{username}_sample_1.wav")
        if os.path.exists(ref_audio_path):
            ref_audio, _ = sf.read(ref_audio_path)
            _, ref_frame_features = processor.extract_features(ref_audio, SAMPLE_RATE)
        else:
            ref_frame_features = test_frame_features
        
        # Compute similarity scores
        scores = processor.compute_similarity(
            stored_features, test_features,
            ref_frame_features, test_frame_features
        )
        
        # Check each metric against its threshold
        checks = {
            'cosine': scores['cosine'] >= thresholds['cosine'],
            'euclidean': scores['euclidean'] >= thresholds['euclidean'],
            'bhattacharyya': scores['bhattacharyya'] >= thresholds['bhattacharyya'],
            'ks_test': scores['ks_test'] >= thresholds['ks_test'],
        }
        
        # Count how many checks passed
        checks_passed = sum(checks.values())
        
        # Need 3 out of 4 to authenticate
        authenticated = checks_passed >= 3
        
        # Calculate combined weighted score
        combined = (
            0.30 * scores['cosine'] +
            0.20 * scores['euclidean'] +
            0.35 * scores['bhattacharyya'] +
            0.15 * scores['ks_test']
        )
        
        print(f"\n📊 Authentication Results:")
        print(f"   Cosine:         {scores['cosine']*100:5.1f}% {'✅' if checks['cosine'] else '❌'}")
        print(f"   Euclidean:      {scores['euclidean']*100:5.1f}% {'✅' if checks['euclidean'] else '❌'}")
        print(f"   Bhattacharyya:  {scores['bhattacharyya']*100:5.1f}% {'✅' if checks['bhattacharyya'] else '❌'}")
        print(f"   KS-Test:        {scores['ks_test']*100:5.1f}% {'✅' if checks['ks_test'] else '❌'}")
        print(f"   Combined:       {combined*100:.1f}%")
        print(f"   Checks Passed:  {checks_passed}/4")
        print(f"   Result:         {'✅ SUCCESS' if authenticated else '❌ FAILED'}\n")
        
        # Log the authentication attempt
        log_entry = {
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'scores': scores,
            'thresholds': thresholds,
            'checks': checks,
            'checks_passed': checks_passed,
            'combined_score': float(combined),
            'success': authenticated
        }
        
        log_file = os.path.join(LOGS_DIR, f"{username}_auth_log.json")
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        logs.append(log_entry)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return {
            "success": authenticated,
            "username": username,
            "scores": scores,
            "thresholds": thresholds,
            "checks": checks,
            "checks_passed": checks_passed,
            "required_checks": 3,
            "combined_score": float(combined),
            "message": "Authentication successful! 🎉" if authenticated else "Authentication failed. Voice does not match."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Authentication error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")


@app.get("/api/stats/{username}")
async def get_stats(username: str):
    """Get authentication statistics for a user"""
    try:
        log_file = os.path.join(LOGS_DIR, f"{username}_auth_log.json")
        
        if not os.path.exists(log_file):
            return {
                "username": username,
                "total_attempts": 0,
                "successful": 0,
                "failed": 0,
                "recent_attempts": []
            }
        
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        total = len(logs)
        successful = sum(1 for log in logs if log['success'])
        failed = total - successful
        
        return {
            "username": username,
            "total_attempts": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "recent_attempts": logs[-10:]  # Last 10 attempts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/users/{username}")
async def delete_user(username: str):
    """Delete a user profile"""
    try:
        profile_path = os.path.join(DATA_DIR, f"{username}_profile.json")
        
        if not os.path.exists(profile_path):
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")
        
        # Delete profile
        os.remove(profile_path)
        
        # Delete audio samples
        for i in range(1, 4):
            sample_path = os.path.join(DATA_DIR, f"{username}_sample_{i}.wav")
            if os.path.exists(sample_path):
                os.remove(sample_path)
        
        # Delete logs
        log_file = os.path.join(LOGS_DIR, f"{username}_auth_log.json")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        print(f"🗑️  Deleted user: {username}")
        
        return {"success": True, "message": f"User '{username}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("="*70)
    print("🎙️  VOICE AUTHENTICATION API SERVER - FIXED VERSION")
    print("="*70)
    print("✨ New in v6.0:")
    print("   • Fixed Euclidean distance calculation")
    print("   • Improved audio format handling")
    print("   • Better error messages")
    print("   • Console feedback for debugging")
    print("="*70)
    print("🚀 Starting server on http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("="*70)
    uvicorn.run(app, host="0.0.0.0", port=8000)