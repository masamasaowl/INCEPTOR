"""
Voice Authentication API Server
FastAPI backend for voice authentication system
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import soundfile as sf
import os
import json
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import ks_2samp
from datetime import datetime
import io
from typing import Optional

app = FastAPI(title="Voice Authentication API", version="5.0")

# Enable CORS for frontend
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


class VoiceProcessor:
    """Voice feature extraction and comparison"""
    
    @staticmethod
    def extract_features(audio):
        """Extract comprehensive voice features"""
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_median = np.median(mfccs, axis=1)
        
        # Delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mean = np.mean(delta_mfccs, axis=1)
        delta_std = np.std(delta_mfccs, axis=1)
        
        # Log-Mel
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=40)
        log_mel = librosa.power_to_db(mel_spec)
        log_mel_mean = np.mean(log_mel, axis=1)
        log_mel_std = np.std(log_mel, axis=1)
        
        # Pitch
        f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), 
                         fmax=librosa.note_to_hz('C7'), sr=SAMPLE_RATE)
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) > 0:
            pitch_features = np.array([
                np.mean(f0_voiced), np.std(f0_voiced),
                np.min(f0_voiced), np.max(f0_voiced),
                np.ptp(f0_voiced), np.median(f0_voiced)
            ])
        else:
            pitch_features = np.zeros(6)
        
        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)[0]
        
        # Combine all
        features = np.concatenate([
            mfcc_mean, mfcc_std, mfcc_median,
            delta_mean, delta_std,
            log_mel_mean, log_mel_std,
            pitch_features,
            [np.mean(centroid), np.std(centroid)],
            [np.mean(rolloff), np.std(rolloff)]
        ])
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Frame features for distribution comparison
        frame_features = {
            'mfccs': mfccs.T,
            'log_mel': log_mel.T,
            'delta_mfccs': delta_mfccs.T,
        }
        
        return features, frame_features
    
    @staticmethod
    def compute_similarity(features1, features2, frame1, frame2):
        """Compute similarity metrics"""
        # Cosine
        cosine_dist = cosine(features1, features2)
        cosine_sim = 1 - cosine_dist
        
        # Euclidean (fixed)
        euclidean_dist = euclidean(features1, features2)
        max_possible_dist = np.sqrt(len(features1))
        euclidean_sim = 1 - (euclidean_dist / (max_possible_dist + 1e-8))
        
        # Bhattacharyya
        mfcc1 = frame1['mfccs']
        mfcc2 = frame2['mfccs']
        
        bhatt_sims = []
        for i in range(min(mfcc1.shape[1], mfcc2.shape[1])):
            hist1, bins = np.histogram(mfcc1[:, i], bins=20, density=True)
            hist2, _ = np.histogram(mfcc2[:, i], bins=bins, density=True)
            
            hist1 = hist1 / (np.sum(hist1) + 1e-8)
            hist2 = hist2 / (np.sum(hist2) + 1e-8)
            
            bc = np.sum(np.sqrt(hist1 * hist2))
            bhatt_sims.append(bc)
        
        bhatt_sim = np.mean(bhatt_sims)
        
        # KS-Test
        ks_pvalues = []
        for i in range(min(5, mfcc1.shape[1], mfcc2.shape[1])):
            _, pvalue = ks_2samp(mfcc1[:, i], mfcc2[:, i])
            ks_pvalues.append(pvalue)
        
        ks_sim = np.mean(ks_pvalues)
        
        return {
            'cosine': float(cosine_sim),
            'euclidean': float(euclidean_sim),
            'bhattacharyya': float(bhatt_sim),
            'ks_test': float(ks_sim)
        }


@app.get("/")
async def root():
    """API status"""
    return {
        "status": "online",
        "message": "Voice Authentication API",
        "version": "5.0"
    }


@app.get("/api/users")
async def list_users():
    """List all registered users"""
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
    """Register a new user with 3 audio samples"""
    try:
        # Check if user exists
        profile_path = os.path.join(DATA_DIR, f"{username}_profile.json")
        if os.path.exists(profile_path):
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Process audio files
        audio_files = [audio1, audio2, audio3]
        feature_vectors = []
        frame_features_list = []
        
        for i, audio_file in enumerate(audio_files, 1):
            # Read audio
            audio_bytes = await audio_file.read()
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            
            # Resample if needed
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            
            # Save audio file
            audio_path = os.path.join(DATA_DIR, f"{username}_sample_{i}.wav")
            sf.write(audio_path, audio, SAMPLE_RATE)
            
            # Extract features
            features, frame_feats = VoiceProcessor.extract_features(audio)
            feature_vectors.append(features)
            frame_features_list.append(frame_feats)
        
        # Calculate average features
        avg_features = np.mean(feature_vectors, axis=0)
        
        # Calibrate thresholds
        similarity_scores = []
        for i in range(len(feature_vectors)):
            for j in range(i+1, len(feature_vectors)):
                scores = VoiceProcessor.compute_similarity(
                    feature_vectors[i], feature_vectors[j],
                    frame_features_list[i], frame_features_list[j]
                )
                similarity_scores.append(scores)
        
        # Calculate statistics
        avg_cosine = np.mean([s['cosine'] for s in similarity_scores])
        avg_euclidean = np.mean([s['euclidean'] for s in similarity_scores])
        avg_bhatt = np.mean([s['bhattacharyya'] for s in similarity_scores])
        avg_ks = np.mean([s['ks_test'] for s in similarity_scores])
        
        std_cosine = np.std([s['cosine'] for s in similarity_scores])
        std_euclidean = np.std([s['euclidean'] for s in similarity_scores])
        std_bhatt = np.std([s['bhattacharyya'] for s in similarity_scores])
        std_ks = np.std([s['ks_test'] for s in similarity_scores])
        
        # Set thresholds
        safety_margin = 2.5
        thresholds = {
            'cosine': max(0.70, avg_cosine - safety_margin * std_cosine),
            'euclidean': max(0.40, avg_euclidean - safety_margin * std_euclidean),
            'bhattacharyya': max(0.50, avg_bhatt - safety_margin * std_bhatt),
            'ks_test': max(0.05, avg_ks - safety_margin * std_ks),
        }
        
        # Save profile
        profile = {
            'username': username,
            'features': avg_features.tolist(),
            'thresholds': thresholds,
            'registration_stats': {
                'cosine_mean': float(avg_cosine),
                'euclidean_mean': float(avg_euclidean),
                'bhattacharyya_mean': float(avg_bhatt),
                'ks_test_mean': float(avg_ks),
            },
            'registered_at': datetime.now().isoformat(),
            'sample_count': 3,
            'feature_count': len(avg_features)
        }
        
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        return {
            "success": True,
            "message": f"User '{username}' registered successfully",
            "username": username,
            "consistency": {
                "cosine": float(avg_cosine),
                "euclidean": float(avg_euclidean),
                "bhattacharyya": float(avg_bhatt),
                "ks_test": float(avg_ks)
            },
            "thresholds": thresholds
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/authenticate")
async def authenticate(
    username: str = Form(...),
    audio: UploadFile = File(...)
):
    """Authenticate a user"""
    try:
        # Load profile
        profile_path = os.path.join(DATA_DIR, f"{username}_profile.json")
        if not os.path.exists(profile_path):
            raise HTTPException(status_code=404, detail="User not found")
        
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        stored_features = np.array(profile['features'])
        thresholds = profile['thresholds']
        
        # Process audio
        audio_bytes = await audio.read()
        test_audio, sr = sf.read(io.BytesIO(audio_bytes))
        
        if sr != SAMPLE_RATE:
            test_audio = librosa.resample(test_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # Extract features
        test_features, test_frame_features = VoiceProcessor.extract_features(test_audio)
        
        # Load reference frame features
        ref_audio_path = os.path.join(DATA_DIR, f"{username}_sample_1.wav")
        if os.path.exists(ref_audio_path):
            ref_audio, _ = sf.read(ref_audio_path)
            _, ref_frame_features = VoiceProcessor.extract_features(ref_audio)
        else:
            ref_frame_features = test_frame_features
        
        # Compute similarities
        scores = VoiceProcessor.compute_similarity(
            stored_features, test_features,
            ref_frame_features, test_frame_features
        )
        
        # Check thresholds
        checks = {
            'cosine': scores['cosine'] >= thresholds['cosine'],
            'euclidean': scores['euclidean'] >= thresholds['euclidean'],
            'bhattacharyya': scores['bhattacharyya'] >= thresholds['bhattacharyya'],
            'ks_test': scores['ks_test'] >= thresholds['ks_test'],
        }
        
        checks_passed = sum(checks.values())
        authenticated = checks_passed >= 3
        
        # Combined score
        combined = (
            0.30 * scores['cosine'] +
            0.20 * scores['euclidean'] +
            0.35 * scores['bhattacharyya'] +
            0.15 * scores['ks_test']
        )
        
        # Log attempt
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
            "message": "Authentication successful!" if authenticated else "Authentication failed!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            raise HTTPException(status_code=404, detail="User not found")
        
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
        
        return {"success": True, "message": f"User '{username}' deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("🎙️  Voice Authentication API Server")
    print("=" * 50)
    print("Starting server on http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)