"""
Voice Authentication System V3
Fixed with better feature extraction and voice activity detection
"""

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import os
import json
from scipy.spatial.distance import cosine, euclidean
from datetime import datetime
import time

class VoiceAuthenticatorV3:
    def __init__(self, data_dir="voice_data"):
        """Initialize the voice authentication system"""
        self.data_dir = data_dir
        self.sample_rate = 16000
        self.duration = 3
        
        # Create directories
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        self.logs_dir = os.path.join(self.data_dir, "logs")
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
    
    def detect_voice_activity(self, audio, energy_threshold=0.02):
        """Detect the portion of audio that contains actual speech"""
        # Calculate energy in frames
        frame_length = 512
        hop_length = 256
        
        # Calculate RMS energy for each frame
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find frames with energy above threshold
        voice_frames = rms > energy_threshold
        
        if not np.any(voice_frames):
            print("⚠️  WARNING: No speech detected! Audio might be too quiet.")
            return audio
        
        # Convert frame indices to sample indices
        voice_samples = np.zeros(len(audio), dtype=bool)
        for i, is_voice in enumerate(voice_frames):
            start = i * hop_length
            end = min(start + frame_length, len(audio))
            if is_voice:
                voice_samples[start:end] = True
        
        # Extract only the voice portions
        voice_audio = audio[voice_samples]
        
        print(f"🎤 Voice Activity: {len(voice_audio)/len(audio)*100:.1f}% of recording contains speech")
        
        return voice_audio if len(voice_audio) > self.sample_rate * 0.5 else audio
    
    def extract_features(self, audio):
        """Extract comprehensive voice features from audio signal"""
        
        # First, detect and isolate speech
        speech_audio = self.detect_voice_activity(audio)
        
        # 1. MFCCs - captures vocal tract characteristics
        mfccs = librosa.feature.mfcc(
            y=speech_audio,
            sr=self.sample_rate,
            n_mfcc=20  # Increased from 13 for more detail
        )
        
        # 2. Delta MFCCs - captures how voice changes over time
        delta_mfccs = librosa.feature.delta(mfccs)
        
        # 3. Pitch/Fundamental frequency - unique to each person
        pitches, magnitudes = librosa.piptrack(y=speech_audio, sr=self.sample_rate)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        # 4. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=speech_audio, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=speech_audio, sr=self.sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=speech_audio, sr=self.sample_rate)
        
        # 5. Zero crossing rate - voice texture
        zcr = librosa.feature.zero_crossing_rate(speech_audio)
        
        # Aggregate features
        features = np.concatenate([
            np.mean(mfccs, axis=1),           # 20 features
            np.std(mfccs, axis=1),            # 20 features - variation
            np.mean(delta_mfccs, axis=1),     # 20 features - dynamics
            [np.mean(pitch_values) if pitch_values else 0],  # 1 feature
            [np.std(pitch_values) if pitch_values else 0],   # 1 feature
            [np.mean(spectral_centroid)],     # 1 feature
            [np.std(spectral_centroid)],      # 1 feature
            [np.mean(spectral_rolloff)],      # 1 feature
            [np.mean(spectral_bandwidth)],    # 1 feature
            [np.mean(zcr)],                   # 1 feature
            [np.std(zcr)]                     # 1 feature
        ])
        
        # Normalize features to 0-1 range for better comparison
        features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
        
        print(f"📊 Extracted {len(features)} features from voice")
        
        return features
    
    def record_audio_with_countdown(self):
        """Record audio with visual countdown"""
        print(f"\n🎤 Recording for {self.duration} seconds...")
        print("Say: 'Hello, this is my voice' clearly")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"Starting in {i}...", end='\r')
            time.sleep(1)
        
        print("\n🔴 RECORDING NOW!          ")
        
        # Record audio
        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        
        # Show recording progress
        for i in range(self.duration):
            time.sleep(1)
            print(f"{'█' * (i+1)}{'░' * (self.duration-i-1)} {i+1}/{self.duration}s")
        
        sd.wait()
        
        # Check audio level
        rms = np.sqrt(np.mean(audio**2))
        max_amp = np.max(np.abs(audio))
        
        print(f"✓ Recording complete!")
        print(f"   Audio level: RMS={rms:.4f}, Peak={max_amp:.4f}")
        
        if max_amp < 0.01:
            print("⚠️  WARNING: Very quiet recording! Speak louder.")
        
        return audio.flatten()
    
    def playback_audio(self, audio):
        """Play back the recorded audio"""
        response = input("\n🔊 Play back your recording? (y/n): ").strip().lower()
        if response == 'y':
            print("Playing back...")
            sd.play(audio, self.sample_rate)
            sd.wait()
            print("✓ Playback finished")
            
            satisfied = input("Happy with this recording? (y/n): ").strip().lower()
            return satisfied == 'y'
        return True
    
    def register_user(self, username):
        """Register a new user with their voice"""
        print(f"\n{'='*70}")
        print(f"REGISTRATION MODE - User: {username}")
        print(f"{'='*70}")
        
        feature_vectors = []
        
        for i in range(3):
            while True:
                print(f"\n{'─'*70}")
                print(f"📝 Sample {i+1}/3")
                print(f"{'─'*70}")
                
                audio = self.record_audio_with_countdown()
                
                # Playback and confirm
                if self.playback_audio(audio):
                    # Save the audio file
                    audio_path = os.path.join(
                        self.data_dir,
                        f"{username}_sample_{i+1}.wav"
                    )
                    sf.write(audio_path, audio, self.sample_rate)
                    
                    # Extract features
                    features = self.extract_features(audio)
                    feature_vectors.append(features)
                    break
                else:
                    print("Let's record again...")
            
            if i < 2:
                input("\n⏸️  Press Enter when ready for next sample...")
        
        # Average the feature vectors
        avg_features = np.mean(feature_vectors, axis=0)
        
        # Calculate consistency between samples
        consistency_scores = []
        for features in feature_vectors:
            similarity = 1 - cosine(avg_features, features)
            consistency_scores.append(similarity)
        
        avg_consistency = np.mean(consistency_scores)
        
        print(f"\n{'='*70}")
        print(f"📊 REGISTRATION ANALYSIS")
        print(f"{'='*70}")
        print(f"Consistency Score: {avg_consistency*100:.2f}%")
        print(f"(How similar your 3 samples are to each other)")
        
        if avg_consistency < 0.75:
            print("⚠️  WARNING: Low consistency! Your recordings are quite different.")
            print("This might affect authentication accuracy.")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return False
        else:
            print("✅ Good consistency! Your voice profile is stable.")
        
        print(f"{'='*70}")
        
        # Save the user profile
        user_profile = {
            'username': username,
            'features': avg_features.tolist(),
            'consistency': float(avg_consistency),
            'registered_at': datetime.now().isoformat(),
            'sample_count': 3,
            'feature_count': len(avg_features)
        }
        
        profile_path = os.path.join(self.data_dir, f"{username}_profile.json")
        with open(profile_path, 'w') as f:
            json.dump(user_profile, f, indent=2)
        
        print(f"\n✅ Registration successful!")
        print(f"📁 Profile saved: {profile_path}")
        return True
    
    def authenticate_user(self, username):
        """Authenticate a user by their voice"""
        print(f"\n{'='*70}")
        print(f"AUTHENTICATION MODE - User: {username}")
        print(f"{'='*70}")
        
        # Check if user exists
        profile_path = os.path.join(self.data_dir, f"{username}_profile.json")
        if not os.path.exists(profile_path):
            print(f"❌ User '{username}' not found. Please register first.")
            return False
        
        # Load user profile
        with open(profile_path, 'r') as f:
            user_profile = json.load(f)
        
        stored_features = np.array(user_profile['features'])
        
        # Record authentication attempt
        audio = self.record_audio_with_countdown()
        
        # Playback option
        response = input("\n🔊 Hear your recording? (y/n): ").strip().lower()
        if response == 'y':
            print("Playing back...")
            sd.play(audio, self.sample_rate)
            sd.wait()
        
        # Extract features
        test_features = self.extract_features(audio)
        
        # Calculate multiple similarity metrics
        cosine_dist = cosine(stored_features, test_features)
        cosine_sim = 1 - cosine_dist
        
        euclidean_dist = euclidean(stored_features, test_features)
        # Normalize euclidean distance to 0-1 range
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # Combined similarity score (weighted average)
        similarity = 0.7 * cosine_sim + 0.3 * euclidean_sim
        
        # Adaptive threshold based on registration consistency
        base_threshold = 0.75  # Lowered from 0.85
        consistency_bonus = max(0, (user_profile.get('consistency', 0.9) - 0.85) * 0.1)
        threshold = base_threshold + consistency_bonus
        
        # Feature analysis
        feature_diff = np.abs(stored_features - test_features)
        max_diff = np.max(feature_diff)
        avg_diff = np.mean(feature_diff)
        
        # Calculate how many features are significantly different
        significant_diffs = np.sum(feature_diff > 0.3)
        
        print(f"\n{'='*70}")
        print(f"📊 AUTHENTICATION ANALYSIS")
        print(f"{'='*70}")
        print(f"Similarity Score:        {similarity*100:.2f}%")
        print(f"  - Cosine Similarity:   {cosine_sim*100:.2f}%")
        print(f"  - Euclidean Similarity: {euclidean_sim*100:.2f}%")
        print(f"Required Threshold:      {threshold*100:.2f}%")
        print(f"─"*70)
        print(f"Feature Analysis:")
        print(f"  - Max Difference:      {max_diff:.4f}")
        print(f"  - Avg Difference:      {avg_diff:.4f}")
        print(f"  - Features Changed:    {significant_diffs}/{len(stored_features)}")
        print(f"{'='*70}")
        
        # Log the attempt
        log_entry = {
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'similarity': float(similarity),
            'cosine_sim': float(cosine_sim),
            'euclidean_sim': float(euclidean_sim),
            'threshold': threshold,
            'success': similarity >= threshold,
            'max_diff': float(max_diff),
            'avg_diff': float(avg_diff),
            'features_changed': int(significant_diffs)
        }
        
        log_file = os.path.join(self.logs_dir, f"{username}_auth_log.json")
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        logs.append(log_entry)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        # Decision
        if similarity >= threshold:
            print(f"\n✅ AUTHENTICATION SUCCESSFUL!")
            print(f"Welcome back, {username}! 🎉")
            return True
        else:
            print(f"\n❌ AUTHENTICATION FAILED!")
            print(f"Voice does not match registered profile.")
            print(f"Similarity: {similarity*100:.2f}% < Required: {threshold*100:.2f}%")
            
            # Helpful feedback
            if significant_diffs > len(stored_features) * 0.5:
                print("\n💡 Many features are different. This might be:")
                print("   - A different person")
                print("   - Very different recording conditions")
                print("   - Background noise interference")
            
            return False
    
    def list_users(self):
        """List all registered users"""
        users = []
        for file in os.listdir(self.data_dir):
            if file.endswith('_profile.json'):
                username = file.replace('_profile.json', '')
                profile_path = os.path.join(self.data_dir, file)
                
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                
                log_file = os.path.join(self.logs_dir, f"{username}_auth_log.json")
                auth_count = 0
                success_count = 0
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        logs = json.load(f)
                        auth_count = len(logs)
                        success_count = sum(1 for log in logs if log['success'])
                
                users.append({
                    'username': username,
                    'registered': profile.get('registered_at', 'Unknown'),
                    'consistency': profile.get('consistency', 0),
                    'auth_attempts': auth_count,
                    'successful_auths': success_count
                })
        
        return users
    
    def show_statistics(self, username):
        """Show detailed statistics for a user"""
        log_file = os.path.join(self.logs_dir, f"{username}_auth_log.json")
        
        if not os.path.exists(log_file):
            print(f"\n📊 No authentication history for {username}")
            return
        
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        if not logs:
            print(f"\n📊 No authentication attempts yet")
            return
        
        total = len(logs)
        successful = sum(1 for log in logs if log['success'])
        failed = total - successful
        
        similarities = [log['similarity'] for log in logs]
        avg_similarity = np.mean(similarities)
        
        print(f"\n{'='*70}")
        print(f"📊 AUTHENTICATION STATISTICS - {username}")
        print(f"{'='*70}")
        print(f"Total Attempts:       {total}")
        print(f"✅ Successful:        {successful} ({successful/total*100:.1f}%)")
        print(f"❌ Failed:            {failed} ({failed/total*100:.1f}%)")
        print(f"─"*70)
        print(f"Average Similarity:   {avg_similarity*100:.2f}%")
        print(f"Best Match:           {max(similarities)*100:.2f}%")
        print(f"Worst Match:          {min(similarities)*100:.2f}%")
        print(f"{'='*70}")
        
        print(f"\n📝 Recent Attempts (last 5):")
        for log in logs[-5:]:
            timestamp = datetime.fromisoformat(log['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            status = "✅ Success" if log['success'] else "❌ Failed"
            print(f"{timestamp} | {status} | {log['similarity']*100:.2f}%")


def main():
    """Main function"""
    print("="*70)
    print("🎙️  VOICE AUTHENTICATION SYSTEM V3")
    print("Enhanced Feature Extraction & Voice Activity Detection")
    print("="*70)
    
    auth = VoiceAuthenticatorV3()
    
    while True:
        print("\n" + "="*70)
        print("MENU:")
        print("1. Register new user")
        print("2. Authenticate user")
        print("3. List registered users")
        print("4. View user statistics")
        print("5. Exit")
        print("="*70)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            username = input("\nEnter username to register: ").strip()
            if username:
                auth.register_user(username)
            else:
                print("❌ Username cannot be empty!")
        
        elif choice == '2':
            username = input("\nEnter username to authenticate: ").strip()
            if username:
                auth.authenticate_user(username)
            else:
                print("❌ Username cannot be empty!")
        
        elif choice == '3':
            users = auth.list_users()
            if users:
                print(f"\n{'='*70}")
                print("📋 REGISTERED USERS")
                print(f"{'='*70}")
                for user in users:
                    print(f"\n👤 {user['username']}")
                    print(f"   Registered: {user['registered'][:10]}")
                    print(f"   Consistency: {user['consistency']*100:.1f}%")
                    print(f"   Auth Attempts: {user['auth_attempts']} ({user['successful_auths']} successful)")
            else:
                print("\n📋 No users registered yet.")
        
        elif choice == '4':
            username = input("\nEnter username to view stats: ").strip()
            if username:
                auth.show_statistics(username)
            else:
                print("❌ Username cannot be empty!")
        
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()