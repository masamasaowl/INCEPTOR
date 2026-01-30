"""
Voice Authentication System V2
Enhanced with playback, visual feedback, and better validation
"""

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import os
import json
from scipy.spatial.distance import cosine
from datetime import datetime
import time

class VoiceAuthenticatorV2:
    def __init__(self, data_dir="voice_data"):
        """Initialize the enhanced voice authentication system"""
        self.data_dir = data_dir
        self.sample_rate = 16000
        self.duration = 3
        
        # Create data directory
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"✓ Created data directory: {self.data_dir}")
        
        # Create logs directory
        self.logs_dir = os.path.join(self.data_dir, "logs")
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
    
    def check_audio_level(self, duration=2):
        """Check microphone audio level before recording"""
        print("\n🎤 Checking microphone level...")
        print("Make some noise or say something!")
        
        # Record a short sample
        test_audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        # Calculate RMS (root mean square) - measure of audio level
        rms = np.sqrt(np.mean(test_audio**2))
        max_amplitude = np.max(np.abs(test_audio))
        
        print(f"\n📊 Audio Level Report:")
        print(f"RMS Level: {rms:.4f}")
        print(f"Peak Amplitude: {max_amplitude:.4f}")
        
        if max_amplitude < 0.01:
            print("⚠️  WARNING: Very low audio! Check if mic is working.")
            print("Try speaking louder or check microphone settings.")
            return False
        elif max_amplitude > 0.9:
            print("⚠️  WARNING: Audio too loud! May cause distortion.")
            print("Try speaking softer or adjust microphone gain.")
            return False
        else:
            print("✅ Audio level is good!")
            return True
    
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
        print("✓ Recording complete!")
        
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
    
    def extract_features(self, audio):
        """Extract MFCC features from audio signal"""
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13
        )
        
        # Extract additional features for better accuracy
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        # Combine features
        mfcc_mean = np.mean(mfccs, axis=1)
        centroid_mean = np.mean(spectral_centroid)
        zcr_mean = np.mean(zero_crossing_rate)
        
        # Create feature vector
        features = np.concatenate([
            mfcc_mean,
            [centroid_mean, zcr_mean]
        ])
        
        return features
    
    def detect_liveness(self, audio):
        """Basic liveness detection - check if it's a real voice vs recording"""
        # Calculate some audio properties
        rms = np.sqrt(np.mean(audio**2))
        
        # Check for unnatural patterns (very basic)
        # Real voices have natural variations
        segments = np.array_split(audio, 10)
        segment_rms = [np.sqrt(np.mean(seg**2)) for seg in segments]
        rms_variance = np.var(segment_rms)
        
        # Real voices should have some variance
        if rms_variance < 0.0001:
            return False, "Audio seems too uniform (possible playback)"
        
        if rms < 0.005:
            return False, "Audio too quiet (possible issue)"
        
        return True, "Liveness check passed"
    
    def register_user(self, username):
        """Register a new user with their voice"""
        print(f"\n{'='*60}")
        print(f"REGISTRATION MODE - User: {username}")
        print(f"{'='*60}")
        
        # Check microphone first
        if not self.check_audio_level():
            retry = input("\nContinue anyway? (y/n): ").strip().lower()
            if retry != 'y':
                return False
        
        feature_vectors = []
        recorded_samples = []
        
        for i in range(3):
            while True:
                print(f"\n{'─'*60}")
                print(f"📝 Sample {i+1}/3")
                print(f"{'─'*60}")
                
                audio = self.record_audio_with_countdown()
                
                # Liveness check
                is_live, message = self.detect_liveness(audio)
                print(f"\n🔍 Liveness Check: {message}")
                
                # Playback and confirm
                if self.playback_audio(audio):
                    recorded_samples.append(audio)
                    
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
        
        # Calculate consistency score
        consistency_scores = []
        for features in feature_vectors:
            similarity = 1 - cosine(avg_features, features)
            consistency_scores.append(similarity)
        
        avg_consistency = np.mean(consistency_scores)
        
        print(f"\n📊 Registration Analysis:")
        print(f"Consistency Score: {avg_consistency*100:.2f}%")
        
        if avg_consistency < 0.80:
            print("⚠️  WARNING: Low consistency between samples!")
            print("Your recordings might be too different.")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return False
        
        # Save the user profile
        user_profile = {
            'username': username,
            'features': avg_features.tolist(),
            'consistency': float(avg_consistency),
            'registered_at': datetime.now().isoformat(),
            'sample_count': 3
        }
        
        profile_path = os.path.join(self.data_dir, f"{username}_profile.json")
        with open(profile_path, 'w') as f:
            json.dump(user_profile, f, indent=2)
        
        print(f"\n✅ Registration successful!")
        print(f"📁 Profile saved to: {profile_path}")
        return True
    
    def authenticate_user(self, username):
        """Authenticate a user by their voice"""
        print(f"\n{'='*60}")
        print(f"AUTHENTICATION MODE - User: {username}")
        print(f"{'='*60}")
        
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
        
        # Liveness check
        is_live, liveness_message = self.detect_liveness(audio)
        print(f"\n🔍 Liveness Check: {liveness_message}")
        
        if not is_live:
            print("⚠️  Warning: Liveness check failed!")
        
        # Playback option
        response = input("\n🔊 Hear your recording? (y/n): ").strip().lower()
        if response == 'y':
            print("Playing back...")
            sd.play(audio, self.sample_rate)
            sd.wait()
        
        # Extract features
        test_features = self.extract_features(audio)
        
        # Calculate similarity
        distance = cosine(stored_features, test_features)
        similarity = 1 - distance
        
        # Threshold for authentication
        threshold = 0.85
        
        # Calculate additional metrics
        feature_diff = np.abs(stored_features - test_features)
        max_diff = np.max(feature_diff)
        avg_diff = np.mean(feature_diff)
        
        print(f"\n{'='*60}")
        print(f"📊 AUTHENTICATION ANALYSIS")
        print(f"{'='*60}")
        print(f"Similarity Score:     {similarity*100:.2f}%")
        print(f"Required Threshold:   {threshold*100:.2f}%")
        print(f"Max Feature Diff:     {max_diff:.4f}")
        print(f"Avg Feature Diff:     {avg_diff:.4f}")
        print(f"Liveness Check:       {'✅ Passed' if is_live else '❌ Failed'}")
        print(f"{'='*60}")
        
        # Log the attempt
        log_entry = {
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'similarity': float(similarity),
            'threshold': threshold,
            'success': similarity >= threshold and is_live,
            'liveness_passed': is_live
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
            if is_live:
                print(f"\n✅ AUTHENTICATION SUCCESSFUL!")
                print(f"Welcome back, {username}! 🎉")
                return True
            else:
                print(f"\n⚠️  AUTHENTICATION WARNING!")
                print(f"Voice matched but liveness check failed.")
                return False
        else:
            print(f"\n❌ AUTHENTICATION FAILED!")
            print(f"Voice does not match registered profile.")
            print(f"Similarity: {similarity*100:.2f}% < Required: {threshold*100:.2f}%")
            return False
    
    def list_users(self):
        """List all registered users with stats"""
        users = []
        for file in os.listdir(self.data_dir):
            if file.endswith('_profile.json'):
                username = file.replace('_profile.json', '')
                
                # Load profile for stats
                profile_path = os.path.join(self.data_dir, file)
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                
                # Load auth log if exists
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
        max_similarity = np.max(similarities)
        min_similarity = np.min(similarities)
        
        print(f"\n{'='*60}")
        print(f"📊 AUTHENTICATION STATISTICS - {username}")
        print(f"{'='*60}")
        print(f"Total Attempts:       {total}")
        print(f"✅ Successful:        {successful} ({successful/total*100:.1f}%)")
        print(f"❌ Failed:            {failed} ({failed/total*100:.1f}%)")
        print(f"─"*60)
        print(f"Average Similarity:   {avg_similarity*100:.2f}%")
        print(f"Best Match:           {max_similarity*100:.2f}%")
        print(f"Worst Match:          {min_similarity*100:.2f}%")
        print(f"{'='*60}")
        
        print(f"\n📝 Recent Attempts (last 5):")
        for log in logs[-5:]:
            timestamp = datetime.fromisoformat(log['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            status = "✅ Success" if log['success'] else "❌ Failed"
            print(f"{timestamp} | {status} | {log['similarity']*100:.2f}%")


def main():
    """Main function"""
    print("="*60)
    print("🎙️  VOICE AUTHENTICATION SYSTEM V2")
    print("Enhanced with Playback & Better Validation")
    print("="*60)
    
    auth = VoiceAuthenticatorV2()
    
    while True:
        print("\n" + "="*60)
        print("MENU:")
        print("1. Register new user")
        print("2. Authenticate user")
        print("3. List registered users")
        print("4. View user statistics")
        print("5. Test microphone")
        print("6. Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
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
                print(f"\n{'='*60}")
                print("📋 REGISTERED USERS")
                print(f"{'='*60}")
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
            auth.check_audio_level(duration=3)
        
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    main()