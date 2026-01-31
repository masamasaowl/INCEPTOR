"""
Voice Authentication System V4
STRICTER AUTHENTICATION - Better discrimination between speakers
"""

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import os
import json
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from datetime import datetime
import time

class VoiceAuthenticatorV4:
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
        frame_length = 512
        hop_length = 256
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        voice_frames = rms > energy_threshold
        
        if not np.any(voice_frames):
            print("⚠️  WARNING: No speech detected! Audio might be too quiet.")
            return audio
        
        voice_samples = np.zeros(len(audio), dtype=bool)
        for i, is_voice in enumerate(voice_frames):
            start = i * hop_length
            end = min(start + frame_length, len(audio))
            if is_voice:
                voice_samples[start:end] = True
        
        voice_audio = audio[voice_samples]
        
        print(f"🎤 Voice Activity: {len(voice_audio)/len(audio)*100:.1f}% of recording contains speech")
        
        return voice_audio if len(voice_audio) > self.sample_rate * 0.5 else audio
    
    def extract_features(self, audio):
        """Extract comprehensive voice features with better discrimination"""
        
        speech_audio = self.detect_voice_activity(audio)
        
        # 1. MFCCs - MORE coefficients for better detail
        mfccs = librosa.feature.mfcc(
            y=speech_audio,
            sr=self.sample_rate,
            n_mfcc=25  # Increased from 20
        )
        
        # 2. Delta and Delta-Delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # 3. Pitch features - MORE DETAILED
        pitches, magnitudes = librosa.piptrack(y=speech_audio, sr=self.sample_rate)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        # 4. Formants estimation (vocal tract resonances) - UNIQUE PER PERSON
        # Using spectral peaks as proxy for formants
        spectral_centroids = librosa.feature.spectral_centroid(y=speech_audio, sr=self.sample_rate)
        
        # 5. More spectral features
        spectral_rolloff = librosa.feature.spectral_rolloff(y=speech_audio, sr=self.sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=speech_audio, sr=self.sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=speech_audio, sr=self.sample_rate)
        spectral_flatness = librosa.feature.spectral_flatness(y=speech_audio)
        
        # 6. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(speech_audio)
        
        # 7. Chroma features - harmonic content
        chroma = librosa.feature.chroma_stft(y=speech_audio, sr=self.sample_rate)
        
        # 8. Energy features
        rms_energy = librosa.feature.rms(y=speech_audio)
        
        # Create comprehensive feature vector with BETTER STATISTICS
        features = np.concatenate([
            # MFCCs - mean and std
            np.mean(mfccs, axis=1),           # 25 features
            np.std(mfccs, axis=1),            # 25 features
            np.median(mfccs, axis=1),         # 25 features - NEW
            
            # Delta MFCCs
            np.mean(delta_mfccs, axis=1),     # 25 features
            np.std(delta_mfccs, axis=1),      # 25 features
            
            # Delta-Delta MFCCs  
            np.mean(delta2_mfccs, axis=1),    # 25 features - NEW
            
            # Pitch - MORE STATS
            [np.mean(pitch_values) if pitch_values else 0],
            [np.std(pitch_values) if pitch_values else 0],
            [np.median(pitch_values) if pitch_values else 0],  # NEW
            [np.max(pitch_values) if pitch_values else 0],     # NEW
            [np.min(pitch_values) if pitch_values else 0],     # NEW
            
            # Spectral features - MORE DETAILED
            [np.mean(spectral_centroids)],
            [np.std(spectral_centroids)],
            [np.median(spectral_centroids)],   # NEW
            [np.mean(spectral_rolloff)],
            [np.std(spectral_rolloff)],
            [np.mean(spectral_bandwidth)],
            [np.std(spectral_bandwidth)],
            
            # Spectral contrast - NEW
            np.mean(spectral_contrast, axis=1),  # 7 features
            
            # Spectral flatness
            [np.mean(spectral_flatness)],
            [np.std(spectral_flatness)],
            
            # Zero crossing
            [np.mean(zcr)],
            [np.std(zcr)],
            
            # Chroma
            np.mean(chroma, axis=1),          # 12 features
            
            # Energy
            [np.mean(rms_energy)],
            [np.std(rms_energy)],
        ])
        
        # BETTER NORMALIZATION: Per-feature normalization (not global)
        # This preserves the relative differences between features
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Min-max normalization per feature
        feature_min = np.min(features)
        feature_max = np.max(features)
        if feature_max > feature_min:
            features = (features - feature_min) / (feature_max - feature_min)
        
        print(f"📊 Extracted {len(features)} discriminative features from voice")
        
        return features
    
    def record_audio_with_countdown(self):
        """Record audio with visual countdown"""
        print(f"\n🎤 Recording for {self.duration} seconds...")
        print("Say: 'Hello, this is my voice' clearly")
        
        for i in range(3, 0, -1):
            print(f"Starting in {i}...", end='\r')
            time.sleep(1)
        
        print("\n🔴 RECORDING NOW!          ")
        
        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        
        for i in range(self.duration):
            time.sleep(1)
            print(f"{'█' * (i+1)}{'░' * (self.duration-i-1)} {i+1}/{self.duration}s")
        
        sd.wait()
        
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
                
                if self.playback_audio(audio):
                    audio_path = os.path.join(
                        self.data_dir,
                        f"{username}_sample_{i+1}.wav"
                    )
                    sf.write(audio_path, audio, self.sample_rate)
                    
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
        
        if avg_consistency < 0.80:  # Raised from 0.75
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
        """Authenticate a user with STRICTER verification"""
        print(f"\n{'='*70}")
        print(f"AUTHENTICATION MODE - User: {username}")
        print(f"{'='*70}")
        
        profile_path = os.path.join(self.data_dir, f"{username}_profile.json")
        if not os.path.exists(profile_path):
            print(f"❌ User '{username}' not found. Please register first.")
            return False
        
        with open(profile_path, 'r') as f:
            user_profile = json.load(f)
        
        stored_features = np.array(user_profile['features'])
        
        audio = self.record_audio_with_countdown()
        
        response = input("\n🔊 Hear your recording? (y/n): ").strip().lower()
        if response == 'y':
            print("Playing back...")
            sd.play(audio, self.sample_rate)
            sd.wait()
        
        test_features = self.extract_features(audio)
        
        # === MULTI-METRIC VERIFICATION ===
        
        # 1. Cosine similarity
        cosine_dist = cosine(stored_features, test_features)
        cosine_sim = 1 - cosine_dist
        
        # 2. Euclidean distance (normalized)
        euclidean_dist = euclidean(stored_features, test_features)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # 3. Pearson correlation - NEW metric
        try:
            pearson_corr, _ = pearsonr(stored_features, test_features)
            pearson_sim = (pearson_corr + 1) / 2  # Convert -1,1 range to 0,1
        except:
            pearson_sim = 0.5
        
        # === STRICTER COMBINED SCORE ===
        # More balanced weighting - euclidean gets more weight now
        similarity = (0.4 * cosine_sim + 0.4 * euclidean_sim + 0.2 * pearson_sim)
        
        # === MUCH STRICTER THRESHOLD ===
        base_threshold = 0.9  # Increased from 0.75 to 0.88
        
        # Adaptive threshold based on registration consistency
        consistency_bonus = max(0, (user_profile.get('consistency', 0.9) - 0.90) * 0.05)
        threshold = base_threshold + consistency_bonus
        
        # === FEATURE ANALYSIS ===
        feature_diff = np.abs(stored_features - test_features)
        max_diff = np.max(feature_diff)
        avg_diff = np.mean(feature_diff)
        significant_diffs = np.sum(feature_diff > 0.2)  # Lowered threshold from 0.3 to 0.2
        
        # === ADDITIONAL SECURITY CHECKS ===
        
        # Check 1: Individual metrics must also pass minimum thresholds
        cosine_pass = cosine_sim >= 0.99  # Cosine must be at least 90%
        euclidean_pass = euclidean_sim >= 0.97  # Euclidean must be at least 50%
        pearson_pass = pearson_sim >= 0.99  # Pearson must be at least 75%
        
        # Check 2: Not too many features can be drastically different
        max_allowed_diffs = len(stored_features) * 0.15  # Only 15% of features can differ significantly
        feature_check_pass = significant_diffs <= max_allowed_diffs
        
        # Check 3: Average difference must be small
        avg_diff_pass = avg_diff < 0.12  # Average difference must be less than 0.12
        
        all_checks_passed = all([
            cosine_pass,
            euclidean_pass, 
            pearson_pass,
            feature_check_pass,
            avg_diff_pass
        ])
        
        print(f"\n{'='*70}")
        print(f"📊 AUTHENTICATION ANALYSIS")
        print(f"{'='*70}")
        print(f"Combined Similarity:     {similarity*100:.2f}%")
        print(f"  - Cosine Similarity:   {cosine_sim*100:.2f}% {'✅' if cosine_pass else '❌'} (need ≥99%)")
        print(f"  - Euclidean Similarity: {euclidean_sim*100:.2f}% {'✅' if euclidean_pass else '❌'} (need ≥97%)")
        print(f"  - Pearson Correlation: {pearson_sim*100:.2f}% {'✅' if pearson_pass else '❌'} (need ≥99%)")
        print(f"Required Threshold:      {threshold*100:.2f}%")
        print(f"─"*70)
        print(f"Security Checks:")
        print(f"  - Features Changed:    {significant_diffs}/{len(stored_features)} {'✅' if feature_check_pass else '❌'} (max {int(max_allowed_diffs)})")
        print(f"  - Avg Difference:      {avg_diff:.4f} {'✅' if avg_diff_pass else '❌'} (max 0.12)")
        print(f"  - Max Difference:      {max_diff:.4f}")
        print(f"{'='*70}")
        
        # === FINAL DECISION ===
        # MUST pass both the similarity threshold AND all security checks
        authentication_success = (similarity >= threshold) and all_checks_passed
        
        # Log the attempt
        log_entry = {
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'similarity': float(similarity),
            'cosine_sim': float(cosine_sim),
            'euclidean_sim': float(euclidean_sim),
            'pearson_sim': float(pearson_sim),
            'threshold': float(threshold),
            'success': bool(authentication_success),
            'cosine_pass': bool(cosine_pass),
            'euclidean_pass': bool(euclidean_pass),
            'pearson_pass': bool(pearson_pass),
            'feature_check_pass': bool(feature_check_pass),
            'avg_diff_pass': bool(avg_diff_pass),
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
        
        # Decision output
        if authentication_success:
            print(f"\n✅ AUTHENTICATION SUCCESSFUL!")
            print(f"Welcome back, {username}! 🎉")
            return True
        else:
            print(f"\n❌ AUTHENTICATION FAILED!")
            
            if similarity < threshold:
                print(f"❌ Similarity too low: {similarity*100:.2f}% < {threshold*100:.2f}%")
            
            if not all_checks_passed:
                print(f"❌ Failed security checks:")
                if not cosine_pass:
                    print(f"   • Cosine similarity too low ({cosine_sim*100:.2f}% < 90%)")
                if not euclidean_pass:
                    print(f"   • Euclidean similarity too low ({euclidean_sim*100:.2f}% < 50%)")
                if not pearson_pass:
                    print(f"   • Pearson correlation too low ({pearson_sim*100:.2f}% < 75%)")
                if not feature_check_pass:
                    print(f"   • Too many features changed ({significant_diffs} > {int(max_allowed_diffs)})")
                if not avg_diff_pass:
                    print(f"   • Average difference too high ({avg_diff:.4f} > 0.12)")
            
            print("\n💡 This could indicate:")
            print("   - Different person speaking")
            print("   - Very different recording environment")
            print("   - Background noise or audio quality issues")
            
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
        
        print(f"\n📝 Recent Attempts (last 10):")
        for log in logs[-10:]:
            timestamp = datetime.fromisoformat(log['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            status = "✅ Success" if log['success'] else "❌ Failed"
            
            # Show which checks failed
            failed_checks = []
            if not log.get('cosine_pass', True):
                failed_checks.append("Cosine")
            if not log.get('euclidean_pass', True):
                failed_checks.append("Euclidean")
            if not log.get('pearson_pass', True):
                failed_checks.append("Pearson")
            if not log.get('feature_check_pass', True):
                failed_checks.append("Features")
            if not log.get('avg_diff_pass', True):
                failed_checks.append("AvgDiff")
            
            failed_str = f" (Failed: {', '.join(failed_checks)})" if failed_checks else ""
            
            print(f"{timestamp} | {status} | {log['similarity']*100:.2f}%{failed_str}")


def main():
    """Main function"""
    print("="*70)
    print("🎙️  VOICE AUTHENTICATION SYSTEM V4")
    print("🔒 STRICTER SECURITY - Better Speaker Discrimination")
    print("="*70)
    
    auth = VoiceAuthenticatorV4()
    
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