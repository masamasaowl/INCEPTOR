"""
Voice Feature Debugger
Shows exactly what features are being extracted and compared
"""

import numpy as np
import json
import os
from scipy.spatial.distance import cosine
import librosa
import soundfile as sf

class VoiceDebugger:
    def __init__(self, data_dir="voice_data"):
        self.data_dir = data_dir
        self.sample_rate = 16000
    
    def load_profile(self, username):
        """Load user profile"""
        profile_path = os.path.join(self.data_dir, f"{username}_profile.json")
        if not os.path.exists(profile_path):
            return None
        
        with open(profile_path, 'r') as f:
            return json.load(f)
    
    def load_audio(self, username, sample_num=1):
        """Load audio file"""
        audio_path = os.path.join(self.data_dir, f"{username}_sample_{sample_num}.wav")
        if not os.path.exists(audio_path):
            return None
        
        audio, sr = sf.read(audio_path)
        return audio
    
    def quick_extract_features(self, audio):
        """Quick feature extraction for comparison"""
        # Voice activity detection
        rms = librosa.feature.rms(y=audio)[0]
        voice_frames = rms > 0.02
        
        if np.any(voice_frames):
            # Simple extraction
            voice_audio = audio[voice_frames.repeat(512)[:len(audio)]]
        else:
            voice_audio = audio
        
        # Basic MFCCs
        mfccs = librosa.feature.mfcc(y=voice_audio, sr=self.sample_rate, n_mfcc=20)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Normalize
        mfcc_mean = (mfcc_mean - np.min(mfcc_mean)) / (np.max(mfcc_mean) - np.min(mfcc_mean) + 1e-8)
        
        return mfcc_mean
    
    def compare_samples(self, username):
        """Compare all samples for a user"""
        print(f"\n{'='*70}")
        print(f"🔍 DEBUGGING USER: {username}")
        print(f"{'='*70}")
        
        profile = self.load_profile(username)
        if not profile:
            print(f"❌ User '{username}' not found!")
            return
        
        print(f"\nRegistered: {profile['registered_at'][:19]}")
        print(f"Feature count: {profile.get('feature_count', 'Unknown')}")
        
        # Load all samples
        samples = []
        for i in range(1, 4):
            audio = self.load_audio(username, i)
            if audio is not None:
                samples.append(audio)
        
        if not samples:
            print("❌ No audio samples found!")
            return
        
        print(f"\n📊 Analyzing {len(samples)} registered samples...")
        
        # Extract features from each sample
        features_list = []
        for i, audio in enumerate(samples, 1):
            features = self.quick_extract_features(audio)
            features_list.append(features)
            
            # Audio stats
            rms = np.sqrt(np.mean(audio**2))
            max_amp = np.max(np.abs(audio))
            
            print(f"\nSample {i}:")
            print(f"  Duration: {len(audio)/self.sample_rate:.2f}s")
            print(f"  RMS Level: {rms:.4f}")
            print(f"  Peak Amplitude: {max_amp:.4f}")
        
        # Compare samples to each other
        print(f"\n{'─'*70}")
        print("🔬 INTER-SAMPLE COMPARISON")
        print(f"{'─'*70}")
        
        for i in range(len(features_list)):
            for j in range(i+1, len(features_list)):
                similarity = 1 - cosine(features_list[i], features_list[j])
                print(f"Sample {i+1} vs Sample {j+1}: {similarity*100:.2f}%")
        
        # Compare to stored profile
        if 'features' in profile:
            stored_features = np.array(profile['features'])
            
            print(f"\n{'─'*70}")
            print("📋 COMPARISON TO STORED PROFILE")
            print(f"{'─'*70}")
            
            for i, features in enumerate(features_list, 1):
                # Ensure same length
                min_len = min(len(features), len(stored_features))
                similarity = 1 - cosine(features[:min_len], stored_features[:min_len])
                print(f"Sample {i} vs Profile: {similarity*100:.2f}%")
    
    def compare_two_users(self, user1, user2):
        """Compare two different users"""
        print(f"\n{'='*70}")
        print(f"🔬 COMPARING TWO USERS")
        print(f"{'='*70}")
        
        profile1 = self.load_profile(user1)
        profile2 = self.load_profile(user2)
        
        if not profile1:
            print(f"❌ User '{user1}' not found!")
            return
        
        if not profile2:
            print(f"❌ User '{user2}' not found!")
            return
        
        features1 = np.array(profile1['features'])
        features2 = np.array(profile2['features'])
        
        # Ensure same length
        min_len = min(len(features1), len(features2))
        
        similarity = 1 - cosine(features1[:min_len], features2[:min_len])
        
        print(f"\nUser 1: {user1}")
        print(f"User 2: {user2}")
        print(f"\n{'─'*70}")
        print(f"Similarity: {similarity*100:.2f}%")
        print(f"{'─'*70}")
        
        if similarity > 85:
            print("🔴 WARNING: Very similar! Same person or problem with features")
        elif similarity > 70:
            print("🟡 Moderately similar - might be related voices")
        else:
            print("🟢 Different people - system working correctly!")
        
        # Feature-by-feature comparison
        print(f"\n📊 Feature Statistics:")
        diff = np.abs(features1[:min_len] - features2[:min_len])
        print(f"Max difference: {np.max(diff):.4f}")
        print(f"Avg difference: {np.mean(diff):.4f}")
        print(f"Features >0.3 different: {np.sum(diff > 0.3)}/{min_len}")


def main():
    print("="*70)
    print("🔍 VOICE AUTHENTICATION DEBUGGER")
    print("See exactly what's happening with your voice features")
    print("="*70)
    
    debugger = VoiceDebugger()
    
    while True:
        print("\n" + "="*70)
        print("DEBUG OPTIONS:")
        print("1. Analyze a user's registered samples")
        print("2. Compare two users")
        print("3. Exit")
        print("="*70)
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            username = input("\nEnter username to analyze: ").strip()
            if username:
                debugger.compare_samples(username)
            else:
                print("❌ Username required!")
        
        elif choice == '2':
            user1 = input("\nEnter first username: ").strip()
            user2 = input("Enter second username: ").strip()
            if user1 and user2:
                debugger.compare_two_users(user1, user2)
            else:
                print("❌ Both usernames required!")
        
        elif choice == '3':
            print("\n👋 Done debugging!")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()