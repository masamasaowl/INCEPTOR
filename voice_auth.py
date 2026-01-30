"""
Voice Authentication System
A simple speaker verification system for hackathon demo
"""

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import os
import json
from scipy.spatial.distance import cosine
from datetime import datetime

class VoiceAuthenticator:
    def __init__(self, data_dir="voice_data"):
        """Initialize the voice authentication system"""
        self.data_dir = data_dir
        self.sample_rate = 16000  # 16kHz is standard for speech
        self.duration = 3  # Record for 3 seconds
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"✓ Created data directory: {self.data_dir}")
    
    def record_audio(self):
        """Record audio from microphone"""
        print(f"\n🎤 Recording for {self.duration} seconds...")
        print("Say: 'Hello' clearly into your microphone")
        print("Recording starts in: 3... 2... 1...")
        
        # Record audio
        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        print("✓ Recording complete!")
        return audio.flatten()
    
    def extract_features(self, audio):
        """Extract MFCC features from audio signal"""
        # Extract MFCCs (Mel-frequency cepstral coefficients)
        # These capture the unique characteristics of a person's voice
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13  # 13 coefficients is standard
        )
        
        # Take the mean across time to get a feature vector
        mfcc_mean = np.mean(mfccs, axis=1)
        
        return mfcc_mean
    
    def register_user(self, username):
        """Register a new user with their voice"""
        print(f"\n{'='*50}")
        print(f"REGISTRATION MODE - User: {username}")
        print(f"{'='*50}")
        
        # Record 3 samples for better accuracy
        feature_vectors = []
        
        for i in range(3):
            print(f"\n--- Sample {i+1}/3 ---")
            audio = self.record_audio()
            
            # Save the audio file
            audio_path = os.path.join(
                self.data_dir,
                f"{username}_sample_{i+1}.wav"
            )
            sf.write(audio_path, audio, self.sample_rate)
            
            # Extract features
            features = self.extract_features(audio)
            feature_vectors.append(features)
            
            if i < 2:
                input("\nPress Enter when ready for next sample...")
        
        # Average the feature vectors for robustness
        avg_features = np.mean(feature_vectors, axis=0)
        
        # Save the user profile
        user_profile = {
            'username': username,
            'features': avg_features.tolist(),
            'registered_at': datetime.now().isoformat()
        }
        
        profile_path = os.path.join(self.data_dir, f"{username}_profile.json")
        with open(profile_path, 'w') as f:
            json.dump(user_profile, f, indent=2)
        
        print(f"\n✅ Registration successful!")
        print(f"Profile saved to: {profile_path}")
        return True
    
    def authenticate_user(self, username):
        """Authenticate a user by their voice"""
        print(f"\n{'='*50}")
        print(f"AUTHENTICATION MODE - User: {username}")
        print(f"{'='*50}")
        
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
        audio = self.record_audio()
        
        # Extract features
        test_features = self.extract_features(audio)
        
        # Calculate similarity using cosine distance
        # Lower distance = more similar (0 = identical, 2 = opposite)
        distance = cosine(stored_features, test_features)
        similarity = 1 - distance  # Convert to similarity (higher = better)
        
        # Threshold for authentication (you can adjust this)
        threshold = 0.85  # 85% similarity required
        
        print(f"\n📊 Analysis:")
        print(f"Similarity Score: {similarity*100:.2f}%")
        print(f"Threshold: {threshold*100:.2f}%")
        
        if similarity >= threshold:
            print(f"\n✅ AUTHENTICATION SUCCESSFUL!")
            print(f"Welcome back, {username}!")
            return True
        else:
            print(f"\n❌ AUTHENTICATION FAILED!")
            print(f"Voice does not match registered profile.")
            return False
    
    def list_users(self):
        """List all registered users"""
        users = []
        for file in os.listdir(self.data_dir):
            if file.endswith('_profile.json'):
                username = file.replace('_profile.json', '')
                users.append(username)
        return users


def main():
    """Main function to run the voice authentication system"""
    print("="*50)
    print("🎙️  VOICE AUTHENTICATION SYSTEM")
    print("="*50)
    
    auth = VoiceAuthenticator()
    
    while True:
        print("\n" + "="*50)
        print("MENU:")
        print("1. Register new user")
        print("2. Authenticate user")
        print("3. List registered users")
        print("4. Exit")
        print("="*50)
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
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
                print("\n📋 Registered Users:")
                for i, user in enumerate(users, 1):
                    print(f"{i}. {user}")
            else:
                print("\n📋 No users registered yet.")
        
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()