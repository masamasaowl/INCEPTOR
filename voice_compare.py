"""
Voice Comparison Demo
Perfect for hackathon demonstrations - compare multiple voices!
"""

import numpy as np
import json
import os
from scipy.spatial.distance import cosine

class VoiceComparer:
    def __init__(self, data_dir="voice_data"):
        self.data_dir = data_dir
    
    def load_user_profile(self, username):
        """Load a user's voice profile"""
        profile_path = os.path.join(self.data_dir, f"{username}_profile.json")
        if not os.path.exists(profile_path):
            return None
        
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        return profile
    
    def compare_users(self, user1, user2):
        """Compare two users' voice profiles"""
        profile1 = self.load_user_profile(user1)
        profile2 = self.load_user_profile(user2)
        
        if not profile1:
            print(f"❌ User '{user1}' not found!")
            return
        
        if not profile2:
            print(f"❌ User '{user2}' not found!")
            return
        
        features1 = np.array(profile1['features'])
        features2 = np.array(profile2['features'])
        
        # Calculate similarity
        distance = cosine(features1, features2)
        similarity = (1 - distance) * 100
        
        print(f"\n{'='*60}")
        print(f"🔬 VOICE COMPARISON ANALYSIS")
        print(f"{'='*60}")
        print(f"User 1: {user1}")
        print(f"User 2: {user2}")
        print(f"─"*60)
        print(f"Similarity Score: {similarity:.2f}%")
        print(f"─"*60)
        
        if similarity > 85:
            print("🔴 HIGH SIMILARITY - These voices are very similar!")
            print("   (Same person or very similar voices)")
        elif similarity > 70:
            print("🟡 MODERATE SIMILARITY - Some common characteristics")
            print("   (Could be related voices or similar vocal ranges)")
        else:
            print("🟢 LOW SIMILARITY - These voices are clearly different!")
            print("   (Different people)")
        
        print(f"{'='*60}")
        
        # Feature breakdown
        print(f"\n📊 Feature Differences:")
        feature_diff = np.abs(features1 - features2)
        
        print(f"Max difference: {np.max(feature_diff):.4f}")
        print(f"Average difference: {np.mean(feature_diff):.4f}")
        print(f"Min difference: {np.min(feature_diff):.4f}")
    
    def compare_all_users(self):
        """Create a comparison matrix of all users"""
        # Get all users
        users = []
        for file in os.listdir(self.data_dir):
            if file.endswith('_profile.json'):
                username = file.replace('_profile.json', '')
                users.append(username)
        
        if len(users) < 2:
            print("❌ Need at least 2 registered users to compare!")
            return
        
        print(f"\n{'='*60}")
        print(f"👥 VOICE SIMILARITY MATRIX")
        print(f"{'='*60}\n")
        
        # Load all profiles
        profiles = {}
        for user in users:
            profile = self.load_user_profile(user)
            if profile:
                profiles[user] = np.array(profile['features'])
        
        # Print header
        print(f"{'':15}", end='')
        for user in users:
            print(f"{user[:12]:>12}", end=' ')
        print("\n" + "─"*60)
        
        # Print matrix
        for user1 in users:
            print(f"{user1[:15]:15}", end='')
            for user2 in users:
                if user1 == user2:
                    print(f"{'---':>12}", end=' ')
                else:
                    distance = cosine(profiles[user1], profiles[user2])
                    similarity = (1 - distance) * 100
                    print(f"{similarity:>11.1f}%", end=' ')
            print()
        
        print(f"\n{'='*60}")
        print("Interpretation:")
        print("  > 85%: Very similar (likely same person)")
        print("  70-85%: Moderately similar")
        print("  < 70%: Clearly different people")


def main():
    print("="*60)
    print("🔬 VOICE COMPARISON TOOL")
    print("Perfect for testing your authentication system!")
    print("="*60)
    
    comparer = VoiceComparer()
    
    while True:
        print("\n" + "="*60)
        print("OPTIONS:")
        print("1. Compare two specific users")
        print("2. Compare all users (similarity matrix)")
        print("3. Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            user1 = input("\nEnter first username: ").strip()
            user2 = input("Enter second username: ").strip()
            if user1 and user2:
                comparer.compare_users(user1, user2)
            else:
                print("❌ Both usernames are required!")
        
        elif choice == '2':
            comparer.compare_all_users()
        
        elif choice == '3':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-3.")


if __name__ == "__main__":
    main()