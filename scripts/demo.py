#!/usr/bin/env python3
"""
Quick Demo Script for Voice Authentication System
Demonstrates the core functionality with a simple test scenario

Note: This demo requires you to have audio files ready.
For a complete test, prepare 4 audio files:
- voice1.wav, voice2.wav, voice3.wav (enrollment samples)
- test_auth.wav (authentication test)
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings, ensure_directories
from core.service import voice_auth_service
from database.manager import db_manager


async def demo():
    """
    Run a complete demo of the voice authentication system
    """
    print("=" * 70)
    print("  🎙️  VOICE AUTHENTICATION SYSTEM - DEMO")
    print("=" * 70)
    print()
    
    # Initialize system
    print("📋 Step 1: Initializing system...")
    ensure_directories()
    await db_manager.initialize_database()
    print("✅ System initialized\n")
    
    # Demo user
    username = "demo_user"
    passphrase = "hello voice authentication"
    
    # Check if we have audio files
    audio_dir = Path("demo_audio")
    if not audio_dir.exists():
        print("⚠️  No demo audio files found!")
        print()
        print("To run this demo, you need to create audio files:")
        print(f"1. Create a directory: {audio_dir}")
        print("2. Add 3 enrollment samples: voice1.wav, voice2.wav, voice3.wav")
        print("3. Add 1 test sample: test_auth.wav")
        print()
        print("All files should contain you saying: 'hello voice authentication'")
        print()
        print("Alternative: Use the CLI to test with your own audio files:")
        print("  python cli.py register --username test_user")
        print("  python cli.py enroll --username test_user --audio your_audio.wav --passphrase 'your phrase'")
        print()
        return
    
    enrollment_files = [
        audio_dir / "voice1.wav",
        audio_dir / "voice2.wav",
        audio_dir / "voice3.wav"
    ]
    
    auth_file = audio_dir / "test_auth.wav"
    
    missing_files = [f for f in enrollment_files + [auth_file] if not f.exists()]
    if missing_files:
        print("⚠️  Missing audio files:")
        for f in missing_files:
            print(f"   - {f}")
        print()
        print("Please create these files with your voice saying:")
        print(f"   '{passphrase}'")
        print()
        return
    
    try:
        # Step 2: Register user
        print(f"📋 Step 2: Registering user '{username}'...")
        result = await voice_auth_service.register_user(username, "demo@example.com")
        if result['success']:
            print(f"✅ {result['message']}")
        else:
            # User might already exist, try to delete and re-register
            await voice_auth_service.delete_user(username)
            result = await voice_auth_service.register_user(username, "demo@example.com")
            print(f"✅ {result['message']}")
        print()
        
        # Step 3: Enroll voice samples
        print(f"📋 Step 3: Enrolling voice samples...")
        for i, audio_file in enumerate(enrollment_files, 1):
            print(f"   Processing sample {i}/3: {audio_file.name}...")
            result = await voice_auth_service.add_voice_sample(
                username=username,
                audio_file_path=str(audio_file),
                passphrase=passphrase
            )
            
            if result['success']:
                print(f"   ✅ Sample {i} added")
                if 'quality_metrics' in result:
                    metrics = result['quality_metrics']
                    print(f"      Duration: {metrics['duration']:.2f}s, SNR: {metrics['snr_db']:.1f} dB")
            else:
                print(f"   ❌ Error: {result['error']}")
                return
        
        print(f"\n✅ Enrollment complete!\n")
        
        # Step 4: Get user info
        print(f"📋 Step 4: Checking user registration status...")
        result = await voice_auth_service.get_user_info(username)
        if result['success']:
            print(f"   Username: {result['username']}")
            print(f"   Registered: {'Yes' if result['is_registered'] else 'No'}")
            print(f"   Samples: {result['samples_collected']}/{result['samples_required']}")
        print()
        
        # Step 5: Authenticate
        print(f"📋 Step 5: Testing authentication...")
        print(f"   Using audio file: {auth_file.name}")
        
        result = await voice_auth_service.authenticate(
            username=username,
            audio_file_path=str(auth_file),
            passphrase=passphrase
        )
        
        print()
        if result['success']:
            if result['authenticated']:
                print("🎉 AUTHENTICATION SUCCESSFUL! 🎉")
                print(f"   Similarity Score: {result['similarity_score']:.3f}")
                print(f"   Threshold: {result['threshold']:.3f}")
                print(f"   Match Quality: {(result['similarity_score'] - result['threshold']) * 100:.1f}% above threshold")
            else:
                print("❌ AUTHENTICATION FAILED")
                print(f"   Similarity Score: {result['similarity_score']:.3f}")
                print(f"   Threshold: {result['threshold']:.3f}")
                print(f"   Gap: {(result['threshold'] - result['similarity_score']) * 100:.1f}% below threshold")
        else:
            print(f"❌ Error: {result['error']}")
        
        print()
        print("=" * 70)
        print("  DEMO COMPLETE")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Try authenticating with different audio samples")
        print("  2. Test with another user to see authentication correctly rejects them")
        print("  3. Adjust SIMILARITY_THRESHOLD in config.py to tune security")
        print("  4. Explore the API at http://localhost:8000/docs")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await db_manager.close()


def print_setup_instructions():
    """Print instructions for setting up demo audio files"""
    print()
    print("=" * 70)
    print("  DEMO SETUP INSTRUCTIONS")
    print("=" * 70)
    print()
    print("This demo requires audio files. Here's how to create them:")
    print()
    print("OPTION 1: Record your own voice (Recommended)")
    print("-" * 70)
    print("1. Create a directory called 'demo_audio'")
    print("2. Record yourself saying: 'hello voice authentication'")
    print("3. Save 3 recordings as: voice1.wav, voice2.wav, voice3.wav")
    print("4. Record one more for testing: test_auth.wav")
    print()
    print("Recording tips:")
    print("  - Use a quiet environment")
    print("  - Speak clearly at normal pace")
    print("  - Keep microphone at same distance")
    print("  - WAV format at 16kHz is ideal")
    print()
    print("OPTION 2: Use online text-to-speech")
    print("-" * 70)
    print("1. Go to https://ttsmaker.com or similar")
    print("2. Enter text: 'hello voice authentication'")
    print("3. Download 4 different audio samples")
    print("4. Save as voice1.wav, voice2.wav, voice3.wav, test_auth.wav")
    print()
    print("OPTION 3: Use the CLI with your own files")
    print("-" * 70)
    print("  python cli.py register --username myuser")
    print("  python cli.py enroll --username myuser --audio file1.wav --passphrase 'test'")
    print("  python cli.py auth --username myuser --audio test.wav --passphrase 'test'")
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    print()
    print("Starting Voice Authentication Demo...")
    print()
    
    # Check if demo audio directory exists
    demo_dir = Path("demo_audio")
    if not demo_dir.exists() or not list(demo_dir.glob("*.wav")):
        print_setup_instructions()
        sys.exit(0)
    
    # Run demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()