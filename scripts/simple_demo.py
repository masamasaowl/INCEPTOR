#!/usr/bin/env python3
"""
SUPER SIMPLE VOICE AUTH DEMO
============================

What this does:
1. You register by saying "Hello" 3 times
2. You login by saying "Hello"
3. If you say anything else, you're denied

That's it! Simple as that!
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings, ensure_directories
from core.service import voice_auth_service
from database.manager import db_manager


def print_header(text):
    """Print a nice header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_success(text):
    """Print success message"""
    print(f"✅ {text}")


def print_error(text):
    """Print error message"""
    print(f"❌ {text}")


def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")


async def simple_demo():
    """
    The simplest possible demo:
    - Register user
    - Say "Hello" 3 times to train
    - Try to login with "Hello"
    """
    
    print_header("🎙️ SIMPLE VOICE AUTHENTICATION DEMO")
    
    # Initialize
    print_info("Setting up the system...")
    ensure_directories()
    await db_manager.initialize_database()
    print_success("System ready!\n")
    
    # Configuration
    username = "demo_user"
    passphrase = "Hello"  # This is what you'll say!
    
    print_info("This demo shows:")
    print("  1️⃣  Register a user")
    print("  2️⃣  Train with 3 'Hello' recordings")
    print("  3️⃣  Authenticate with 'Hello'")
    print("  4️⃣  Fail authentication with wrong voice\n")
    
    # Check for audio files
    print_info("Looking for audio files...")
    demo_dir = Path("demo_audio")
    
    if not demo_dir.exists():
        print_error("No demo_audio folder found!")
        print("\n📝 To run this demo:")
        print("  1. Create a folder: demo_audio/")
        print("  2. Record yourself saying 'Hello' 3 times")
        print("  3. Save as: voice1.wav, voice2.wav, voice3.wav")
        print("  4. Record once more for testing: test_hello.wav")
        print("\n💡 Recording tips:")
        print("  - Use your phone's voice recorder")
        print("  - Speak clearly: 'Hello'")
        print("  - Save as WAV format")
        print("  - Each recording should be 1-3 seconds\n")
        return
    
    # Required files
    training_files = [
        demo_dir / "voice1.wav",
        demo_dir / "voice2.wav", 
        demo_dir / "voice3.wav"
    ]
    test_file = demo_dir / "test_hello.wav"
    
    # Check all files exist
    missing = []
    for f in training_files + [test_file]:
        if not f.exists():
            missing.append(f.name)
    
    if missing:
        print_error(f"Missing audio files: {', '.join(missing)}")
        print("\n📝 Please create these files in demo_audio/ folder:")
        for m in missing:
            print(f"  - {m}")
        print("\n💡 Each file should be you saying 'Hello'\n")
        return
    
    print_success("Found all audio files!\n")
    
    try:
        # STEP 1: Register user
        print_header("STEP 1: Register User")
        print(f"Creating user: {username}")
        
        # Delete if exists
        await voice_auth_service.delete_user(username)
        
        result = await voice_auth_service.register_user(username, "demo@example.com")
        if result['success']:
            print_success(f"User '{username}' registered!")
        else:
            print_error(f"Registration failed: {result['error']}")
            return
        
        # STEP 2: Train with voice samples
        print_header("STEP 2: Train Your Voice")
        print(f"Teaching the system YOUR way of saying '{passphrase}'...\n")
        
        for i, audio_file in enumerate(training_files, 1):
            print(f"📼 Processing recording {i}/3: {audio_file.name}")
            
            result = await voice_auth_service.add_voice_sample(
                username=username,
                audio_file_path=str(audio_file),
                passphrase=passphrase
            )
            
            if result['success']:
                print_success(f"Sample {i} learned!")
                
                # Show quality
                if 'quality_metrics' in result:
                    metrics = result['quality_metrics']
                    print(f"   Duration: {metrics['duration']:.1f}s")
                    print(f"   Quality: {metrics['snr_db']:.0f} dB SNR")
                print()
            else:
                print_error(f"Failed: {result['error']}")
                return
        
        print_success("Training complete! System knows your voice now!\n")
        
        # STEP 3: Test authentication
        print_header("STEP 3: Test Authentication")
        print(f"Now let's see if the system recognizes you...\n")
        print(f"🎤 Testing with: {test_file.name}")
        print(f"🗣️  You should be saying: '{passphrase}'")
        
        result = await voice_auth_service.authenticate(
            username=username,
            audio_file_path=str(test_file),
            passphrase=passphrase
        )
        
        print()
        if result['success'] and result['authenticated']:
            similarity = result['similarity_score']
            threshold = result['threshold']
            
            print_header("🎉 AUTHENTICATION SUCCESSFUL! 🎉")
            print(f"✓ Voice recognized as {username}")
            print(f"✓ Similarity: {similarity:.1%} (need {threshold:.0%})")
            print(f"✓ Quality: {(similarity - threshold) * 100:.1f}% above threshold")
            print("\n🔓 ACCESS GRANTED!")
            
        else:
            similarity = result.get('similarity_score', 0)
            threshold = result.get('threshold', 0.75)
            
            print_header("❌ AUTHENTICATION FAILED")
            print(f"✗ Voice not recognized")
            print(f"✗ Similarity: {similarity:.1%} (need {threshold:.0%})")
            print(f"✗ Gap: {(threshold - similarity) * 100:.1f}% below threshold")
            print("\n🔒 ACCESS DENIED!")
        
        # STEP 4: Show what happens with wrong passphrase
        print_header("BONUS: What if you say something different?")
        print_info("The system checks BOTH:")
        print("  1. Is this the right voice? (Your voice)")
        print("  2. Are they saying the right thing? ('Hello')")
        print("\nIf you say anything else, even in your voice:")
        print("❌ DENIED - Wrong passphrase!\n")
        
        # Final summary
        print_header("📊 Summary")
        user_info = await voice_auth_service.get_user_info(username)
        if user_info['success']:
            print(f"Username: {user_info['username']}")
            print(f"Registered: {'Yes' if user_info['is_registered'] else 'No'}")
            print(f"Voice samples: {user_info['samples_collected']}")
            print(f"Passphrase: '{passphrase}'")
        
        print_header("🎯 What You Just Learned")
        print("1. Registered a user account")
        print("2. Trained the AI with 3 voice samples")
        print("3. Successfully authenticated with your voice")
        print("4. The system now knows YOUR unique voice!\n")
        
        print_info("Try this:")
        print("  1. Have a friend say 'Hello' and test")
        print("  2. Change your voice and test")
        print("  3. Record in a noisy environment and test\n")
        
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await db_manager.close()


def show_quick_test_instructions():
    """Show how to quickly test the system"""
    print_header("🚀 Quick Test Instructions")
    print("\n1️⃣  Record 4 audio files:")
    print("   - voice1.wav (you saying 'Hello')")
    print("   - voice2.wav (you saying 'Hello')")
    print("   - voice3.wav (you saying 'Hello')")
    print("   - test_hello.wav (you saying 'Hello')")
    
    print("\n2️⃣  Save them in: demo_audio/ folder")
    
    print("\n3️⃣  Run this script:")
    print("   python scripts/simple_demo.py")
    
    print("\n💡 How to record:")
    print("   macOS: Use QuickTime or Voice Memos")
    print("   Windows: Use Voice Recorder")
    print("   Phone: Record and transfer to computer")
    print("   Online: https://online-voice-recorder.com/\n")


if __name__ == "__main__":
    print("\n🎙️ Starting Simple Voice Auth Demo...\n")
    
    # Check if demo audio exists
    demo_dir = Path("demo_audio")
    if not demo_dir.exists() or not list(demo_dir.glob("*.wav")):
        show_quick_test_instructions()
        sys.exit(0)
    
    # Run demo
    try:
        asyncio.run(simple_demo())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()