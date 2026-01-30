"""
Installation Test Script
Run this to check if all dependencies are installed correctly
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    
    print("🧪 Testing Voice Authentication System Setup")
    print("=" * 50)
    
    packages = [
        ('numpy', 'NumPy'),
        ('librosa', 'Librosa'),
        ('sounddevice', 'SoundDevice'),
        ('soundfile', 'SoundFile'),
        ('scipy', 'SciPy')
    ]
    
    all_good = True
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name:15s} - Installed")
        except ImportError:
            print(f"❌ {name:15s} - NOT installed")
            all_good = False
    
    print("=" * 50)
    
    if all_good:
        print("\n🎉 All dependencies installed successfully!")
        print("You're ready to run: python3 voice_auth.py")
        
        # Test microphone access
        print("\n🎤 Testing microphone access...")
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            print(f"✅ Found {len(devices)} audio devices")
            
            # Show default input device
            default_input = sd.query_devices(kind='input')
            print(f"Default microphone: {default_input['name']}")
            
        except Exception as e:
            print(f"⚠️  Microphone test failed: {e}")
            print("You may need to install PortAudio: brew install portaudio")
        
    else:
        print("\n❌ Some dependencies are missing!")
        print("Run this command to install them:")
        print("pip3 install -r requirements.txt")
        print("\nOr install individually:")
        print("pip3 install numpy librosa sounddevice soundfile scipy")
    
    return all_good

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)