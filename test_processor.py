"""
Test Script for Voice Processor
Run this to verify the core logic works correctly
"""

import numpy as np
from voice_processor import VoiceProcessor, calculate_adaptive_thresholds

def test_euclidean_fix():
    """Test that Euclidean distance is now properly normalized"""
    print("="*70)
    print("🧪 Testing Euclidean Distance Fix")
    print("="*70)
    
    processor = VoiceProcessor()
    
    # Create two random feature vectors (simulating voice features)
    features1 = np.random.randn(172)
    features2 = features1 + np.random.randn(172) * 0.1  # Similar but slightly different
    
    # Create dummy frame features
    frame_features1 = {
        'mfccs': np.random.randn(100, 13),
        'log_mel': np.random.randn(100, 40),
        'delta_mfccs': np.random.randn(100, 13),
    }
    frame_features2 = {
        'mfccs': frame_features1['mfccs'] + np.random.randn(100, 13) * 0.1,
        'log_mel': frame_features1['log_mel'] + np.random.randn(100, 40) * 0.1,
        'delta_mfccs': frame_features1['delta_mfccs'] + np.random.randn(100, 13) * 0.1,
    }
    
    # Compute similarity
    scores = processor.compute_similarity(features1, features2, frame_features1, frame_features2)
    
    print("\n📊 Similarity Scores:")
    print(f"   Cosine:         {scores['cosine']*100:6.2f}%")
    print(f"   Euclidean:      {scores['euclidean']*100:6.2f}%  ← Should be 0-100%!")
    print(f"   Bhattacharyya:  {scores['bhattacharyya']*100:6.2f}%")
    print(f"   KS-Test:        {scores['ks_test']*100:6.2f}%")
    
    # Verify all scores are in valid range
    all_valid = all(0 <= score <= 1 for score in scores.values())
    
    if all_valid:
        print("\n✅ All scores are in valid range (0-100%)!")
        print("✅ Euclidean distance is FIXED!")
    else:
        print("\n❌ Some scores are out of range!")
        for metric, score in scores.items():
            if score < 0 or score > 1:
                print(f"   ❌ {metric}: {score}")
    
    print("="*70)
    return all_valid


def test_feature_extraction():
    """Test feature extraction with synthetic audio"""
    print("\n" + "="*70)
    print("🧪 Testing Feature Extraction")
    print("="*70)
    
    processor = VoiceProcessor()
    
    # Create synthetic audio (1 second of sine wave)
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz (A note)
    
    print("\n🎵 Created synthetic audio (1 second, 440Hz)")
    
    # Extract features
    features, frame_features = processor.extract_features(audio, sample_rate)
    
    print(f"\n📊 Extracted Features:")
    print(f"   Total features: {len(features)}")
    print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"   Frame features:")
    print(f"      MFCCs: {frame_features['mfccs'].shape}")
    print(f"      Log-Mel: {frame_features['log_mel'].shape}")
    print(f"      Delta MFCCs: {frame_features['delta_mfccs'].shape}")
    
    # Verify features
    has_valid_features = len(features) > 0 and not np.any(np.isnan(features))
    
    if has_valid_features:
        print("\n✅ Feature extraction works correctly!")
    else:
        print("\n❌ Feature extraction has issues!")
    
    print("="*70)
    return has_valid_features


def main():
    print("\n" + "="*70)
    print("🎙️  VOICE PROCESSOR TEST SUITE")
    print("="*70)
    
    # Run tests
    test1_passed = test_feature_extraction()
    test2_passed = test_euclidean_fix()
    
    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    print(f"   Feature Extraction: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Euclidean Fix:      {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Core logic is working correctly!")
        print("✨ You're ready to start the server!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()