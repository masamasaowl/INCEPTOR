"""
Voice Authentication Core - Fixed Version
Handles feature extraction and similarity computation
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import ks_2samp
from typing import Tuple, Dict
import io


class VoiceProcessor:
    """
    Voice feature extraction and comparison
    This is like a voice fingerprint scanner - it looks at many aspects of your voice
    to create a unique signature that can identify you!
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    @staticmethod
    def extract_features(audio: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, Dict]:
        """
        Extract comprehensive voice features
        
        Think of this like taking measurements of a person:
        - Height, weight, eye color, fingerprints, etc.
        - For voice: pitch, tone, resonance, rhythm, etc.
        
        We extract 172 different measurements to create a unique voice signature!
        """
        
        # === 1. MFCCs (Mel-Frequency Cepstral Coefficients) ===
        # This captures the "shape" of your vocal tract - like a fingerprint!
        # It's the most important feature for voice recognition
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)      # Average values
        mfcc_std = np.std(mfccs, axis=1)        # How much they vary
        mfcc_median = np.median(mfccs, axis=1)  # Middle values
        
        # === 2. Delta MFCCs ===
        # This captures how your voice CHANGES over time
        # Like measuring acceleration vs just speed
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mean = np.mean(delta_mfccs, axis=1)
        delta_std = np.std(delta_mfccs, axis=1)
        
        # === 3. Log-Mel Spectrogram ===
        # This represents the "texture" of your voice
        # Like looking at the grain pattern in wood
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        log_mel = librosa.power_to_db(mel_spec)
        log_mel_mean = np.mean(log_mel, axis=1)
        log_mel_std = np.std(log_mel, axis=1)
        
        # === 4. Pitch Features ===
        # How high or low your voice is - think soprano vs bass
        f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), 
                         fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_voiced = f0[f0 > 0]  # Only keep actual voice parts
        
        if len(f0_voiced) > 0:
            pitch_features = np.array([
                np.mean(f0_voiced),    # Average pitch
                np.std(f0_voiced),     # Pitch variation
                np.min(f0_voiced),     # Lowest pitch
                np.max(f0_voiced),     # Highest pitch
                np.ptp(f0_voiced),     # Pitch range
                np.median(f0_voiced)   # Middle pitch
            ])
        else:
            pitch_features = np.zeros(6)
        
        # === 5. Spectral Features ===
        # These capture the "brightness" and "richness" of your voice
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        # === Combine All Features ===
        # Stack everything together into one big feature vector
        features = np.concatenate([
            mfcc_mean, mfcc_std, mfcc_median,     # 39 features
            delta_mean, delta_std,                 # 26 features
            log_mel_mean, log_mel_std,             # 80 features
            pitch_features,                        # 6 features
            [np.mean(centroid), np.std(centroid)], # 2 features
            [np.mean(rolloff), np.std(rolloff)]    # 2 features
        ])
        
        # Clean up any invalid values (NaN or Infinity)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store frame-level features for distribution comparison
        # This is like keeping the "raw footage" for detailed analysis
        frame_features = {
            'mfccs': mfccs.T,
            'log_mel': log_mel.T,
            'delta_mfccs': delta_mfccs.T,
        }
        
        return features, frame_features
    
    @staticmethod
    def compute_similarity(features1: np.ndarray, features2: np.ndarray, 
                          frame1: Dict, frame2: Dict) -> Dict[str, float]:
        """
        Compute similarity metrics between two voice samples
        
        Think of this like comparing two fingerprints:
        - We use 4 different methods to check if they match
        - Each method looks at the data in a different way
        - Like using multiple detectives to solve the same case!
        
        Returns scores from 0.0 (completely different) to 1.0 (identical)
        """
        
        # === 1. Cosine Similarity (30% weight) ===
        # Measures the "angle" between two feature vectors
        # Like checking if two arrows point in the same direction
        # Good for: Pattern matching regardless of magnitude
        cosine_dist = cosine(features1, features2)
        cosine_sim = 1 - cosine_dist
        
        # === 2. Euclidean Distance (20% weight) - FIXED! ===
        # Measures the straight-line distance between two points
        # Like measuring the distance between two cities on a map
        euclidean_dist = euclidean(features1, features2)
        
        # FIXED: Proper normalization to get a similarity score 0-1
        # We normalize by the maximum possible distance in the feature space
        # The max distance is when all features are at opposite extremes
        feature_range = np.max(features1) - np.min(features1) + np.max(features2) - np.min(features2)
        if feature_range > 0:
            # Normalize: closer distances = higher similarity
            euclidean_sim = 1 - min(1.0, euclidean_dist / (feature_range * np.sqrt(len(features1))))
        else:
            euclidean_sim = 1.0  # If no variation, features are identical
        
        # Ensure it's in valid range [0, 1]
        euclidean_sim = max(0.0, min(1.0, euclidean_sim))
        
        # === 3. Bhattacharyya Coefficient (35% weight) ===
        # Compares the DISTRIBUTION of features
        # Like comparing the overall "shape" of two histograms
        # Good for: Capturing the statistical nature of voice patterns
        mfcc1 = frame1['mfccs']
        mfcc2 = frame2['mfccs']
        
        bhatt_sims = []
        for i in range(min(mfcc1.shape[1], mfcc2.shape[1])):
            # Create histograms for each MFCC coefficient
            hist1, bins = np.histogram(mfcc1[:, i], bins=20, density=True)
            hist2, _ = np.histogram(mfcc2[:, i], bins=bins, density=True)
            
            # Normalize histograms
            hist1 = hist1 / (np.sum(hist1) + 1e-8)
            hist2 = hist2 / (np.sum(hist2) + 1e-8)
            
            # Calculate Bhattacharyya coefficient
            bc = np.sum(np.sqrt(hist1 * hist2))
            bhatt_sims.append(bc)
        
        bhatt_sim = np.mean(bhatt_sims)
        
        # === 4. Kolmogorov-Smirnov Test (15% weight) ===
        # Statistical test to see if two samples come from the same distribution
        # Like a scientific hypothesis test
        # Returns a p-value: higher = more likely to be the same person
        ks_pvalues = []
        for i in range(min(5, mfcc1.shape[1], mfcc2.shape[1])):
            _, pvalue = ks_2samp(mfcc1[:, i], mfcc2[:, i])
            ks_pvalues.append(pvalue)
        
        ks_sim = np.mean(ks_pvalues)
        
        return {
            'cosine': float(cosine_sim),
            'euclidean': float(euclidean_sim),
            'bhattacharyya': float(bhatt_sim),
            'ks_test': float(ks_sim)
        }


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio from bytes (from web upload) and convert to proper format
    
    This handles the conversion from browser audio to the format we need.
    Like converting a JPEG to PNG - same picture, different format!
    """
    try:
        # Read audio from bytes
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        
        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        return audio
    except Exception as e:
        raise ValueError(f"Failed to load audio: {str(e)}")


def calculate_adaptive_thresholds(feature_vectors: list, frame_features_list: list, 
                                 processor: VoiceProcessor) -> Dict[str, float]:
    """
    Calculate personalized thresholds based on registration samples
    
    Think of this like calibrating a scale:
    - We measure the same person 3 times
    - See how consistent the measurements are
    - Set the "acceptable range" based on that consistency
    
    This makes the system adapt to each person's unique voice!
    """
    
    # Compare all pairs of registration samples
    similarity_scores = []
    for i in range(len(feature_vectors)):
        for j in range(i+1, len(feature_vectors)):
            scores = processor.compute_similarity(
                feature_vectors[i], feature_vectors[j],
                frame_features_list[i], frame_features_list[j]
            )
            similarity_scores.append(scores)
    
    # Calculate statistics
    avg_cosine = np.mean([s['cosine'] for s in similarity_scores])
    avg_euclidean = np.mean([s['euclidean'] for s in similarity_scores])
    avg_bhatt = np.mean([s['bhattacharyya'] for s in similarity_scores])
    avg_ks = np.mean([s['ks_test'] for s in similarity_scores])
    
    std_cosine = np.std([s['cosine'] for s in similarity_scores])
    std_euclidean = np.std([s['euclidean'] for s in similarity_scores])
    std_bhatt = np.std([s['bhattacharyya'] for s in similarity_scores])
    std_ks = np.std([s['ks_test'] for s in similarity_scores])
    
    # Set thresholds: mean - 2.5 * std
    # This creates a "safety margin" - we're being generous to account for natural variation
    safety_margin = 2.5
    
    thresholds = {
        'cosine': max(0.70, avg_cosine - safety_margin * std_cosine),
        'euclidean': max(0.40, avg_euclidean - safety_margin * std_euclidean),
        'bhattacharyya': max(0.50, avg_bhatt - safety_margin * std_bhatt),
        'ks_test': max(0.05, avg_ks - safety_margin * std_ks),
    }
    
    return thresholds, {
        'cosine_mean': float(avg_cosine),
        'euclidean_mean': float(avg_euclidean),
        'bhattacharyya_mean': float(avg_bhatt),
        'ks_test_mean': float(avg_ks),
    }