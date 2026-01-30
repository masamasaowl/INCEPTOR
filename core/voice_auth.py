"""
Core Voice Authentication Engine
Handles voice processing, embedding extraction, and speaker verification
Uses state-of-the-art ECAPA-TDNN model from SpeechBrain
"""

import torch
import torchaudio
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional, List
import logging
from datetime import datetime

from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceAuthEngine:
    """
    Voice Authentication Engine using ECAPA-TDNN speaker recognition
    
    This engine:
    1. Loads and preprocesses audio files
    2. Extracts speaker embeddings (voiceprints)
    3. Compares embeddings to verify speaker identity
    """
    
    def __init__(self):
        """
        Initialize the voice authentication engine
        Loads the pre-trained speaker recognition model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.sample_rate = settings.SAMPLE_RATE
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        
        logger.info(f"VoiceAuthEngine initialized on device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """
        Load the SpeechBrain ECAPA-TDNN speaker recognition model
        This is a state-of-the-art model trained on VoxCeleb dataset
        """
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            logger.info(f"Loading speaker recognition model: {settings.SPEAKER_MODEL}")
            self.model = EncoderClassifier.from_hparams(
                source=settings.SPEAKER_MODEL,
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            logger.info("Speaker recognition model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load speaker recognition model: {e}")
    
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and resample to target sample rate
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is unsupported
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if audio_path.suffix.lower() not in settings.ALLOWED_AUDIO_FORMATS:
            raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
        
        try:
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            
            logger.info(f"Audio loaded: {audio_path.name}, duration: {waveform.shape[1]/self.sample_rate:.2f}s")
            return waveform, self.sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            raise ValueError(f"Could not load audio file: {e}")
    
    def preprocess_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio: normalize, remove silence, reduce noise
        
        Args:
            waveform: Audio tensor
            
        Returns:
            Preprocessed audio tensor
        """
        # Convert to numpy for librosa processing
        audio_np = waveform.squeeze().numpy()
        
        # Normalize audio
        if settings.NORMALIZE_AUDIO:
            audio_np = librosa.util.normalize(audio_np)
        
        # Remove silence (trim leading/trailing silence)
        audio_np, _ = librosa.effects.trim(
            audio_np,
            top_db=20,  # Threshold in dB below reference
            frame_length=2048,
            hop_length=512
        )
        
        # Optional: Basic noise reduction
        if settings.NOISE_REDUCTION:
            # Apply a simple high-pass filter to remove low-frequency noise
            audio_np = librosa.effects.preemphasis(audio_np, coef=0.97)
        
        # Convert back to tensor
        preprocessed = torch.from_numpy(audio_np).unsqueeze(0).float()
        
        return preprocessed
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract speaker embedding (voiceprint) from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Numpy array containing the speaker embedding
            
        Raises:
            ValueError: If audio is too short or extraction fails
        """
        # Load and preprocess audio
        waveform, sr = self.load_audio(audio_path)
        
        # Check audio duration
        duration = waveform.shape[1] / sr
        if duration < settings.MIN_AUDIO_LENGTH:
            raise ValueError(
                f"Audio too short: {duration:.2f}s (minimum: {settings.MIN_AUDIO_LENGTH}s)"
            )
        if duration > settings.MAX_AUDIO_LENGTH:
            logger.warning(f"Audio longer than maximum ({settings.MAX_AUDIO_LENGTH}s), will be truncated")
            max_samples = int(settings.MAX_AUDIO_LENGTH * sr)
            waveform = waveform[:, :max_samples]
        
        # Preprocess
        waveform = self.preprocess_audio(waveform)
        
        # Move to device
        waveform = waveform.to(self.device)
        
        try:
            # Extract embedding using the model
            with torch.no_grad():
                embedding = self.model.encode_batch(waveform)
                embedding = embedding.squeeze().cpu().numpy()
            
            logger.info(f"Embedding extracted: shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            raise ValueError(f"Could not extract voice embedding: {e}")
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Similarity score between 0 and 1 (higher = more similar)
        """
        # Normalize embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        # Convert to 0-1 range (cosine similarity is between -1 and 1)
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def verify_speaker(
        self,
        test_embedding: np.ndarray,
        reference_embeddings: List[np.ndarray],
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Verify if test embedding matches reference embeddings
        
        Args:
            test_embedding: Embedding to verify
            reference_embeddings: List of reference embeddings from enrollment
            threshold: Similarity threshold (uses config default if not provided)
            
        Returns:
            Tuple of (is_match, average_similarity_score)
        """
        if not reference_embeddings:
            raise ValueError("No reference embeddings provided")
        
        threshold = threshold or self.similarity_threshold
        
        # Calculate similarity with each reference embedding
        similarities = [
            self.calculate_similarity(test_embedding, ref_emb)
            for ref_emb in reference_embeddings
        ]
        
        # Use average similarity across all samples
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        
        # Decision: match if average similarity exceeds threshold
        is_match = avg_similarity >= threshold
        
        logger.info(
            f"Verification result: {'MATCH' if is_match else 'NO MATCH'} "
            f"(avg: {avg_similarity:.3f}, max: {max_similarity:.3f}, threshold: {threshold})"
        )
        
        return is_match, float(avg_similarity)
    
    def save_embedding(self, embedding: np.ndarray, save_path: str) -> str:
        """
        Save embedding to disk as .npy file
        
        Args:
            embedding: Speaker embedding to save
            save_path: Path where to save (without extension)
            
        Returns:
            Full path to saved file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add .npy extension if not present
        if save_path.suffix != '.npy':
            save_path = save_path.with_suffix('.npy')
        
        np.save(save_path, embedding)
        logger.info(f"Embedding saved to: {save_path}")
        
        return str(save_path)
    
    def load_embedding(self, embedding_path: str) -> np.ndarray:
        """
        Load embedding from disk
        
        Args:
            embedding_path: Path to .npy file
            
        Returns:
            Speaker embedding as numpy array
        """
        embedding_path = Path(embedding_path)
        
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        embedding = np.load(embedding_path)
        logger.info(f"Embedding loaded from: {embedding_path}")
        
        return embedding
    
    def calculate_audio_quality_metrics(self, audio_path: str) -> dict:
        """
        Calculate audio quality metrics (SNR, duration, etc.)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with quality metrics
        """
        waveform, sr = self.load_audio(audio_path)
        audio_np = waveform.squeeze().numpy()
        
        # Duration
        duration = len(audio_np) / sr
        
        # Simple SNR estimation (not perfect but useful)
        # Calculate power of signal
        signal_power = np.mean(audio_np ** 2)
        
        # Estimate noise from silent parts (bottom 10% of amplitude)
        sorted_abs = np.sort(np.abs(audio_np))
        noise_samples = sorted_abs[:int(len(sorted_abs) * 0.1)]
        noise_power = np.mean(noise_samples ** 2)
        
        # Calculate SNR in dB
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'snr_db': float(snr),
            'rms_amplitude': float(np.sqrt(signal_power))
        }


# Singleton instance for use across the application
voice_auth_engine = VoiceAuthEngine()