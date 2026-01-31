"""
Voice Authentication System V5 - POLISHED
Industry-Standard Features with Calibrated Thresholds
Optimized for hackathon demonstration
"""

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import os
import json
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import ks_2samp
from datetime import datetime
import time

class VoiceAuthenticatorV5:
    def __init__(self, data_dir="voice_data"):
        """Initialize with industry-standard configuration"""
        self.data_dir = data_dir
        self.sample_rate = 16000
        self.duration = 3
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        self.logs_dir = os.path.join(self.data_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def extract_pitch_features(self, audio):
        """Extract F0 (pitch) features - highly discriminative for speaker ID"""
        f0 = librosa.yin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'), 
            sr=self.sample_rate
        )
        
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) == 0:
            return np.zeros(6)
        
        return np.array([
            np.mean(f0_voiced),      # Mean pitch
            np.std(f0_voiced),       # Pitch variability
            np.min(f0_voiced),       # Minimum pitch
            np.max(f0_voiced),       # Maximum pitch
            np.ptp(f0_voiced),       # Pitch range
            np.median(f0_voiced),    # Median pitch
        ])
    
    def extract_jitter_shimmer(self, audio):
        """Jitter: Frequency perturbation, Shimmer: Amplitude perturbation"""
        f0 = librosa.yin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'), 
            sr=self.sample_rate
        )
        
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) < 10:
            return np.zeros(2)
        
        # Jitter calculation
        periods = 1.0 / (f0_voiced + 1e-8)
        period_diffs = np.abs(np.diff(periods))
        jitter = np.mean(period_diffs) / np.mean(periods) if len(periods) > 0 else 0
        
        # Shimmer calculation
        rms = librosa.feature.rms(y=audio)[0]
        rms_diffs = np.abs(np.diff(rms))
        shimmer = np.mean(rms_diffs) / np.mean(rms) if len(rms) > 0 else 0
        
        return np.array([jitter, shimmer])
    
    def extract_formants(self, audio):
        """Formant frequencies (F1, F2, F3) - vocal tract resonances"""
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Get spectral peaks as formant estimates
        spec = np.abs(librosa.stft(emphasized_audio))
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        spec_mean = np.mean(spec, axis=1)
        
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(spec_mean, height=np.max(spec_mean) * 0.1)
        
        peak_freqs = freqs[peaks]
        peak_freqs = peak_freqs[(peak_freqs > 200) & (peak_freqs < 4000)]
        peak_freqs = np.sort(peak_freqs)
        
        if len(peak_freqs) >= 3:
            return np.array([peak_freqs[0], peak_freqs[1], peak_freqs[2]])
        return np.zeros(3)
    
    def extract_spectral_dynamics(self, audio):
        """Short-term spectral dynamics"""
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        
        # Spectral flux
        spec = np.abs(librosa.stft(audio))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
        
        # Spectral entropy
        spec_normalized = spec / (np.sum(spec, axis=0) + 1e-8)
        entropy = -np.sum(spec_normalized * np.log(spec_normalized + 1e-8), axis=0)
        
        return np.array([
            np.mean(centroid), np.std(centroid),
            np.mean(rolloff), np.std(rolloff),
            np.mean(flux), np.std(flux),
            np.mean(entropy), np.std(entropy),
        ])
    
    def extract_vad_quality_metrics(self, audio):
        """Voice Activity Detection + Signal Quality"""
        rms = librosa.feature.rms(y=audio)[0]
        voice_threshold = np.mean(rms) * 0.5
        vad = rms > voice_threshold
        speech_ratio = np.sum(vad) / len(vad)
        
        # SNR estimate
        sorted_rms = np.sort(rms)
        noise_floor = np.mean(sorted_rms[:int(len(sorted_rms) * 0.1)])
        signal_power = np.mean(rms[vad]) if np.any(vad) else np.mean(rms)
        snr = 10 * np.log10((signal_power**2) / (noise_floor**2 + 1e-8))
        
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        
        return np.array([speech_ratio, snr, energy_mean, energy_std])
    
    def extract_duration_timing_features(self, audio):
        """Duration and timing statistics"""
        rms = librosa.feature.rms(y=audio)[0]
        voice_threshold = np.mean(rms) * 0.5
        vad = rms > voice_threshold
        
        speech_frames = np.sum(vad)
        hop_length = 512
        speech_duration = speech_frames * hop_length / self.sample_rate
        
        vad_diff = np.diff(np.concatenate([[0], vad.astype(int), [0]]))
        starts = np.where(vad_diff == 1)[0]
        ends = np.where(vad_diff == -1)[0]
        num_utterances = len(starts)
        speech_rate = num_utterances / (len(audio) / self.sample_rate) if len(audio) > 0 else 0
        
        pause_lengths = []
        for i in range(len(starts) - 1):
            pause_length = starts[i+1] - ends[i]
            pause_lengths.append(pause_length)
        
        avg_pause = np.mean(pause_lengths) if pause_lengths else 0
        
        return np.array([speech_duration, speech_rate, avg_pause, num_utterances])
    
    def extract_comprehensive_features(self, audio):
        """Extract all industry-standard features"""
        # Voice activity detection
        rms = librosa.feature.rms(y=audio)[0]
        voice_threshold = np.mean(rms) * 0.3
        vad = rms > voice_threshold
        
        if np.sum(vad) < 5:
            print("⚠️  Very little speech detected!")
        
        # 1. MFCCs (13 coefficients - industry standard)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_median = np.median(mfccs, axis=1)
        
        # 2. Delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mean = np.mean(delta_mfccs, axis=1)
        delta_std = np.std(delta_mfccs, axis=1)
        
        # 3. Log-Mel filterbank energies (40 bands)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=40)
        log_mel = librosa.power_to_db(mel_spec)
        log_mel_mean = np.mean(log_mel, axis=1)
        log_mel_std = np.std(log_mel, axis=1)
        
        # 4. Pitch features
        pitch_features = self.extract_pitch_features(audio)
        
        # 5. Jitter & Shimmer
        jitter_shimmer = self.extract_jitter_shimmer(audio)
        
        # 6. Formants
        formants = self.extract_formants(audio)
        
        # 7. Spectral dynamics
        spectral_features = self.extract_spectral_dynamics(audio)
        
        # 8. Quality metrics
        quality_features = self.extract_vad_quality_metrics(audio)
        
        # 9. Timing features
        timing_features = self.extract_duration_timing_features(audio)
        
        # Combine all features
        summary_features = np.concatenate([
            mfcc_mean, mfcc_std, mfcc_median,      # 39
            delta_mean, delta_std,                  # 26
            log_mel_mean, log_mel_std,              # 80
            pitch_features,                         # 6
            jitter_shimmer,                         # 2
            formants,                               # 3
            spectral_features,                      # 8
            quality_features,                       # 4
            timing_features,                        # 4
        ])
        
        # Clean up NaN/Inf
        summary_features = np.nan_to_num(summary_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store frame-level features for distribution comparison
        frame_features = {
            'mfccs': mfccs.T,
            'log_mel': log_mel.T,
            'delta_mfccs': delta_mfccs.T,
        }
        
        print(f"📊 Extracted {len(summary_features)} features")
        
        return summary_features, frame_features
    
    def compute_statistical_distance(self, features1, features2, frame_features1, frame_features2):
        """Compute multiple statistical distance metrics"""
        
        # 1. Cosine similarity
        cosine_dist = cosine(features1, features2)
        cosine_sim = 1 - cosine_dist
        
        # 2. Euclidean distance (fixed normalization)
        euclidean_dist = euclidean(features1, features2)
        # Normalize based on feature vector length for better scaling
        max_possible_dist = np.sqrt(len(features1))  # Maximum possible distance
        euclidean_sim = 1 - (euclidean_dist / (max_possible_dist + 1e-8))
        
        # 3. Bhattacharyya distance on MFCC distributions
        mfcc1 = frame_features1['mfccs']
        mfcc2 = frame_features2['mfccs']
        
        bhattacharyya_sims = []
        for i in range(min(mfcc1.shape[1], mfcc2.shape[1])):
            hist1, bins = np.histogram(mfcc1[:, i], bins=20, density=True)
            hist2, _ = np.histogram(mfcc2[:, i], bins=bins, density=True)
            
            hist1 = hist1 / (np.sum(hist1) + 1e-8)
            hist2 = hist2 / (np.sum(hist2) + 1e-8)
            
            bc = np.sum(np.sqrt(hist1 * hist2))
            bhattacharyya_sims.append(bc)
        
        bhattacharyya_sim = np.mean(bhattacharyya_sims)
        
        # 4. Kolmogorov-Smirnov test
        ks_pvalues = []
        for i in range(min(5, mfcc1.shape[1], mfcc2.shape[1])):
            _, pvalue = ks_2samp(mfcc1[:, i], mfcc2[:, i])
            ks_pvalues.append(pvalue)
        
        ks_similarity = np.mean(ks_pvalues)
        
        return {
            'cosine': cosine_sim,
            'euclidean': euclidean_sim,
            'bhattacharyya': bhattacharyya_sim,
            'ks_test': ks_similarity
        }
    
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
            print("⚠️  WARNING: Very quiet! Speak louder.")
        
        return audio.flatten()
    
    def playback_audio(self, audio):
        """Play back the recorded audio"""
        response = input("\n🔊 Play back recording? (y/n): ").strip().lower()
        if response == 'y':
            print("Playing...")
            sd.play(audio, self.sample_rate)
            sd.wait()
            print("✓ Playback finished")
            
            satisfied = input("Happy with this? (y/n): ").strip().lower()
            return satisfied == 'y'
        return True
    
    def register_user(self, username):
        """Register new user with calibrated thresholds"""
        print(f"\n{'='*70}")
        print(f"REGISTRATION - User: {username}")
        print(f"{'='*70}")
        
        feature_vectors = []
        frame_features_list = []
        
        for i in range(3):
            while True:
                print(f"\n{'─'*70}")
                print(f"📝 Sample {i+1}/3")
                print(f"{'─'*70}")
                
                audio = self.record_audio_with_countdown()
                
                if self.playback_audio(audio):
                    audio_path = os.path.join(self.data_dir, f"{username}_sample_{i+1}.wav")
                    sf.write(audio_path, audio, self.sample_rate)
                    
                    features, frame_feats = self.extract_comprehensive_features(audio)
                    feature_vectors.append(features)
                    frame_features_list.append(frame_feats)
                    break
                else:
                    print("Recording again...")
            
            if i < 2:
                input("\n⏸️  Press Enter for next sample...")
        
        # Calculate statistics
        avg_features = np.mean(feature_vectors, axis=0)
        
        # === CALIBRATION ===
        print(f"\n{'='*70}")
        print(f"🔬 CALIBRATING THRESHOLDS")
        print(f"{'='*70}")
        
        similarity_scores = []
        for i in range(len(feature_vectors)):
            for j in range(i+1, len(feature_vectors)):
                scores = self.compute_statistical_distance(
                    feature_vectors[i], feature_vectors[j],
                    frame_features_list[i], frame_features_list[j]
                )
                similarity_scores.append(scores)
        
        # Calculate averages
        avg_cosine = np.mean([s['cosine'] for s in similarity_scores])
        avg_euclidean = np.mean([s['euclidean'] for s in similarity_scores])
        avg_bhattacharyya = np.mean([s['bhattacharyya'] for s in similarity_scores])
        avg_ks = np.mean([s['ks_test'] for s in similarity_scores])
        
        # Standard deviations
        std_cosine = np.std([s['cosine'] for s in similarity_scores])
        std_euclidean = np.std([s['euclidean'] for s in similarity_scores])
        std_bhattacharyya = np.std([s['bhattacharyya'] for s in similarity_scores])
        std_ks = np.std([s['ks_test'] for s in similarity_scores])
        
        # Set adaptive thresholds (2.5 std below mean)
        safety_margin = 2.5
        
        thresholds = {
            'cosine': max(0.70, avg_cosine - safety_margin * std_cosine),
            'euclidean': max(0.40, avg_euclidean - safety_margin * std_euclidean),
            'bhattacharyya': max(0.50, avg_bhattacharyya - safety_margin * std_bhattacharyya),
            'ks_test': max(0.05, avg_ks - safety_margin * std_ks),
        }
        
        print(f"\nSample Similarities:")
        print(f"  Cosine:         {avg_cosine*100:.1f}% (±{std_cosine*100:.1f}%)")
        print(f"  Euclidean:      {avg_euclidean*100:.1f}% (±{std_euclidean*100:.1f}%)")
        print(f"  Bhattacharyya:  {avg_bhattacharyya*100:.1f}% (±{std_bhattacharyya*100:.1f}%)")
        print(f"  KS-Test:        {avg_ks*100:.1f}% (±{std_ks*100:.1f}%)")
        
        print(f"\nCalibrated Thresholds:")
        print(f"  Cosine:         ≥{thresholds['cosine']*100:.1f}%")
        print(f"  Euclidean:      ≥{thresholds['euclidean']*100:.1f}%")
        print(f"  Bhattacharyya:  ≥{thresholds['bhattacharyya']*100:.1f}%")
        print(f"  KS-Test:        ≥{thresholds['ks_test']*100:.1f}%")
        
        if avg_cosine < 0.85:
            print("\n⚠️  Low consistency between samples!")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm != 'y':
                return False
        else:
            print("\n✅ Good registration quality!")
        
        print(f"{'='*70}")
        
        # Save profile
        user_profile = {
            'username': username,
            'features': avg_features.tolist(),
            'thresholds': thresholds,
            'registration_stats': {
                'cosine_mean': float(avg_cosine),
                'euclidean_mean': float(avg_euclidean),
                'bhattacharyya_mean': float(avg_bhattacharyya),
                'ks_test_mean': float(avg_ks),
            },
            'registered_at': datetime.now().isoformat(),
            'sample_count': 3,
            'feature_count': len(avg_features)
        }
        
        profile_path = os.path.join(self.data_dir, f"{username}_profile.json")
        with open(profile_path, 'w') as f:
            json.dump(user_profile, f, indent=2)
        
        print(f"\n✅ Registration successful!")
        print(f"📁 Profile: {profile_path}")
        print(f"🎯 Thresholds calibrated")
        return True
    
    def authenticate_user(self, username):
        """Authenticate using calibrated thresholds"""
        print(f"\n{'='*70}")
        print(f"AUTHENTICATION - User: {username}")
        print(f"{'='*70}")
        
        profile_path = os.path.join(self.data_dir, f"{username}_profile.json")
        if not os.path.exists(profile_path):
            print(f"❌ User '{username}' not found!")
            return False
        
        with open(profile_path, 'r') as f:
            user_profile = json.load(f)
        
        stored_features = np.array(user_profile['features'])
        thresholds = user_profile['thresholds']
        
        audio = self.record_audio_with_countdown()
        
        response = input("\n🔊 Hear recording? (y/n): ").strip().lower()
        if response == 'y':
            sd.play(audio, self.sample_rate)
            sd.wait()
        
        test_features, test_frame_features = self.extract_comprehensive_features(audio)
        
        # Load reference for frame comparison
        ref_audio_path = os.path.join(self.data_dir, f"{username}_sample_1.wav")
        if os.path.exists(ref_audio_path):
            ref_audio, _ = sf.read(ref_audio_path)
            _, ref_frame_features = self.extract_comprehensive_features(ref_audio)
        else:
            ref_frame_features = test_frame_features
        
        # Compute metrics
        scores = self.compute_statistical_distance(
            stored_features, test_features,
            ref_frame_features, test_frame_features
        )
        
        # Check thresholds
        checks = {
            'cosine': scores['cosine'] >= thresholds['cosine'],
            'euclidean': scores['euclidean'] >= thresholds['euclidean'],
            'bhattacharyya': scores['bhattacharyya'] >= thresholds['bhattacharyya'],
            'ks_test': scores['ks_test'] >= thresholds['ks_test'],
        }
        
        # Weighted voting: need 3 out of 4 to pass
        checks_passed = sum(checks.values())
        all_pass = checks_passed >= 3
        
        # Combined score (INCREASED Bhattacharyya weight from 0.25 to 0.35)
        combined_score = (
            0.30 * scores['cosine'] +
            0.20 * scores['euclidean'] +
            0.35 * scores['bhattacharyya'] +  # INCREASED
            0.15 * scores['ks_test']
        )
        
        print(f"\n{'='*70}")
        print(f"📊 AUTHENTICATION ANALYSIS")
        print(f"{'='*70}")
        print(f"Metric Scores vs Thresholds:")
        print(f"─"*70)
        print(f"  Cosine:           {scores['cosine']*100:5.1f}% {'✅' if checks['cosine'] else '❌'} (≥{thresholds['cosine']*100:.1f}%)")
        print(f"  Euclidean:        {scores['euclidean']*100:5.1f}% {'✅' if checks['euclidean'] else '❌'} (≥{thresholds['euclidean']*100:.1f}%)")
        print(f"  Bhattacharyya:    {scores['bhattacharyya']*100:5.1f}% {'✅' if checks['bhattacharyya'] else '❌'} (≥{thresholds['bhattacharyya']*100:.1f}%)")
        print(f"  KS-Test:          {scores['ks_test']*100:5.1f}% {'✅' if checks['ks_test'] else '❌'} (≥{thresholds['ks_test']*100:.1f}%)")
        print(f"─"*70)
        print(f"Combined Score:     {combined_score*100:.1f}%")
        print(f"Checks Passed:      {checks_passed}/4 (need ≥3)")
        print(f"{'='*70}")
        
        # Log attempt
        log_entry = {
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'scores': {k: float(v) for k, v in scores.items()},
            'thresholds': {k: float(v) for k, v in thresholds.items()},
            'checks': {k: bool(v) for k, v in checks.items()},
            'checks_passed': int(checks_passed),
            'combined_score': float(combined_score),
            'success': bool(all_pass)
        }
        
        log_file = os.path.join(self.logs_dir, f"{username}_auth_log.json")
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        logs.append(log_entry)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        # Decision
        if all_pass:
            print(f"\n✅ AUTHENTICATION SUCCESSFUL!")
            print(f"Welcome, {username}! 🎉")
            print(f"({checks_passed}/4 metrics passed)")
            return True
        else:
            print(f"\n❌ AUTHENTICATION FAILED!")
            print(f"Only {checks_passed}/4 metrics passed (need ≥3)")
            
            print(f"\nFailed checks:")
            for metric, passed in checks.items():
                if not passed:
                    print(f"  ❌ {metric}: {scores[metric]*100:.1f}% < {thresholds[metric]*100:.1f}%")
            
            print("\n💡 Possible reasons:")
            print("   - Different speaker")
            print("   - Different recording environment")
            print("   - Voice affected by illness/stress")
            
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
                    'auth_attempts': auth_count,
                    'successful_auths': success_count
                })
        
        return users
    
    def show_statistics(self, username):
        """Show detailed statistics"""
        log_file = os.path.join(self.logs_dir, f"{username}_auth_log.json")
        
        if not os.path.exists(log_file):
            print(f"\n📊 No history for {username}")
            return
        
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        if not logs:
            print(f"\n📊 No attempts yet")
            return
        
        total = len(logs)
        successful = sum(1 for log in logs if log['success'])
        failed = total - successful
        
        print(f"\n{'='*70}")
        print(f"📊 STATISTICS - {username}")
        print(f"{'='*70}")
        print(f"Total:      {total}")
        print(f"✅ Success: {successful} ({successful/total*100:.1f}%)")
        print(f"❌ Failed:  {failed} ({failed/total*100:.1f}%)")
        print(f"{'='*70}")
        
        print(f"\n📝 Recent Attempts:")
        for log in logs[-10:]:
            ts = datetime.fromisoformat(log['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            status = "✅" if log['success'] else "❌"
            passed = log['checks_passed']
            combined = log['combined_score'] * 100
            print(f"{ts} | {status} | {passed}/4 | {combined:.1f}%")


def main():
    """Main function"""
    print("="*70)
    print("🎙️  VOICE AUTHENTICATION SYSTEM V5 - POLISHED")
    print("Industry-Standard Features & Calibrated Thresholds")
    print("="*70)
    
    auth = VoiceAuthenticatorV5()
    
    while True:
        print("\n" + "="*70)
        print("MENU:")
        print("1. Register new user")
        print("2. Authenticate user")
        print("3. List users")
        print("4. View statistics")
        print("5. Exit")
        print("="*70)
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == '1':
            username = input("\nUsername: ").strip()
            if username:
                auth.register_user(username)
            else:
                print("❌ Username required!")
        
        elif choice == '2':
            username = input("\nUsername: ").strip()
            if username:
                auth.authenticate_user(username)
            else:
                print("❌ Username required!")
        
        elif choice == '3':
            users = auth.list_users()
            if users:
                print(f"\n{'='*70}")
                print("📋 REGISTERED USERS")
                print(f"{'='*70}")
                for user in users:
                    print(f"\n👤 {user['username']}")
                    print(f"   Registered: {user['registered'][:10]}")
                    print(f"   Attempts: {user['auth_attempts']} ({user['successful_auths']} successful)")
            else:
                print("\n📋 No users yet")
        
        elif choice == '4':
            username = input("\nUsername: ").strip()
            if username:
                auth.show_statistics(username)
            else:
                print("❌ Username required!")
        
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()