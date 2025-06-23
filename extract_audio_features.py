import librosa
import numpy as np

def extract_audio_features(audio_path):
    try:
        # Load audio
        y, sr = librosa.load(audio_path)

        # Extract features
        # 1. MFCCs (13 coefficients) - captures timbre and tonal characteristics
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        print(f"MFCCs shape: {mfccs.shape}")  # Debug: Should be (13,)

        # 2. Spectral Contrast (7 bands) - captures the difference between peaks and valleys
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        print(f"Spectral Contrast shape: {spectral_contrast.shape}")  # Debug: Should be (7,)

        # 3. Pitch (refined)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        # 4. Energy (RMS)
        energy = np.mean(librosa.feature.rms(y=y))

        # 5. Tempo (fixed for librosa 0.10.0+)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)  # Compute onset envelope
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]  # Extract tempo

        # 6. Zero Crossing Rate - indicates noisiness/speech clarity
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # 7. Spectral Rolloff - frequency below which 85% of the spectral energy lies
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # 8. Mel Spectrogram Statistics - captures energy distribution across frequency bands
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel_spec)
        mel_var = np.var(mel_spec)

        # Combine features into a single array
        features = np.concatenate([
            mfccs,                  # 13 features
            spectral_contrast,      # 7 features
            [pitch, energy, tempo, zcr],  # 4 features
            [rolloff, mel_mean, mel_var]  # 3 features
        ])

        # Debug: Print the number of features
        print(f"Extracted features shape: {features.shape}")  # Should be (27,)

        return features  # Total: 13 + 7 + 4 + 3 = 27 features

    except Exception as e:
        print(f"Error extracting features from {audio_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    audio_path = "audio_logs/recording_20250429_004158.wav"  # Use a sample WAV file
    features = extract_audio_features(audio_path)
    if features is not None:
        print("Extracted Features:", features)
        print("Feature Length:", len(features))