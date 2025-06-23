import pandas as pd
import os
import glob
from extract_audio_features import extract_audio_features

# Define RAVDESS path and emotion-to-mood mapping
ravdess_path = 'C:/MindHaven/archive'
emotion_map = {
    '01': 'neutral',  # neutral
    '02': 'mindful',  # calm
    '03': 'excited',  # happy
    '04': 'sadness',  # sad
    '05': 'stress',   # angry
    '06': 'stress',   # fearful
    '07': 'sadness',  # disgust
    '08': 'excited'   # surprise
}

# Collect RAVDESS audio files and their labels
ravdess_data = []
total_files = 0
for actor_folder in glob.glob(os.path.join(ravdess_path, 'Actor_*')):
    actor_files = glob.glob(os.path.join(actor_folder, '*.wav'))
    total_files += len(actor_files)
    for audio_file in actor_files:
        filename = os.path.basename(audio_file)
        parts = filename.split('-')
        emotion_code = parts[2]  # e.g., '01' for neutral
        mood = emotion_map.get(emotion_code, None)
        if mood:
            ravdess_data.append({
                'audio_path': audio_file,
                'mood': mood
            })

print(f"Total audio files found: {total_files}")
print(f"Total valid audio files (after mood mapping): {len(ravdess_data)}")

# Convert to DataFrame
df_ravdess = pd.DataFrame(ravdess_data)

# Filter supported moods
supported_moods = ['stress', 'sadness', 'neutral', 'excited', 'mindful']
df_ravdess = df_ravdess[df_ravdess['mood'].isin(supported_moods)]

# Extract features for each audio file
features_list = []
labels = []
skipped_files = 0
for idx, row in df_ravdess.iterrows():
    audio_path = row['audio_path']
    features = extract_audio_features(audio_path)
    if features is not None:
        features_list.append(features)
        labels.append(row['mood'])
    else:
        print(f"Skipping {audio_path} due to feature extraction error")
        skipped_files += 1

print(f"Total files processed: {len(df_ravdess)}")
print(f"Files skipped due to feature extraction errors: {skipped_files}")
print(f"Files with successful feature extraction: {len(features_list)}")

# Save to CSV
if features_list:
    feature_df = pd.DataFrame(features_list, columns=[f'feature_{i}' for i in range(len(features_list[0]))])
    feature_df['mood'] = labels
    feature_df.to_csv('audio_mood_data.csv', index=False)
    print(f"Saved {len(feature_df)} samples to audio_mood_data.csv")
else:
    print("No valid audio samples found")