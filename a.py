import pandas as pd

# Load both CSVs
speech_df = pd.read_csv('audio_mood_data.csv')
song_df = pd.read_csv('music_mood_data.csv')



# Concatenate
merged_df = pd.concat([speech_df, song_df], ignore_index=True)

# Shuffle the merged DataFrame (recommended)
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new CSV
merged_df.to_csv('ravdess_combined_features.csv', index=False)
