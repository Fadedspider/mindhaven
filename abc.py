import pandas as pd
df = pd.read_csv('audio_mood_data.csv')
print(f"Number of columns: {len(df.columns)}")
print(f"Number of samples: {len(df)}")
print(df.head())
print(df.describe())