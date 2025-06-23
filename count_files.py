import glob
import os

ravdess_path = 'C:/MindHaven/archive'  # Updated path
audio_files = glob.glob(os.path.join(ravdess_path, 'Actor_*/*.wav'))
print(f"Total audio files: {len(audio_files)}")
for actor_folder in glob.glob(os.path.join(ravdess_path, 'Actor_*')):
    actor_files = glob.glob(os.path.join(actor_folder, '*.wav'))
    print(f"{os.path.basename(actor_folder)}: {len(actor_files)} files")