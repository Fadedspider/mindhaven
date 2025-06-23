import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Path to your RAVDESS speech audio files
DATA_PATH = 'C:/MindHaven/archive'  # Updated path

# Emotion mapping based on RAVDESS file naming
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path, max_pad_len=862):
    # Load audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # Extract Mel-spectrogram
    melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    log_melspec = librosa.power_to_db(melspec)
    # Pad or truncate to fixed length
    if log_melspec.shape[1] < max_pad_len:
        pad_width = max_pad_len - log_melspec.shape[1]
        log_melspec = np.pad(log_melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        log_melspec = log_melspec[:, :max_pad_len]
    return log_melspec

# Gather features and labels
features = []
labels = []

for file in os.listdir(DATA_PATH):
    if file.endswith('.wav'):
        emotion_code = file.split('-')[2]
        if emotion_code in emotion_map:
            label = emotion_map[emotion_code]
            file_path = os.path.join(DATA_PATH, file)
            try:
                melspec = extract_features(file_path)
                features.append(melspec)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file}: {e}")

features = np.array(features)
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels_categorical, test_size=0.2, random_state=42, stratify=labels_categorical
)

# Expand dims for CNN input
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 862, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(labels_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.3f}")

# Plot training history
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
