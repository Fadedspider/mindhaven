from flask import Flask, render_template, request, jsonify
import whisper
import librosa
import soundfile as sf
import numpy as np
from gtts import gTTS
import os
import tempfile
import sqlite3
from datetime import datetime

app = Flask(__name__)

@app.route("/interaction")
def interaction():
    return render_template("interactions.html")
    
# Load Whisper model
model = whisper.load_model("medium")  # Use "base" if needed

# SQLite database setup
def init_db():
    
    conn = sqlite3.connect('mindheaven.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 timestamp TEXT NOT NULL,
                 input_type TEXT NOT NULL,
                 transcript TEXT NOT NULL,
                 mood TEXT NOT NULL,
                 response TEXT NOT NULL,
                 audio_path TEXT  -- New column for audio file path
                 )''')
    conn.commit()
    conn.close()

# Initialize the database on startup
def init_db():
    conn = sqlite3.connect('mindheaven.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 timestamp TEXT NOT NULL,
                 input_type TEXT NOT NULL,
                 transcript TEXT NOT NULL,
                 mood TEXT NOT NULL,
                 response TEXT NOT NULL,
                 audio_path TEXT  -- New column for audio file path
                 )''')
    try:
        c.execute("ALTER TABLE interactions ADD COLUMN audio_path TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    conn.close()

init_db()
# Predefined responses for different moods
responses = {
    "stress": ["You’re stronger than this storm.", "Let’s breathe together.", "Take it slow—you’ve got this!"],
    "sadness": ["You’re not alone—Seren is here.", "A new day will bring light.", "Hold on—better moments are coming."],
    "neutral": ["You’re doing great—keep it up!", "Seren chirps for you!", "You’re in a good place today!"],
    "thinking": ["Let’s ponder together.", "Seren is curious too!"],
    "comforting": ["Seren is here to hug you.", "You’re safe with me."],
    "excited": ["Wow, that’s amazing!", "Seren is thrilled for you!"],
    "mindful": ["Let’s take a deep breath.", "Feel the peace with Seren."]
}

# Function to detect mood using text and tone
def detect_mood(text, audio_path=None):
    text = text.lower()
    text_mood = None
    tone_mood = None

    # Text-based mood detection
    if any(word in text for word in ["stressed", "anxious", "overwhelmed"]):
        text_mood = "stress"
    elif any(word in text for word in ["sad", "down", "lonely"]):
        text_mood = "sadness"
    elif any(word in text for word in ["wonder", "think", "curious"]):
        text_mood = "thinking"
    elif any(word in text for word in ["comfort", "hug", "safe"]):
        text_mood = "comforting"
    elif any(word in text for word in ["excited", "happy", "thrilled"]):
        text_mood = "excited"
    elif any(word in text for word in ["calm", "peace", "mindful"]):
        text_mood = "mindful"
    else:
        text_mood = "neutral"

    # Tone-based mood detection (if audio is provided)
    if audio_path:
        try:
            y, sr = librosa.load(audio_path)
            pitch = librosa.pitch_tuning(y)
            tempo, _ = librosa.beat.tempo(y, sr=sr)
            energy = np.mean(librosa.feature.rms(y=y))

            if pitch > 0.5 and tempo > 120 and energy > 0.05:
                tone_mood = "excited"
            elif pitch < -0.5 and tempo < 80 and energy < 0.02:
                tone_mood = "sadness"
            elif tempo < 60 and energy < 0.01:
                tone_mood = "mindful"
            elif pitch > 0.3 and tempo > 100:
                tone_mood = "stress"
            elif tempo > 80 and energy > 0.03:
                tone_mood = "comforting"
            else:
                tone_mood = "neutral"
        except Exception as e:
            print(f"Tone analysis error: {e}")
            tone_mood = None

    # Combine text and tone
    final_mood = tone_mood if tone_mood else text_mood
    if text_mood == "sadness" and tone_mood in ["excited", "comforting"]:
        final_mood = "sadness"
    elif text_mood == "excited" and tone_mood in ["sadness", "mindful"]:
        final_mood = "excited"
    elif text_mood == "mindful" and tone_mood == "stress":
        final_mood = "mindful"

    return final_mood

# Function to log interaction to SQLite
def log_interaction(input_type, transcript, mood, response, audio_path=None):
    conn = sqlite3.connect('mindheaven.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('''INSERT INTO interactions (timestamp, input_type, transcript, mood, response, audio_path)
                 VALUES (?, ?, ?, ?, ?, ?)''', (timestamp, input_type, transcript, mood, response, audio_path))
    conn.commit()
    conn.close()

# Function to generate audio response
def text_to_speech(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(fp.name)
        return fp.name

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        audio_path = None
        if 'audio' not in request.files:
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({"error": "No audio or text provided"}), 400
            user_input = data['text']
            transcript = user_input
            mood = detect_mood(transcript)
            input_type = "text"
        else:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({"error": "No audio file selected"}), 400
            # Save audio permanently
            audio_path = f"audio_logs/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            audio_file.save(audio_path)
            print(f"Audio file saved at: {audio_path}, Size: {os.path.getsize(audio_path)} bytes")
            result = model.transcribe(audio_path)
            transcript = result["text"]
            print(f"Transcribed text: {transcript}")
            mood = detect_mood(transcript, audio_path=audio_path)
            print(f"Detected mood: {mood}")
            input_type = "audio"

        response_text = np.random.choice(responses[mood])
        audio_file = text_to_speech(response_text)
        static_audio_path = os.path.join(app.static_folder, os.path.basename(audio_file))
        os.rename(audio_file, static_audio_path)

        log_interaction(input_type, transcript, mood, response_text, audio_path)

        response = {
            "mood": mood,
            "response": response_text,
            "audio": f"/static/{os.path.basename(audio_file)}"
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/interactions')
def view_interactions():
    conn = sqlite3.connect('mindheaven.db')
    c = conn.cursor()
    c.execute("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    return render_template('interactions.html', interactions=rows)

if __name__ == '__main__':
    app.run(debug=True)