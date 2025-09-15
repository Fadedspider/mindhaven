from flask import Flask, render_template, request, jsonify
import whisper
import librosa
import soundfile as sf
import numpy as np
from gtts import gTTS  # Fallback TTS
import os
import tempfile
import sqlite3
from datetime import datetime
import google.auth  # For potential future use, but not needed here

# NEW: Import for Gemini
import google.generativeai as genai

# NEW: Imports for pyttsx3 emotional TTS
import pyttsx3
from pydub import AudioSegment
from pydub.effects import normalize

app = Flask(__name__)

# Configure Gemini with API key from env var (secure) - FIXED: Use 'GEMINI_API_KEY' as the var name
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required. Set it and restart.")
genai.configure(api_key=api_key)

# Lazy load Whisper model
model = None

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
                 audio_path TEXT
                 )''')
    try:
        c.execute("ALTER TABLE interactions ADD COLUMN audio_path TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

init_db()

# Function to generate dynamic response with Gemini (fine-tuned)
def generate_dynamic_response(transcript, mood):
    gen_model = genai.GenerativeModel('gemini-1.5-flash')  # Stable model as of Sep 2025
    
    # Mood-based tone adaptation
    tone_map = {
        "sadness": "gentle and reassuring",
        "stress": "calm and supportive",
        "excited": "upbeat and enthusiastic",
        "mindful": "serene and reflective",
        "neutral": "warm and friendly",
        "thinking": "curious and thoughtful",
        "comforting": "nurturing and safe"
    }
    tone = tone_map.get(mood, "warm and friendly")
    
    prompt = f"""
    You are Seren, an AI-powered, confidential mental wellness companion for youth aged 13-25. 
    Your goal: Provide empathetic, non-judgmental support; reduce stigma by normalizing feelings; 
    guide toward help without giving medical advice. Always end with: "Remember, this isn't a substitute for professional therapy."
    
    User transcript: "{transcript.strip()}"
    Detected mood: {mood}
    
    Respond in a {tone} tone (use casual, relatable language like a trusted friend). 
    - Limit to 2-3 sentences (50-70 words max).
    - For negative moods (stress/sadness), suggest one simple coping strategy (e.g., 4-7-8 breathing), encourage reaching out to a trusted adult or hotline, and normalize: "It's okay to feel this way—many teens do."
    - For positive moods (excited/mindful), amplify the good vibes and ask an open question like "What's making you feel that way?"
    - If crisis indicators (e.g., self-harm, hopelessness), lead with: "I'm really worried—please text HOME to 741741 (US Crisis Text Line) right now for immediate support."
    - Stay confidential: No questions about personal details.
    """
    
    try:
        response = gen_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,  # Balanced: Empathetic yet consistent
                top_p=0.9,        # Stay on-topic
                max_output_tokens=100  # Enforce brevity
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        fallback_responses = {
            "stress": "Hey, stress is tough, but you're tougher. Try this: Breathe in for 4, hold 7, out 8. Talk to someone you trust—it's a sign of strength. Remember, this isn't therapy.",
            "sadness": "It's totally valid to feel down sometimes. You're not alone in this. Reach out to a friend or hotline if needed. Remember, this isn't therapy.",
            "excited": "That's awesome—love that energy! What sparked it? Remember, this isn't therapy.",
            # Add more as needed
        }
        return fallback_responses.get(mood, "I'm here listening. What's on your mind? Remember, this isn't therapy.")

# Improved mood detection
def detect_mood(text, audio_path=None):
    text = text.lower().strip()
    text_mood = None
    tone_mood = None

    # Expanded text-based keywords
    stress_keywords = ["stressed", "anxious", "overwhelmed", "worried", "panic"]
    sadness_keywords = ["sad", "down", "lonely", "hopeless", "cry"]
    thinking_keywords = ["wonder", "think", "curious", "confused"]
    comforting_keywords = ["comfort", "hug", "safe", "hold"]
    excited_keywords = ["excited", "happy", "thrilled", "joyful", "amazing"]
    mindful_keywords = ["calm", "peace", "mindful", "relax", "zen"]

    if any(word in text for word in stress_keywords):
        text_mood = "stress"
    elif any(word in text for word in sadness_keywords):
        text_mood = "sadness"
    elif any(word in text for word in thinking_keywords):
        text_mood = "thinking"
    elif any(word in text for word in comforting_keywords):
        text_mood = "comforting"
    elif any(word in text for word in excited_keywords):
        text_mood = "excited"
    elif any(word in text for word in mindful_keywords):
        text_mood = "mindful"
    else:
        text_mood = "neutral"

    # Tone-based (refined thresholds for realism)
    if audio_path:
        try:
            y, sr = librosa.load(audio_path)
            pitch = librosa.pitch_tuning(y)
            tempo, _ = librosa.beat.tempo(y, sr=sr)
            energy = np.mean(librosa.feature.rms(y=y))

            if pitch > 0.2 and tempo > 110 and energy > 0.03:  # Adjusted for excitement
                tone_mood = "excited"
            elif pitch < -0.2 and tempo < 90 and energy < 0.015:  # Adjusted for sadness
                tone_mood = "sadness"
            elif tempo < 70 and energy < 0.01:
                tone_mood = "mindful"
            elif pitch > 0.1 and tempo > 95 and energy > 0.02:  # Adjusted for stress
                tone_mood = "stress"
            elif tempo > 90 and energy > 0.025:
                tone_mood = "comforting"
            else:
                tone_mood = "neutral"
        except Exception as e:
            print(f"Tone analysis error: {e}")
            tone_mood = None

    # Combine (prioritize text for conflicts)
    final_mood = text_mood if text_mood else tone_mood
    if tone_mood and text_mood and tone_mood != text_mood:
        # Simple override logic: Tone influences but text dominates negatives
        if text_mood in ["sadness", "stress"]:
            final_mood = text_mood

    return final_mood

# Log interaction
def log_interaction(input_type, transcript, mood, response, audio_path=None):
    conn = sqlite3.connect('mindheaven.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('''INSERT INTO interactions (timestamp, input_type, transcript, mood, response, audio_path)
                 VALUES (?, ?, ?, ?, ?, ?)''', (timestamp, input_type, transcript, mood, response, audio_path))
    conn.commit()
    conn.close()

# Emotional TTS with pyttsx3 (free/offline, works on Python 3.12) + gTTS fallback
def text_to_speech(text, mood="neutral"):
    # Mood-based params (emulates emotion)
    param_map = {
        "sadness": {'rate': 120, 'pitch': -0.2, 'volume': 0.8},  # Slower, lower pitch, softer
        "stress": {'rate': 140, 'pitch': 0.0, 'volume': 1.0},    # Steady
        "excited": {'rate': 180, 'pitch': 0.3, 'volume': 1.0},   # Faster, higher pitch
        "mindful": {'rate': 110, 'pitch': -0.1, 'volume': 0.9},  # Calm, relaxed
        "neutral": {'rate': 150, 'pitch': 0.0, 'volume': 1.0}    # Default
    }
    params = param_map.get(mood, param_map["neutral"])
    
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        # Select a natural voice (e.g., first female-like; adjust index for your OS)
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.id.lower():  # Windows example
                engine.setProperty('voice', voice.id)
                break
        
        engine.setProperty('rate', params['rate'])      # Speed: 100-300 wpm
        engine.setProperty('volume', params['volume'])  # 0.0-1.0
        
        # Save to temp WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            engine.save_to_file(text, fp.name)
            engine.runAndWait()
        
        # Load and apply pitch shift (simple semitone adjustment)
        audio = AudioSegment.from_wav(fp.name)
        octaves = params['pitch']  # e.g., -0.2 = lower pitch
        new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
        audio_with_pitch = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(22050)
        audio_with_pitch = normalize(audio_with_pitch)  # Clean up
        
        # Export to MP3
        mp3_path = fp.name.replace('.wav', '.mp3')
        audio_with_pitch.export(mp3_path, format="mp3")
        os.unlink(fp.name)  # Clean up
        
        print(f"pyttsx3 generated emotional audio for mood: {mood}")
        return mp3_path
        
    except Exception as e:
        print(f"pyttsx3 error: {e}; falling back to gTTS")
    
    # Fallback to gTTS
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        slow = (mood in ["sadness", "mindful"])
        tts = gTTS(text=text, lang='en', slow=slow)
        tts.save(fp.name)
        return fp.name

@app.route("/interaction")
def interaction():
    return render_template("interactions.html")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    global model  # Lazy load
    try:
        if model is None:
            model = whisper.load_model("medium")
            print("Whisper model loaded.")

        audio_path = None
        if 'audio' not in request.files:
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({"error": "No audio or text provided"}), 400
            user_input = data['text'].strip()
            transcript = user_input
            mood = detect_mood(transcript)
            input_type = "text"
        else:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({"error": "No audio file selected"}), 400
            audio_path = f"audio_logs/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            audio_file.save(audio_path)
            print(f"Audio saved: {audio_path}, Size: {os.path.getsize(audio_path)} bytes")
            result = model.transcribe(audio_path)
            transcript = result["text"].strip()
            print(f"Transcribed: {transcript}")
            mood = detect_mood(transcript, audio_path=audio_path)
            print(f"Detected mood: {mood}")
            input_type = "audio"

        response_text = generate_dynamic_response(transcript, mood)
        print(f"Generated response: {response_text}")

        audio_file = text_to_speech(response_text, mood)  # Emotional TTS
        static_audio_path = os.path.join(app.static_folder or "static", os.path.basename(audio_file))
        os.makedirs(os.path.dirname(static_audio_path), exist_ok=True)
        os.rename(audio_file, static_audio_path)

        log_interaction(input_type, transcript, mood, response_text, audio_path)

        return jsonify({
            "mood": mood,
            "response": response_text,
            "audio": f"/static/{os.path.basename(audio_file)}"
        })

    except Exception as e:
        print(f"Process error: {str(e)}")
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
