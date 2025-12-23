# MindHaven - Mental Health Companion Chat App


**MindHaven** is a web-based mental health companion application designed to provide empathetic, emotionally aware responses to users. It supports both **text** and **voice/audio** inputs, analyzes the user's emotional state using a combination of audio-based machine learning models and text sentiment analysis, and generates supportive responses using Google's Gemini AI.

The app aims to offer a safe, non-judgmental space for users to express their feelings and receive understanding, compassionate replies.

## Features

- **Dual Input Modes**: Users can type messages or record voice/audio inputs.
- **Audio Sentiment Analysis**: 
  - Converts audio to WAV format.
  - Extracts acoustic features (MFCC, chroma, mel-spectrogram, etc.).
  - Uses an ensemble of SVM, XGBoost, and Random Forest classifiers for emotion detection.
- **Text Sentiment Analysis**: 
  - Transcribes audio inputs.
  - Checks for emotion-related keywords.
  - Performs sentiment analysis on transcript if keywords are present; otherwise relies on audio model.
- **Empathetic Responses**: Powered by Google Gemini API for contextually appropriate, emotionally intelligent replies.
- **Secure Authentication**: Google Sign-In via Firebase Authentication.
- **Data Storage**: User inputs and chat history stored locally in SQLite (can be extended to cloud).
- **Text-to-Speech Support**: Responses can be read aloud for better accessibility.
- **Responsive UI**: Clean, calming chat interface built with HTML, CSS, and JavaScript.

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript (with Bootstrap for styling)
- **Authentication & Storage**: Firebase (Google Sign-In and optional Firestore)
- **AI Response Generation**: Google Gemini API
- **Audio Processing**: 
  - `pydub` (audio conversion)
  - `librosa` (feature extraction)
  - `speechrecognition` or Whisper (transcription)
- **Machine Learning Models**: 
  - Scikit-learn (SVM, Random Forest)
  - XGBoost
  - Pickle-serialized ensemble model (`train_audio_model.pkl`)
- **Database**: SQLite
- **Other Libraries**: numpy, pandas, etc.

## Project Pipeline

1. User submits **text** or **audio** input via the web interface.
2. Input is saved to SQLite database.
3. If audio:
   - Converted to `.wav` format.
   - Features extracted using librosa.
   - Transcript generated.
4. Sentiment Analysis:
   - If transcript contains emotion-related keywords → analyze text sentiment.
   - Otherwise → feed audio features to the pre-trained ensemble model (SVM + XGBoost + Random Forest).
5. Detected sentiment/emotion is passed as context to the **Gemini API**.
6. Gemini generates an empathetic, supportive response.
7. Response is displayed and optionally converted to speech.

## Installation & Setup

### Prerequisites

- Python 3.8+
- Firebase project with Google Authentication enabled
- Google Gemini API key
- FFmpeg (for audio processing)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MindHaven.git
   cd MindHaven
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (create a `.env` file):
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. Configure Firebase:
   - Place your Firebase config in `app.py` or use environment variables.
   - Enable Google Sign-In in Firebase Console.

6. Run the application:
   ```bash
   python app.py
   ```

7. Open your browser and go to `http://127.0.0.1:5000`


---

**Made with ❤️ for mental well-being**

Feel free to customize the repository name, add screenshots, or update the banner image link once you have visuals ready! Let me know if you want a `requirements.txt` example or badges (like Python version, license, etc.) added.
