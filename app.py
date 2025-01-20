from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
import speech_recognition as sr
from google import genai
from pydub import AudioSegment
import logging

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY environment variable not set")

# Initialize the GenAI client
client = genai.Client(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS, HTML, etc.) from the 'static' folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route to serve the HTML page (index.html)
@app.get("/", response_class=HTMLResponse)
async def get_html():
    try:
        with open(os.path.join("static", "index.html")) as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="HTML file not found")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Endpoint for speech-to-text and content generation
@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    temp_input_path = f"temp_{file.filename}"
    temp_wav_path = "temp_audio.wav"

    try:
        # Check file type
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="The uploaded file is not an audio file")

        # Save uploaded file temporarily
        with open(temp_input_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Convert to WAV format
        try:
            audio = AudioSegment.from_file(temp_input_path)
            audio.export(temp_wav_path, format="wav")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error converting audio file: {e}")

        # Process audio with speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = recognizer.record(source)

        try:
            recognized_text = recognizer.recognize_google(audio_data)
            logger.info(f"Recognized text: {recognized_text}")
        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error with Google Speech Recognition service: {e}"
            )

        # Generate response using GenAI
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=recognized_text
            )
            logger.info(f"Generated response: {response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating response from GenAI: {e}")

        return {"recognized_text": recognized_text, "generated_response": response.text}

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    finally:
        # Cleanup temporary files
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            logger.info("Temporary audio files cleaned up")
