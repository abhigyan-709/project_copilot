from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import speech_recognition as sr
from google import genai
from pydub import AudioSegment

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

# Endpoint for speech-to-text and content generation
# @app.post("/process-audio")
# async def process_audio(file: UploadFile = File(...)):
#     temp_input_path = f"temp_{file.filename}"
#     with open(temp_input_path, "wb") as temp_file:
#         temp_file.write(await file.read())

#     temp_wav_path = "temp_audio.wav"
#     try:
#         audio = AudioSegment.from_file(temp_input_path)
#         audio.export(temp_wav_path, format="wav")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error converting audio file: {e}")
#     finally:
#         os.remove(temp_input_path)

#     recognizer = sr.Recognizer()
#     try:
#         with sr.AudioFile(temp_wav_path) as source:
#             audio_data = recognizer.record(source)
#         recognized_text = recognizer.recognize_google(audio_data)
#     except sr.UnknownValueError:
#         raise HTTPException(status_code=400, detail="Speech Recognition could not understand the audio")
#     except sr.RequestError as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error with Google Speech Recognition service: {e}"
#         )
#     finally:
#         os.remove(temp_wav_path)

#     try:
#         response = client.models.generate_content(
#             model="gemini-2.0-flash-exp",
#             contents=recognized_text
#         )
#         return {"recognized_text": recognized_text, "generated_response": response.text}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating response from GenAI: {e}")

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    temp_input_path = f"temp_{file.filename}"
    temp_wav_path = "temp_audio.wav"

    try:
        # Save uploaded file temporarily
        with open(temp_input_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Convert to WAV format
        audio = AudioSegment.from_file(temp_input_path)
        audio.export(temp_wav_path, format="wav")

        # Process audio with speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = recognizer.record(source)
        recognized_text = recognizer.recognize_google(audio_data)

        # Generate response using GenAI
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=recognized_text
        )
        return {"recognized_text": recognized_text, "generated_response": response.text}

    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error with Google Speech Recognition service: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    finally:
        # Cleanup temporary files
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

