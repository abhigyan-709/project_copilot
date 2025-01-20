from google import genai
import os
from dotenv import load_dotenv
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
api_key = os.getenv("API_KEY")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(model='gemini-2.0-flash-exp', contents='How does AI work?')
print(response.text)
