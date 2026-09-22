import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

class HealthCheck(BaseModel):
    status: str
    message: str

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Confirm the environment is set up correctly.",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=HealthCheck,
    ),
)

result: HealthCheck = response.parsed
print(result)