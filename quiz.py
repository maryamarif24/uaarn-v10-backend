import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import AsyncGenerator

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    RunConfig
)

# Create router instead of FastAPI app
router = APIRouter(prefix="/quiz", tags=["Quiz"])

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env file")

# External Gemini client setup
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

# Run configuration
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

class QuizRequest(BaseModel):
    topic: str

# Define AI Agent
agent = Agent(
    name="QuizAgent",
    instructions=(
        "You are a quiz generator AI. Generate multiple-choice questions for the given topic. "
        "Each question must have 4 options and one correct answer. "
        "Respond ONLY in valid JSON format like this:\n\n"
        "["
        "{\"question\": \"What is Python?\", \"options\": [\"Snake\", \"Language\", \"Game\", \"OS\"], \"answer\": \"Language\"},"
        "{\"question\": \"2+2?\", \"options\": [\"1\", \"2\", \"3\", \"4\"], \"answer\": \"4\"}"
        "]"
    ),
)

# ---------------------------
# STREAMING QUIZ ENDPOINT
# ---------------------------
@router.post("/")
async def generate_quiz(req: QuizRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            async for event in Runner.stream(agent, f"Generate quiz on {topic}", run_config=config):
                chunk = getattr(event, "output_text", None)
                if chunk:
                    yield chunk
        except Exception as e:
            yield f"\n[ERROR] {str(e)}"

    return StreamingResponse(event_stream(), media_type="text/plain")

# Optional test route
@router.get("/")
def root():
    return {"message": "✅ Quiz router is working!"}
