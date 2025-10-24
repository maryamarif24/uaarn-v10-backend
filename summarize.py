from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig
from agents import AsyncOpenAI
from dotenv import load_dotenv
import os

# Create router
router = APIRouter()

# Load environment variables
load_dotenv()

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

# Setup external client
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Define model and config
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
config = RunConfig(model=model)

# Request and response models
class SummarizeRequest(BaseModel):
    content: str

class SummarizeResponse(BaseModel):
    summary: str

# Summarization Agent
summarizer_agent = Agent(
    name="Summarizer Agent",
    instructions="""
You are a summarization agent that can handle text, transcripts, or lecture notes.
Generate a clear, concise, and readable summary that captures the key ideas.
If timestamps are present (e.g., [00:03:15]), include them properly with their summarized context.
""",
)

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """
    Summarize text using OpenAI Agents SDK and Gemini model.
    """
    try:
        result = await Runner.run(
            summarizer_agent,
            f"Summarize this content:\n\n{request.content}",
            run_config=config
        )

        summary = getattr(result, "final_output", str(result))
        return SummarizeResponse(summary=summary.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
