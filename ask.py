import os
import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    input_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered
)

# Router
router = APIRouter(prefix="/ask", tags=["Ask"])

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables")

# Gemini client (OpenAI-style wrapper)
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Global RunConfig
config = RunConfig(
    model=model,
    tracing_disabled=True
)

# Token system
CREDITS = {}
DEFAULT_CREDIT_TOKENS = int(os.getenv("DEFAULT_CREDIT_TOKENS", "100000"))
USER_NAMES = {}

def get_user_id(header_user_id: Optional[str]) -> str:
    return header_user_id or "anonymous"

def get_user_name(user_id: str, header_user_name: Optional[str]) -> str:
    if header_user_name:
        USER_NAMES[user_id] = header_user_name
    return USER_NAMES.get(user_id, "there")

def ensure_user_in_credits(user_id: str):
    if user_id not in CREDITS:
        CREDITS[user_id] = {
            "tokens_left": DEFAULT_CREDIT_TOKENS,
            "last_reset": datetime.utcnow()
        }

def deduct_tokens(user_id: str, tokens: int) -> bool:
    ensure_user_in_credits(user_id)
    if CREDITS[user_id]["tokens_left"] >= tokens:
        CREDITS[user_id]["tokens_left"] -= tokens
        return True
    return False

# Output cleaner
def format_response(text: str) -> str:
    text = re.sub(r'(?<=\d\.)\s+', ' ', text)
    text = re.sub(r'(?<=\d\))\s+', ' ', text)
    text = re.sub(r'(\d+\.\s+)', r'\n\1', text)
    text = re.sub(r'(\d+\)\s+)', r'\n\1', text)
    text = re.sub(r'([\-•]\s+)', r'\n\1', text)
    text = text.replace("\\n", "\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# Request / Response models
class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9

class ChatResponse(BaseModel):
    reply: str
    detected_language: Optional[str] = None
    redirected_to: Optional[str] = None
    tokens_used_estimate: Optional[int] = None
    tokens_remaining: Optional[int] = None

# Guardrail
@input_guardrail
async def study_guardrail(ctx, agent: Agent, user_input: str | list) -> GuardrailFunctionOutput:
    text = user_input if isinstance(user_input, str) else " ".join(
        item["content"] for item in user_input
    )

    study_keywords = [
        "study", "explain", "summarize", "lecture", "homework",
        "exercise", "math", "physics", "chemistry", "biology",
        "history", "essay", "exam", "concept", "cells",
        "what", "why", "when", "how", "where", "solve", "tell"
    ]

    if not any(kw in text.lower() for kw in study_keywords):
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

# Agent creator
def create_study_agent():
    return Agent(
        name="UAARN Study Agent",
        instructions="""
You are UAARN's Study Agent.

Rules:
1. Only answer study-related questions.
2. Use structured Markdown.
3. Each point starts on a new line.
4. Use proper headings.
""",
        input_guardrails=[study_guardrail]
    )

# Main endpoint
@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    x_user_id: Optional[str] = Header(None),
    x_user_name: Optional[str] = Header(None)
):
    user_id = get_user_id(x_user_id)
    user_name = get_user_name(user_id, x_user_name)
    ensure_user_in_credits(user_id)

    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    # Greetings shortcut
    greetings = ["hi", "hello", "hey", "salam", "assalam", "assalamu", "assalamualaikum"]
    if any(text.lower().startswith(g) for g in greetings):
        return ChatResponse(
            reply=f"👋 Hello {user_name}! How can I help you study today?",
            tokens_used_estimate=0,
            tokens_remaining=CREDITS[user_id]["tokens_left"]
        )

    # Token estimate
    max_tokens = min(1024, req.max_tokens or 512)
    estimated_tokens = max(1, int(len(text) / 4)) + max_tokens

    if not deduct_tokens(user_id, estimated_tokens):
        raise HTTPException(status_code=402, detail="Insufficient tokens")

    agent = create_study_agent()
    user_prompt = f"User question: {text}"

    try:
        result = await Runner.run(agent, user_prompt, run_config=config)
        reply_text = getattr(result, "final_output", "No response from agent")
        formatted = format_response(reply_text)

    except InputGuardrailTripwireTriggered:
        return ChatResponse(
            reply="⚠️ Please ask only study-related questions.",
            tokens_used_estimate=0,
            tokens_remaining=CREDITS[user_id]["tokens_left"]
        )

    except Exception as e:
        CREDITS[user_id]["tokens_left"] += estimated_tokens
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        reply=formatted,
        tokens_used_estimate=estimated_tokens,
        tokens_remaining=CREDITS[user_id]["tokens_left"]
    )
