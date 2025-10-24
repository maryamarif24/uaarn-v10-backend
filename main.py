import os
import re
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from quiz import router as quiz_router
from agents import Agent
from summarize import app as summarizer_router


from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    RunConfig,
    input_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered
)

app = FastAPI(title="UAARN AI Agents")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(quiz_router, prefix="/quiz")
app.include_router(summarizer_router, prefix="/summarize")


# ✅ Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

# ✅ Setup Gemini client
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ✅ FIXED: Model must receive the same `external_client`
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# ✅ RunConfig (only pass model, not model_provider)
config = RunConfig(
    model=model,
    tracing_disabled=True
)


# ✅ In-memory credit tracking
CREDITS = {}
DEFAULT_CREDIT_TOKENS = int(os.getenv("DEFAULT_CREDIT_TOKENS", "100000"))
USER_NAMES = {}

# ----------------------------------------------------------------------
# 🧠 Utility functions
# ----------------------------------------------------------------------

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

def format_response(text: str) -> str:
    """
    Cleans and enforces structured formatting from model output.
    - Adds newlines after list items.
    - Preserves Markdown formatting.
    """
    # Normalize bullets and numbers
    text = re.sub(r'(?<=\d\.)\s+', ' ', text)   # fix "1. something"
    text = re.sub(r'(?<=\d\))\s+', ' ', text)   # fix "1) something"

    # Add newline before each new list item
    text = re.sub(r'(\d+\.\s+)', r'\n\1', text)
    text = re.sub(r'(\d+\)\s+)', r'\n\1', text)
    text = re.sub(r'([\-•]\s+)', r'\n\1', text)

    # Replace escaped newlines
    text = text.replace("\\n", "\n")

    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip unnecessary spaces
    return text.strip()


# ----------------------------------------------------------------------
# 📥 Request/Response Models
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# 🛡️ Guardrail (content control)
# ----------------------------------------------------------------------

@input_guardrail
async def study_guardrail(ctx, agent: Agent, user_input: str | list) -> GuardrailFunctionOutput:
    text = user_input if isinstance(user_input, str) else " ".join(
        item["content"] for item in user_input
    )

    study_keywords = [
        "study", "explain", "summarize", "lecture", "homework", "exercise", "math",
        "what", "why", "when", "how", "where", "solve",
        "physics", "chemistry", "biology", "history", "essay", "exam", "concept", "cells"
    ]
    if not any(kw in text.lower() for kw in study_keywords):
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

# ----------------------------------------------------------------------
# 🤖 Agent definition
# ----------------------------------------------------------------------

def create_study_agent():
    return Agent(
        name="UAARN Study Agent",
        instructions="""
You are UAARN's Study Agent.

Your role:
- Explain study-related questions clearly and concisely.
- Always format your answers using structured Markdown.
- When listing steps, rules, examples, or explanations:
  1. Use **numbered lists** for sequences or procedures.
  2. Use **bulleted lists** for general points.
  3. Each new point or step must start on a **new line**.
  4. Never return all points in a single paragraph.
  5. Use headings and subheadings (like '### Definition:' or '### Example:') where appropriate.

Example output:
### Newton’s Laws of Motion
1. **First Law (Inertia):** An object remains at rest or uniform motion unless acted upon by a force.
2. **Second Law (F = ma):** Force equals mass times acceleration.
3. **Third Law (Action-Reaction):** Every action has an equal and opposite reaction.

Avoid any unrelated, harmful, or non-study topics.
""",
        input_guardrails=[study_guardrail]
    )

# ----------------------------------------------------------------------
# 💬 Chat Endpoint
# ----------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
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

    # Handle greetings
    greetings = ["hi", "hello", "hey", "salam", "assalam", "assalamu", "assalamualaikum"]
    if any(text.lower().startswith(g) for g in greetings):
        return ChatResponse(
            reply=f"👋 Hello {user_name}! How can I help you with your studies today?",
            tokens_used_estimate=0,
            tokens_remaining=CREDITS[user_id]["tokens_left"]
        )

    # Token management
    max_tokens = min(1024, req.max_tokens or 512)
    estimated_tokens = max(1, int(len(text) / 4)) + max_tokens

    if not deduct_tokens(user_id, estimated_tokens):
        raise HTTPException(status_code=402, detail="Insufficient tokens")

    agent = create_study_agent()
    user_prompt = f"User question: {text}"

    try:
        result = await Runner.run(agent, user_prompt, run_config=config)
        reply_text = getattr(result, "final_output", str(result))
        formatted = format_response(reply_text)

    except InputGuardrailTripwireTriggered:
        return ChatResponse(
            reply="⚠️ Please ask only study-related questions.",
            redirected_to=None,
            tokens_used_estimate=0,
            tokens_remaining=CREDITS[user_id]["tokens_left"]
        )
    except Exception as e:
        # Restore deducted tokens if error occurred
        CREDITS[user_id]["tokens_left"] += estimated_tokens
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return ChatResponse(
        reply=formatted,
        tokens_used_estimate=estimated_tokens,
        tokens_remaining=CREDITS[user_id]["tokens_left"]
    )

# ----------------------------------------------------------------------
# ✅ Health Check Endpoint
# ----------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "UAARN Backend is running successfully 🚀"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
