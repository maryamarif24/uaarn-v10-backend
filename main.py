import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import uvicorn
import os

from quiz import router as quiz_router
from summarize import router as summarize_router
from ask import router as ask_router

from agent import create_career_mentor, config
from agents import Runner
from utils.tts_pdf import text_to_speech_bytes, text_to_pdf_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="UAARN + AI Career Mentor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(quiz_router)
app.include_router(summarize_router)
app.include_router(ask_router)



class ChatRequest(BaseModel):
    message: str
    user_id: str | None = None

class ChatResponse(BaseModel):
    reply: str

class TTSRequest(BaseModel):
    text: str

@app.post("/careerapi/chat", response_model=ChatResponse)
async def career_chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    agent = create_career_mentor()
    try:
        logger.info(f"Career chat: {req.message[:100]}")
        result = await Runner.run(agent, req.message, run_config=config)
        reply = result.final_output or "I'm your AI Career Mentor! How can I help you today?"
        return ChatResponse(reply=reply)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail="Mentor is busy. Try again!")

@app.post("/careerapi/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    if len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="CV too short")

    agent = create_career_mentor()
    prompt = f"Analyze this CV and give detailed feedback:\n\n{text[:30000]}"
    result = await Runner.run(agent, prompt, run_config=config)
    return {"analysis": result.final_output}

@app.post("/careerapi/tts")
async def tts(req: TTSRequest):
    audio_bytes = text_to_speech_bytes(req.text)
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=career-advice.mp3"}
    )

@app.post("/careerapi/download/txt")
async def download_txt(req: TTSRequest):
    return StreamingResponse(
        io.BytesIO(req.text.encode('utf-8')),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=career-roadmap.txt"}
    )

@app.post("/careerapi/download/pdf")
async def download_pdf(req: TTSRequest):
    pdf_bytes = text_to_pdf_bytes(req.text)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=career-roadmap.pdf"}
    )


@app.get("/")
def root():
    return {"message": "UAARN Backend Running"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)