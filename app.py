import os
import json
import base64
import asyncio
import uvicorn
import edge_tts
from io import BytesIO
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    HTTPException)

load_dotenv()

GEMINI_key = os.getenv("GEMINI_API_KEY")
DEEPSEEK_key = os.getenv("DEEPSEEK_API_KEY")

GEMINI_model_name = "gemini-3.1-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_key)

DEEPSEEK_model_name = "deepseek-flash"
DEEPSEEK_translation_model_name = "deepseek-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
deepseek = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_key)

with open("reader_prompt.txt", "r", encoding="utf-8") as f:
    READER_system_prompt = f.read()

with open("summarizer_prompt.txt", "r", encoding="utf-8") as f:
    SUMMARIZER_system_prompt = f.read()

app = FastAPI(title="pathoGEN")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

VOICE_MAP = {
    "English": "en-US-JennyNeural",
    "Hindi": "hi-IN-SwaraNeural",
    "Marathi": "mr-IN-AarohiNeural",
}

# --- Helper functions ---

def findings(encoded_pdf):
    response = gemini.chat.completions.create(
        model=GEMINI_model_name,
        messages=[
            {
                "role": "system",
                "content": READER_system_prompt
            },
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{encoded_pdf}"}}]
            }
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def summary(findings_response):
    response = deepseek.chat.completions.create(
        model=DEEPSEEK_model_name,
        messages=[
            {
                "role": "system",
                "content": SUMMARIZER_system_prompt
            },
            {
                "role": "user",
                "content": findings_response
            }
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

async def text_to_speech(findings_list: list[str], lang: str = "English"):
    text = '\n'.join(findings_list)
    voice = VOICE_MAP[lang]
    communicate = edge_tts.Communicate(text, voice)

    audio = bytearray()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])

    return bytes(audio)

# --- Pydantic models ---

class SummarizeRequest(BaseModel):
    findings_json: str

class TTSRequest(BaseModel):
    statements: list[str]
    lang: str

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
        )

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Read and encode
    pdf_bytes = await file.read()

    if len(pdf_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="PDF exceeds 10 MB.")

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Invalid PDF file.")

    try:
        PdfReader(BytesIO(pdf_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted or invalid PDF.")

    encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    try:
        raw_findings = findings(encoded_pdf)
        parsed = json.loads(raw_findings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse findings from the AI model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

    if "ERROR" in parsed.keys():
        raise HTTPException(status_code=400, detail=str(parsed["ERROR"]))

    return JSONResponse(content=parsed)

@app.post("/summarize")
async def summarize(body: SummarizeRequest):
    try:
        result = summary(body.findings_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary error: {str(e)}")

    return JSONResponse(content=result)

@app.post("/tts")
async def tts(body: TTSRequest):
    if body.lang not in VOICE_MAP:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {body.lang}")
    if not body.statements:
        raise HTTPException(status_code=400, detail="No statements provided.")

    text = '\n'.join(body.statements)
    voice = VOICE_MAP[body.lang]

    async def audio_stream():
        communicate = edge_tts.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    return StreamingResponse(audio_stream(), media_type="audio/mpeg")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)