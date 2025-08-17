import os
import time
from collections import defaultdict, deque
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
GEN_MODEL = os.getenv("GENERATION_MODEL", "gemini-1.5-flash")
DAILY_FREE_LIMIT = int(os.getenv("DAILY_FREE_LIMIT", "100"))

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY missing. Set it in .env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:*/",
        "chrome-extension://*",
        "https://youtube-helper-chrome-extension-backend.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# In-memory caches/storage
# -----------------------
VIDEO_CACHE: Dict[str, Dict[str, Any]] = {}
HITS: Dict[str, deque] = defaultdict(lambda: deque())

def _day_bucket() -> int:
    return int(time.time() // 86400)

CURRENT_BUCKET = _day_bucket()

def rate_limit(remote: str) -> bool:
    global CURRENT_BUCKET
    b = _day_bucket()
    if b != CURRENT_BUCKET:
        HITS.clear()
        CURRENT_BUCKET = b

    dq = HITS[remote]
    now = time.time()
    while dq and now - dq[0] > 86400:
        dq.popleft()

    if len(dq) >= DAILY_FREE_LIMIT:
        return False

    dq.append(now)
    return True


class ProcessInput(BaseModel):
    video_id: str
    question: str


# -----------------------
# Transcript Fetch with User-Agent Patch
# -----------------------
session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
})
# Patch the internal session used by YouTubeTranscriptApi
YouTubeTranscriptApi._YouTubeTranscriptApi__session = session


def fetch_transcript_text(video_id: str) -> str:
    fetched = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    return " ".join(snippet["text"] for snippet in fetched)


# --- rest of your code unchanged (vector store, LLM, routes, etc.) ---
