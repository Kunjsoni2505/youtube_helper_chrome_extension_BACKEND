import os
import time
from collections import defaultdict, deque
from typing import Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import FAISS vector store (community path is recommended in recent LangChain)
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS  # fallback for older installs

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

# Allow extension to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# In-memory caches/storage
# -----------------------
# Cache per video_id: {"vector": FAISS, "chunks": List[Document], "raw_text": str}
VIDEO_CACHE: Dict[str, Dict[str, Any]] = {}

# Simple IP-based daily rate limit: store timestamps of hits within today
HITS: Dict[str, deque] = defaultdict(lambda: deque())

# Reset window each day (midnight)
def _day_bucket() -> int:
    return int(time.time() // 86400)

CURRENT_BUCKET = _day_bucket()


def rate_limit(remote: str) -> bool:
    """
    Return True if the requester is allowed; False if rate-limited.
    """
    global CURRENT_BUCKET
    b = _day_bucket()
    if b != CURRENT_BUCKET:
        # new day: clear hits
        HITS.clear()
        CURRENT_BUCKET = b

    dq = HITS[remote]
    now = time.time()
    # keep only today's hits
    while dq and now - dq[0] > 86400:
        dq.popleft()

    if len(dq) >= DAILY_FREE_LIMIT:
        return False

    dq.append(now)
    return True


class ProcessInput(BaseModel):
    video_id: str
    question: str


def fetch_transcript_text(video_id: str) -> str:
    # youtube_transcript_api 1.2.2 returns iterable of FetchedTranscriptSnippet
    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id, languages=["en"])
    text = " ".join(snippet.text for snippet in fetched)
    return text


def build_vector_store(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    vs = FAISS.from_documents(docs, embeddings)
    return vs, docs


PROMPT = PromptTemplate(
    template=(
        "You are a helpful assistant.\n"
        "Answer only from the provided transcript context.\n"
        "If the content is insufficient, say you don't know.\n\n"
        "{context}\n\n"
        "Question: {question}"
    ),
    input_variables=["context", "question"],
)

def answer_with_gemini(context_text: str, question: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model=GEN_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )
    final_prompt = PROMPT.format(context=context_text, question=question)
    resp = llm.invoke(final_prompt)
    return getattr(resp, "content", str(resp))


@app.post("/process")
async def process(req: Request, payload: ProcessInput):
    client_ip = req.client.host if req.client else "unknown"

    allowed = rate_limit(client_ip)
    if not allowed:
        return {
            "rate_limited": True,
            "message": "Daily free limit reached. Try later or upgrade.",
            "answer": None,
        }

    video_id = payload.video_id.strip()
    question = payload.question.strip()
    if not video_id or not question:
        return {"error": "video_id and question are required"}

    # Get or build cache
    if video_id not in VIDEO_CACHE:
        try:
            text = fetch_transcript_text(video_id)
        except TranscriptsDisabled:
            return {"error": f"Transcripts are disabled for video {video_id}"}
        except Exception as e:
            return {"error": f"Transcript fetch error: {e}"}

        vs, docs = build_vector_store(text)
        VIDEO_CACHE[video_id] = {"vector": vs, "chunks": docs, "raw_text": text}

    vs = VIDEO_CACHE[video_id]["vector"]
    # Retrieve top-k relevant chunks
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # If nothing retrieved, avoid empty call
    if not context_text.strip():
        return {
            "answer": "I don't know. The transcript didn't contain information for this question.",
            "rate_limited": False,
            "video_id": video_id,
        }

    # Ask Gemini
    try:
        answer = answer_with_gemini(context_text, question)
    except Exception as e:
        return {"error": f"Gemini API error: {e}"}

    return {
        "answer": answer,
        "rate_limited": False,
        "video_id": video_id,
        "used_chunks": len(retrieved_docs),
    }


@app.get("/")
async def root():
    return {"ok": True, "service": "yt-helper-backend"}
