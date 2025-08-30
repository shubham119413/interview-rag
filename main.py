from fastapi import FastAPI, UploadFile, File, HTTPException
import os
from pypdf import PdfReader
import whisper
from moviepy import VideoFileClip
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from fastapi import Body
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import BackgroundTasks
from uuid import uuid4
from typing import Dict

PROGRESS: Dict[str, dict] = {}  # { job_id: {"pct": int, "stage": str, "file": str, "done": bool, "error": str|None} }

def set_progress(job_id: str, pct: int, stage: str, file: str = "", done: bool = False, error: str | None = None):
    PROGRESS[job_id] = {"pct": pct, "stage": stage, "file": file, "done": done, "error": error}


# --- 1. Create FastAPI app ---
app = FastAPI()

# --- 2. Global setup ---
print("🔄 Loading AI models...")
whisper_model = whisper.load_model("base")  # Whisper model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Sentence embeddings
print("✅ AI models loaded")

# FAISS index (cosine similarity via normalized embeddings)
dimension = 384  # embedding size for all-MiniLM-L6-v2
index = faiss.IndexFlatIP(dimension)
text_data = []  # stores metadata for each chunk

# Directory setup
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# --- 3. Chunking Helpers ---
# =========================
def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 150):
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if not chunk.strip():
            break

        chunks.append({"text": chunk, "start": start, "end": end})
        start += chunk_size - overlap
    return chunks

def store_embedding(text: str, source: str, mode: str = "qa"):
    """
    Split text into chunks, embed each, and add to FAISS with metadata.
    mode = 'qa' (short chunks) or 'summary' (longer chunks).
    """
    global text_data
    if mode == "qa":
        chunk_size, overlap = 1000, 150
    else:
        chunk_size, overlap = 2500, 300

    chunks = split_text_into_chunks(text, chunk_size, overlap)

    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk["text"]).astype(np.float32)
        embedding /= np.linalg.norm(embedding)  # normalize
        index.add(np.array([embedding]))

        text_data.append({
            "text": chunk["text"],
            "source": source,
            "chunk_id": i,
            "start": chunk["start"],
            "end": chunk["end"]
        })

    print(f"✅ Stored {len(chunks)} chunks for {source}")

# =========================
# --- 4. Processing functions ---
# =========================
def process_pdf(file_path):
    print(f"📄 Extracting text from PDF: {file_path}")
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    store_embedding(text, file_path)
    print("✅ PDF processed")
    return text

def process_audio(file_path):
    print(f"🎙️ Transcribing audio: {file_path}")
    result = whisper_model.transcribe(file_path)
    text = result["text"]
    store_embedding(text, file_path)
    print("✅ Audio processed")
    return text

def process_video(file_path):
    print(f"🎥 Extracting audio from video: {file_path}")
    clip = VideoFileClip(file_path)
    audio_path = file_path.rsplit(".", 1)[0] + ".wav"
    clip.audio.write_audiofile(audio_path)
    clip.close()
    return process_audio(audio_path)

def _process_file_job(job_id: str, file_path: str):
    try:
        # File already marked as "uploading" in /upload_async
        set_progress(job_id, 20, "file received", file_path)

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            set_progress(job_id, 25, "extracting", file_path)
            _process_pdf_with_progress(job_id, file_path)
        elif ext in (".mp3", ".wav"):
            set_progress(job_id, 30, "transcribing", file_path)
            text = process_audio(file_path)
            set_progress(job_id, 75, "transcribing complete", file_path)
            _store_with_progress(job_id, text, file_path)
        elif ext in (".mp4", ".mov"):
            set_progress(job_id, 30, "transcribing", file_path)
            text = process_video(file_path)
            set_progress(job_id, 75, "transcribing complete", file_path)
            _store_with_progress(job_id, text, file_path)
        else:
            set_progress(job_id, 100, "failed", file_path, done=True, error="Unsupported file type")
            return

        # Let embedding be visible for at least one poll cycle before done
        set_progress(job_id, 100, "done", file_path, done=True)

    except Exception as e:
        set_progress(job_id, 100, "failed", file_path, done=True, error=str(e))



def _process_pdf_with_progress(job_id: str, file_path: str):
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        n = len(reader.pages) or 1
        texts = []
        for i, page in enumerate(reader.pages, start=1):
            t = page.extract_text() or ""
            texts.append(t)

            # Smooth progress between 20% and 60%
            pct = 20 + int(40 * i / n)
            set_progress(job_id, pct, f"📄 Extracting PDF text ({i}/{n})", file_path)

    full_text = " ".join(texts)
    _store_with_progress(job_id, full_text, file_path)


def _store_with_progress(job_id: str, text: str, source: str):
    # Chunking step
    set_progress(job_id, 85, "chunking", source)

    # Actually embed & store
    store_embedding(text, source)

    # Mark embedding AFTER storage so it’s visible to UI
    set_progress(job_id, 95, "embedding", source)





# =========================
# --- 5. Retrieval Helper ---
# =========================
def retrieve_chunks(query: str, mode: str = "qa", top_k: int = 6):
    """Retrieve chunks relevant to query, depending on mode."""
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="❌ No embeddings found. Upload files first.")

    query_embedding = embedding_model.encode(query).astype(np.float32)
    query_embedding /= np.linalg.norm(query_embedding)

    initial_k = 48 if mode == "qa" else 80
    distances, indices = index.search(np.array([query_embedding]), initial_k)

    final_k = 6 if mode == "qa" else 30
    results = []
    for idx, dist in zip(indices[0][:final_k], distances[0][:final_k]):
        meta = text_data[idx]
        results.append({
            "text": meta["text"][:300] + "...",
            "source": meta["source"],
            "chunk_id": meta["chunk_id"],
            "distance": float(dist)
        })
    return results

# =========================
# --- 6. Endpoints ---
# =========================
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI 🎉"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"📂 File uploaded: {file.filename}")

        if file.filename.endswith(".pdf"):
            process_pdf(file_path)
        elif file.filename.endswith((".mp3", ".wav")):
            process_audio(file_path)
        elif file.filename.endswith((".mp4", ".mov")):
            process_video(file_path)
        else:
            raise HTTPException(status_code=400, detail="❌ Unsupported file type")

        return {"message": f"✅ File '{file.filename}' uploaded and processed successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Upload failed: {str(e)}")
    
# =========================
# --- 7. Search Endpoint ---
# =========================
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/search/")
async def search_text(request: SearchRequest = Body(...)):
    results = retrieve_chunks(request.query, mode="qa", top_k=request.top_k)
    return {"query": request.query, "results": results}

# =========================
# --- 8. Ask Endpoint (Gemini) ---
# =========================
class AskRequest(BaseModel):
    question: str
    mode: str = "auto"

@app.post("/ask/")
async def ask_question(request: AskRequest = Body(...)):
    mode = request.mode
    if mode == "auto":
        q = request.question.lower()
        if any(word in q for word in ["summarize", "overview", "elaborate", "explain", "detailed"]):
            mode = "summary"
        else:
            mode = "qa"

    retrieved = retrieve_chunks(request.question, mode=mode)
    context = "\n\n".join([r["text"] for r in retrieved])

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Context:\n{context}\n\nQuestion: {request.question}\n\nAnswer:")

    return {
        "mode": mode,
        "question": request.question,
        "answer": response.text,
        "retrieved_chunks": retrieved   # full text + metadata + distances
    }

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in PROGRESS:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return PROGRESS[job_id]

@app.post("/upload_async/")
async def upload_file_async(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    job_id = str(uuid4())
    filename = file.filename
    set_progress(job_id, 5, "starting", filename)

    # Save file to disk
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Mark uploading here (only once)
    set_progress(job_id, 10, "uploading", filename)

    # Kick off processing in the background
    background_tasks.add_task(_process_file_job, job_id, file_path)

    return {"job_id": job_id, "message": f"Upload accepted for {filename}"}



# --- 9. Gemini Setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))