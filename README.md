# 🧠 Interview RAG — AI-Powered Interview Insights

**Interview RAG** is a lightweight Retrieval-Augmented Generation (RAG) app built for product managers and researchers to analyze interviews and summarize insights from PDFs, audio, and video files — all locally and cost-efficiently. It uses:

- ✅ FastAPI for backend API
- ✅ Streamlit for front-end
- ✅ Whisper for transcription
- ✅ SentenceTransformers + FAISS for retrieval
- ✅ Gemini 2.0 Flash for AI-powered answers
- ✅ Progress tracked for better debuggability

---

## ⚙️ Architecture Overview
            ┌──────────────────────────┐
            │      Streamlit UI        │
            │  (upload + ask + debug)  │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │       FastAPI Backend     │
            │  - Async upload + status  │
            │  - Whisper + pdf extract  │
            │  - Chunk + FAISS embed    │
            │  - Gemini QA/Summary API  │
            └────────────┬─────────────┘
                         │
                         ▼
         ┌─────────────────────────────┐
         │  FAISS Vector DB (in-memory) │
         │                             │
         │  Text chunks + embeddings    │
         └─────────────────────────────┘




### 🚀 Features
- Upload and process:PDF documents, Audio files (.mp3, .wav), Video files (.mp4, .mov)
- Transcribes audio/video using Whisper
- Extracts text from PDFs
- Generates embeddings using SentenceTransformers
- Creates short/long contextual chunks for efficient retreival
- Stores embeddings in FAISS
- Answers questions using Gemini 2.0 Flash

---
## 🛠️ How to Run Locally

1. Install dependencies:
    pip install -r requirements.txt

2. Create a `.env` file with your Gemini key:
    GEMINI_API_KEY=your-gemini-api-key-here

3. Run the FastAPI server:
    uvicorn main:app --host=0.0.0.0 --port=8000 --reload

Then open:
    http://localhost:8000/docs


## 🧩 Detailed steps

### 1. Clone the repo
```bash
git clone https://github.com/shubham119413/interview-rag.git
cd interview-rag
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key

Create a file named .env in the project root:
```bash
GEMINI_API_KEY=your_api_key_here
```

(If you don’t have a key, create one at https://aistudio.google.com/app/apikey
)

### 🚀 Run Locally
**Step 1 — Start the backend (FastAPI)**
```bash
uvicorn main:app --reload --port 8000
```

This starts the server at http://127.0.0.1:8000/docs
.

**Step 2 — Start the frontend (Streamlit)**

Open a new terminal (while keeping FastAPI running):
```bash
streamlit run app.py
```

You can now upload files, view real-time progress, and ask questions through the UI.
