# 🧠 Interview RAG — AI-Powered Interview Insights

**Interview RAG** is a lightweight Retrieval-Augmented Generation (RAG) app built for product managers and researchers to analyze interviews and summarize insights from PDFs, audio, and video files — all locally and cost-efficiently.

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


### Components
- **FastAPI** → handles file upload, transcription, embedding, and Gemini calls  
- **Streamlit** → frontend to upload, visualize progress, and query  
- **Whisper (base)** → transcribes audio/video files  
- **FAISS Index** → semantic search over text chunks  
- **Gemini API** → generates contextual answers  

---

## 🧩 1️⃣ Local Setup Steps

### 1. Clone the repo
```bash
git clone https://github.com/shubham119413/interview-rag.git
cd interview-rag

2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

3. Install dependencies
pip install -r requirements.txt

4. Add your Gemini API key

Create a file named .env in the project root:

GEMINI_API_KEY=your_api_key_here


(If you don’t have a key, create one at https://aistudio.google.com/app/apikey
)

🚀 2️⃣ Run Locally
Step 1 — Start the backend (FastAPI)
uvicorn main:app --reload --port 8000


This starts the server at http://127.0.0.1:8000/docs
.

Step 2 — Start the frontend (Streamlit)

Open a new terminal (while keeping FastAPI running):

streamlit run app.py


You can now upload files, view real-time progress, and ask questions through the UI.