import requests
import streamlit as st
import time

# ---------- Page setup ----------
st.set_page_config(page_title="Interview RAG", page_icon="🔎", layout="wide")
st.title("🔎 Interview RAG — Upload & Ask")

# ---------- Server config ----------
if "server_url" not in st.session_state:
    st.session_state.server_url = "http://127.0.0.1:8000"

def api(path: str) -> str:
    return f"{st.session_state.server_url.rstrip('/')}{path}"

with st.sidebar:
    st.header("Server")
    st.session_state.server_url = st.text_input(
        "FastAPI base URL",
        value=st.session_state.server_url,
        help="Default is http://127.0.0.1:8000 when running locally."
    )
    cols = st.columns(2)
    if cols[0].button("Ping /"):
        try:
            r = requests.get(api("/"))
            st.success(r.json())
        except Exception as e:
            st.error(f"Ping failed: {e}")

# ---------- Tabs ----------
tab_upload, tab_ask, tab_search = st.tabs(["📤 Upload", "💬 Ask", "🔍 Search (debug)"])

# ====== Upload tab ======
with tab_upload:
    st.subheader("Upload PDF / Audio / Video")
    st.caption("Supported: .pdf, .mp3, .wav, .mp4, .mov")

    f = st.file_uploader(
        "Choose a file",
        type=["pdf", "mp3", "wav", "mp4", "mov"],
        accept_multiple_files=False
    )

    if st.button("Upload"):
        if not f:
            st.warning("Please choose a file.")
        else:
            try:
                # Step 1: send file
                st.info("📂 Step 1: Sending file to backend…")
                files = {"file": (f.name, f.getvalue(), f.type or "application/octet-stream")}
                init = requests.post(api("/upload_async/"), files=files, timeout=60)

                if not init.ok:
                    st.error(f"Upload failed: {init.status_code}\n{init.text}")
                else:
                    data = init.json()
                    job_id = data["job_id"]
                    st.success(f"✅ Step 1 complete: File accepted. Job ID: `{job_id}`")

                    # Progress UI
                    prog = st.progress(0)
                    status_line = st.empty()

                    # Checklist placeholders
                    checklist = {
                        "uploading": st.empty(),
                        "extracting": st.empty(),
                        "chunking": st.empty(),
                        "embedding": st.empty(),
                        "done": st.empty()
                    }
                    checklist["uploading"].write("⏳ Uploading")
                    checklist["extracting"].write("⏳ Extracting / Transcribing")
                    checklist["chunking"].write("⏳ Chunking")
                    checklist["embedding"].write("⏳ Embedding & Indexing")
                    checklist["done"].write("⏳ Finalizing")

                    # One-time flags to avoid flicker
                    did_upload = did_extract = did_chunk = did_embed = False

                    while True:
                        try:
                            s = requests.get(api(f"/status/{job_id}"), timeout=10)
                            if not s.ok:
                                st.error(f"Status check failed: {s.status_code}\n{s.text}")
                                break
                            info = s.json()
                        except Exception as e:
                            st.error(f"Status error: {e}")
                            break

                        pct = int(info.get("pct", 0))
                        stage = (info.get("stage") or "").lower()
                        err = info.get("error")
                        done = bool(info.get("done"))

                        # Update progress bar + line
                        prog.progress(min(max(pct, 0), 100))
                        status_line.write(f"**{stage or 'working…'}** — {pct}%")

                        # --- Robust step detection (stage OR pct) ---
                        if not did_upload and (
                            "upload" in stage or "saved" in stage or pct >= 15
                        ):
                            checklist["uploading"].write("✅ Uploading")
                            did_upload = True

                        if not did_extract and (
                            "extract" in stage or "transcrib" in stage or pct >= 60
                        ):
                            checklist["extracting"].write("✅ Extracting / Transcribing")
                            did_extract = True

                        if not did_chunk and (
                            "chunk" in stage or pct >= 85
                        ):
                            checklist["chunking"].write("✅ Chunking")
                            did_chunk = True

                        if not did_embed and (
                            "embed" in stage or "index" in stage or pct >= 95 or done
                        ):
                            checklist["embedding"].write("✅ Embedding & Indexing")
                            did_embed = True

                        if done:
                            if err:
                                checklist["done"].write(f"❌ Failed: {err}")
                                st.error(f"❌ Failed: {err}")
                            else:
                                checklist["done"].write("✅ Finalizing")
                                st.success("🎉 All steps complete. File processed and embeddings stored.")
                            break

                        time.sleep(0.6)

            except Exception as e:
                st.error(f"Error: {e}")


# ====== Ask tab ======
with tab_ask:
    st.subheader("Ask a question about your interviews")
    mode = st.selectbox(
        "Mode",
        options=["auto", "qa", "summary"],
        index=0,
        help="Auto picks based on your question. QA = concise; Summary = broader context."
    )
    question = st.text_area(
        "Your question",
        placeholder="e.g., Summarize the key themes discussed in the latest interview.",
        height=100,
    )
    show_chunks = st.checkbox("Show retrieved chunks", value=True)

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            payload = {"question": question, "mode": mode}
            try:
                st.info("🧭 Step 1: Routing question to QA/Summary mode…")
                r = requests.post(api("/ask/"), json=payload, timeout=120)

                if not r.ok:
                    st.error(f"/ask failed: {r.status_code}\n{r.text}")
                else:
                    data = r.json()
                    st.success(f"✅ Step 1 complete: Mode = `{data.get('mode')}`")

                    st.info("🔎 Step 2: Retrieving relevant chunks from FAISS…")
                    chunks = data.get("retrieved_chunks") or []
                    if show_chunks and chunks:
                        for i, ch in enumerate(chunks, 1):
                            src = ch.get("source", "unknown")
                            cid = ch.get("chunk_id", "?")
                            dist = ch.get("distance", None)
                            title = f"Chunk {i} — {src} (id={cid}"
                            if isinstance(dist, (int, float)):
                                title += f", dist={dist:.4f}"
                            title += ")"
                            with st.expander(title, expanded=(i == 1)):
                                st.write(ch.get("text", ""))
                    st.success(f"✅ Step 2 complete: {len(chunks)} chunks retrieved.")

                    st.info("🤖 Step 3: Asking Gemini for final answer…")
                    st.success("✅ Step 3 complete: Answer generated.")

                    st.markdown("### 🎯 Final Answer")
                    st.write(data.get("answer", ""))

            except Exception as e:
                st.error(f"Error: {e}")

# ====== Search tab (debug helper) ======
with tab_search:
    st.subheader("Search (debug) — see what the retriever surfaces")
    q = st.text_input("Search query", placeholder="Keyword or short question…")
    if st.button("Search"):
        if not q.strip():
            st.warning("Please type something to search.")
        else:
            try:
                payload = {"query": q, "top_k": 3}
                with st.spinner("Searching…"):
                    r = requests.post(api("/search/"), json=payload, timeout=60)
                if not r.ok:
                    st.error(f"/search failed: {r.status_code}\n{r.text}")
                else:
                    data = r.json()
                    results = data.get("results", [])
                    if not results:
                        st.info("No results.")
                    for i, res in enumerate(results, 1):
                        src = res.get("source", "unknown")
                        dist = res.get("distance", None)
                        header = f"Result {i} — {src}"
                        if isinstance(dist, (int, float)):
                            header += f" (dist={dist:.4f})"
                        with st.expander(header):
                            st.write(res.get("text", ""))
            except Exception as e:
                st.error(f"Search error: {e}")
