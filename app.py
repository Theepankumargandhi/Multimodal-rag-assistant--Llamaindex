# app.py
import os
import tempfile
from PIL import Image
import streamlit as st
from dotenv import load_dotenv

# Hybrid RAG
from vector_search import init_rag_system, run_multimodal_qa

# Neo4j (for loading past chat history)
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import config  # so we can toggle GRAPH_RAG_ENABLED at runtime

load_dotenv()

# ==============================
# Helpers
# ==============================
def load_user_history_from_neo4j(user_id: str):
    query = """
    MATCH (u:User {id: $user_id})-[:HAS_SESSION]->(s:Session)-[:ASKED]->(q:Query)-[:ANSWERED_BY]->(a:Answer)
    RETURN q.text AS question, a.text AS answer
    ORDER BY q.timestamp DESC
    LIMIT 50
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        results = session.run(query, parameters={"user_id": user_id})
        return [(record["question"], record["answer"]) for record in results]


def reset_session():
    st.session_state.page = "start"
    st.session_state.user_id = None
    st.session_state.chat_history = []
    for k in ("llm_model", "temperature", "max_tokens"):
        if k in st.session_state:
            del st.session_state[k]


# ==============================
# Initialize RAG System once
# ==============================
if "rag_initialized" not in st.session_state:
    with st.spinner("🚀 Initializing Hybrid RAG System (LlamaIndex + LangChain)..."):
        try:
            init_rag_system(llm_model="gpt-4o-mini", temperature=0.0)
            st.session_state.rag_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            st.stop()

# ==============================
# Session state defaults
# ==============================
st.session_state.setdefault("user_id", None)
st.session_state.setdefault("page", "start")
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("llm_model", "gpt-4o-mini")
st.session_state.setdefault("temperature", 0.2)
st.session_state.setdefault("max_tokens", 1024)
st.session_state.setdefault("hist_n", 5)  # how many history rows to show in UI


# ==============================
# Sidebar: Global toggles
# ==============================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    use_graph = st.checkbox("Enable Graph-RAG", value=config.GRAPH_RAG_ENABLED)
    config.GRAPH_RAG_ENABLED = bool(use_graph)
    st.caption("When ON, answers may include relationship facts from Neo4j for relational questions.")
    if st.button("🧹 Clear chat"):
        st.session_state.chat_history = []
        st.success("Cleared current session chat.")


# ==============================
# Page 1: Landing
# ==============================
if st.session_state.page == "start":
    st.title("🤖 Multimodal Retrieval-Augmented Generation (RAG) Assistant")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            """
This system integrates **document, audio, image, and video** into a unified RAG framework.

**Powered by**
- 🦙 LlamaIndex (multimodal retrieval)
- 🔗 LangChain (LLM orchestration & memory)
- 🗄️ Neo4j (chat history + Graph-RAG)
- 🎯 ChromaDB (vector store)
"""
        )
    with col2:
        st.success("✅ RAG System Ready")

    with st.expander("📖 Project Info", expanded=False):
        st.markdown(
            """
### Architecture (high level)
1) Input (text/media)  
2) Classify (history / follow-up / new)  
3) Multimodal retrieval (LlamaIndex)  
4) LLM rerank  
5) Answer generation  
6) Persist to Neo4j (history + optional graph facts)
"""
        )

    st.markdown("---")
    user_id = st.text_input("👤 Enter your User ID", key="user_id_input")

    st.markdown("---")
    st.markdown("### 🧪 Model Settings")
    model_choice = st.selectbox(
        "Choose LLM Model", ["OpenAI (gpt-4o-mini)", "Groq (llama-3.3-70b-versatile)"]
    )
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    with col2:
        max_tokens = st.slider("Max Tokens", 128, 4096, 1024, 64)

    if "OpenAI" in model_choice:
        st.info("ℹ️ `gpt-4o-mini` has large context; try 2048–4096 tokens for long answers.")
    else:
        st.info("ℹ️ `llama-3.3-70b-versatile` (~8k context); 2048–4096 tokens is typical.")

    if st.button("🚀 Continue", type="primary"):
        if user_id.strip():
            st.session_state.user_id = user_id.strip()
            st.session_state.page = "chat"
            st.session_state.llm_model = "gpt-4o-mini" if "OpenAI" in model_choice else "llama-3.3-70b-versatile"
            st.session_state.temperature = temperature
            st.session_state.max_tokens = max_tokens

            # hydrate chat history from Neo4j
            past = load_user_history_from_neo4j(st.session_state.user_id)
            if past:
                st.session_state.chat_history = [
                    {"user": q, "bot": a, "source": "history"} for q, a in past[-10:]
                ]
            st.rerun()
        else:
            st.warning("⚠️ User ID cannot be empty.")


# ==============================
# Page 2: Chat
# ==============================
elif st.session_state.page == "chat":
    st.title("💬 Ask Your Question")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**👤 User:** `{st.session_state.user_id}`")
        st.markdown(f"**🤖 Model:** `{st.session_state.llm_model}`")
        st.caption(f"Graph-RAG: {'ON' if config.GRAPH_RAG_ENABLED else 'OFF'}")
    with col2:
        if st.button("🚪 Sign Out", type="secondary"):
            reset_session()
            st.rerun()

    st.markdown("---")

    # Input type + history
    col1, col2 = st.columns([3, 1])
    with col1:
        input_type = st.radio(
            "Choose input type",
            ["Text", "Image", "Audio", "Video"],
            key="input_type_radio",
            horizontal=True,
        )
    with col2:
        if st.button("📜 History"):
            with st.expander("Conversation History", expanded=True):
                if not st.session_state.chat_history:
                    st.info("No conversation history yet.")
                else:
                    N = st.session_state.get("hist_n", 5)
                    to_show = st.session_state.chat_history[-N:]
                    for item in to_show:
                        st.markdown(f"**You:** {item['user']}")
                        label = {
                            "retriever": "🔍 **Bot (Document):**",
                            "llm": "🤖 **Bot (LLM-only):**",
                            "summary": "📝 **Bot (Summary):**",
                        }.get(item.get("source", ""), "**Bot:**")
                        st.markdown(f"{label} {item['bot']}")
                        st.markdown("---")
                    if len(st.session_state.chat_history) > N:
                        if st.button("Load more"):
                            st.session_state.hist_n = N + 5
                            st.rerun()
            st.stop()

    # Collect input
    query = None
    file_path = None

    if input_type == "Text":
        query = st.text_area("💭 Type your question here", height=100)

    elif input_type == "Image":
        uploaded_file = st.file_uploader("📷 Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            query = st.text_area("❓ Ask a question about this image", height=100)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                image.save(tmp.name)
                file_path = tmp.name

    elif input_type == "Audio":
        uploaded_file = st.file_uploader("🎵 Upload an audio file", type=["mp3", "wav", "m4a"])
        if uploaded_file:
            st.audio(uploaded_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                file_path = tmp.name

    elif input_type == "Video":
        uploaded_file = st.file_uploader("🎬 Upload a video file", type=["mp4", "mov", "avi"])
        if uploaded_file:
            st.video(uploaded_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                file_path = tmp.name

    # Submit
    if st.button("🚀 Submit", type="primary"):
        # Basic validation
        if input_type == "Text" and not (query and query.strip()):
            st.warning("⚠️ Please enter a question.")
            st.stop()
        if input_type == "Image" and (not file_path or not (query and query.strip())):
            st.warning("⚠️ Please upload an image and ask a question about it.")
            st.stop()
        if input_type in {"Audio", "Video"} and not file_path:
            st.warning(f"⚠️ Please upload a {input_type.lower()} file.")
            st.stop()

        # Exit shortcut
        if input_type == "Text" and query.strip().lower() == "exit":
            reset_session()
            st.success("✅ You have been logged out.")
            st.rerun()

        with st.spinner("🔄 Processing your request..."):
            # Transcribe if needed
            if input_type in ["Audio", "Video"]:
                try:
                    from openai import OpenAI
                    client = OpenAI()
                    with open(file_path, "rb") as f:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1", file=f
                        )
                    query = (transcript.text or "").strip()
                    if not query:
                        st.warning(f"⚠️ Could not transcribe the {input_type.lower()} file.")
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Transcription failed: {e}")
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                    st.stop()

            # Run QA
            try:
                result = run_multimodal_qa(
                    user_id=st.session_state.user_id,
                    query=query,
                    input_type=input_type.lower(),
                    file_path=file_path,
                    llm_model=st.session_state.llm_model,
                    temperature=st.session_state.temperature,
                )
            except Exception as e:
                st.error("❌ Error processing query.")
                st.exception(e)
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                st.stop()

            answer = result["answer"]
            source = result["source"]
            graph_facts = result.get("graph_facts", [])
            source_types = result.get("source_types", [])

            # Display
            shown_query = (
                f"[{input_type} transcription] {query[:180]}{'...' if len(query) > 180 else ''}"
                if input_type in ["Audio", "Video"]
                else (f"[Image] {query}" if input_type == "Image" else query)
            )
            st.markdown("### 💬 Response")
            st.markdown(f"**You:** {shown_query}")

            label = {
                "retriever": "🔍 **Bot (Document Retrieval):**",
                "llm": "🤖 **Bot (LLM Direct):**",
                "summary": "📝 **Bot (History Summary):**",
            }.get(source, "**Bot:**")
            st.markdown(f"{label} {answer}")

            if source_types:
                st.caption(f"📚 Sources: {', '.join(source_types)}")
            if graph_facts:
                st.caption("🧠 Used Graph-RAG context")
                with st.expander(f"🔗 Supporting Graph Facts (Neo4j) — {len(graph_facts)} found"):
                    # show unique doc ids
                    docs = set()
                    for f in graph_facts:
                        # dataclass or dict
                        e1 = getattr(f, "e1", None) or f.get("e1")
                        rel = getattr(f, "rel", None) or f.get("rel")
                        e2 = getattr(f, "e2", None) or f.get("e2")
                        doc = getattr(f, "source_doc", None) or f.get("source_doc", "unknown")
                        chunk = getattr(f, "chunk_id", None) or f.get("chunk_id", "unknown")
                        docs.add(doc)
                        st.write(f"- {e1} {rel} {e2} (doc: {doc}, chunk: {chunk})")
                    if docs:
                        st.caption("📄 Docs: " + ", ".join(sorted([d for d in docs if d])))

            # Add to chat history
            st.session_state.chat_history.append(
                {"user": shown_query, "bot": answer, "source": source}
            )

        # Clean temp file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

    # Recent conversation
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📝 Recent Conversation")
        for item in st.session_state.chat_history[-3:]:
            with st.container():
                st.markdown(f"**You:** {item['user']}")
                tag = {
                    "retriever": "🔍 **Bot:**",
                    "llm": "🤖 **Bot:**",
                }.get(item.get("source", ""), "**Bot:**")
                st.markdown(f"{tag} {item['bot']}")
                st.markdown("---")
