import streamlit as st
from PIL import Image
import tempfile
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Import from new hybrid vector_search
from vector_search import (
    init_rag_system,
    run_multimodal_qa,
)

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

load_dotenv()

# ------------------ Neo4j Load History ------------------
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

# ------------------ Initialize RAG System (Once) ------------------
if "rag_initialized" not in st.session_state:
    with st.spinner("🚀 Initializing Hybrid RAG System (LlamaIndex + LangChain)..."):
        try:
            init_rag_system(llm_model="gpt-4o-mini", temperature=0.0)
            st.session_state.rag_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            st.stop()

# ------------------ Session State Setup ------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "page" not in st.session_state:
    st.session_state.page = "start"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ Page 1: Landing ------------------
if st.session_state.page == "start":
    st.title("🤖 Multimodal Retrieval-Augmented Generation (RAG) Assistant")
    
    # Show system status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
This system integrates **document, audio, image, and video data** into a unified RAG framework for enhanced contextual understanding.

**Powered by:**
- 🦙 **LlamaIndex** - Advanced RAG & multimodal retrieval
- 🔗 **LangChain** - LLM orchestration & conversation memory
- 🗄️ **Neo4j** - Persistent chat history
- 🎯 **ChromaDB** - Vector storage across modalities
""")
    with col2:
        st.success("✅ RAG System Ready")

    with st.expander("📖 Project Info", expanded=False):
        st.markdown("""
### Multimodal RAG Assistant – Architecture

This assistant integrates **text, audio, image, and video data** into a unified **Retrieval-Augmented Generation (RAG)** system using a **hybrid LlamaIndex + LangChain architecture**.

**🛠️ Technology Stack**
- **LlamaIndex** - Query engines, vector indexing, multimodal retrieval
- **LangChain** - LLM orchestration, conversation chains, memory management
- **ChromaDB** - Vector storage (text/audio/image/video)
- **OpenAI Whisper** - Audio/video transcription
- **CLIP** - Image embeddings
- **OpenAI / Groq LLMs** - QA and reranking
- **Neo4j** - User-specific chat history persistence
- **RRF (Reciprocal Rank Fusion)** + **LLM-based reranking**

**🔄 Workflow**
1) **Input** (text or media) → 
2) **Classify** (history/follow-up/new) →
3) **Multimodal retrieval** (LlamaIndex) → 
4) **LLM rerank** → 
5) **Answer generation** →
6) **Persist to Neo4j**

**🎯 Key Features**
- Hybrid architecture leveraging strengths of both frameworks
- Multimodal vector search with fusion
- Context-aware follow-up handling
- Persistent conversation memory
- Dynamic model selection (OpenAI/Groq)
        """)

    st.markdown("---")
    user_id = st.text_input("👤 Enter your User ID", key="user_id_input")
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")

    model_choice = st.selectbox(
        "Choose LLM Model", 
        ["OpenAI (gpt-4o-mini)", "Groq (llama-3.3-70b-versatile)"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    with col2:
        max_tokens = st.slider("Max Tokens", 128, 4096, 1024, 64)

    if "OpenAI" in model_choice:
        st.info("ℹ️ `GPT-4o-mini` supports ~128k context; recommended: 2048–4096 tokens")
    else:
        st.info("ℹ️ `llama-3.3-70b-versatile` (Groq) supports ~8k context; recommended: 2048–4096 tokens")

    if st.button("🚀 Continue", type="primary"):
        if user_id.strip():
            st.session_state.user_id = user_id.strip()
            st.session_state.page = "chat"

            # Store model settings
            if "OpenAI" in model_choice:
                st.session_state.llm_model = "gpt-4o-mini"
            else:
                st.session_state.llm_model = "llama-3.3-70b-versatile"
            
            st.session_state.temperature = temperature
            st.session_state.max_tokens = max_tokens

            # Load past chat history from Neo4j
            past_chats = load_user_history_from_neo4j(user_id.strip())
            if past_chats:
                st.session_state.chat_history = [
                    {"user": q, "bot": a, "source": "history"} 
                    for q, a in past_chats[-10:]  # Last 10 conversations
                ]

            st.rerun()
        else:
            st.warning("⚠️ User ID cannot be empty.")

# ------------------ Page 2: Chat UI ------------------
elif st.session_state.page == "chat":
    st.title("💬 Ask Your Question")
    
    # Header with user info and sign out
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**👤 User:** `{st.session_state.user_id}`")
        st.markdown(f"**🤖 Model:** `{st.session_state.llm_model}`")
    with col2:
        if st.button("🚪 Sign Out", type="secondary"):
            st.session_state.page = "start"
            st.session_state.user_id = None
            st.session_state.chat_history = []
            if 'llm_model' in st.session_state:
                del st.session_state.llm_model
            if 'temperature' in st.session_state:
                del st.session_state.temperature
            if 'max_tokens' in st.session_state:
                del st.session_state.max_tokens
            st.rerun()

    st.markdown("---")

    # Input type selection and history button
    col1, col2 = st.columns([3, 1])
    with col1:
        input_type = st.radio(
            "Choose input type", 
            ["Text", "Image", "Audio", "Video"], 
            key="input_type_radio",
            horizontal=True
        )
    with col2:
        if st.button("📜 History"):
            with st.expander("Conversation History", expanded=True):
                if not st.session_state.chat_history:
                    st.info("No conversation history yet.")
                else:
                    for item in st.session_state.chat_history:
                        st.markdown(f"**You:** {item['user']}")
                        
                        source = item.get('source', 'unknown')
                        if source == "retriever":
                            label = "🔍 **Bot (Document):**"
                        elif source == "llm":
                            label = "🤖 **Bot (LLM-only):**"
                        elif source == "summary":
                            label = "📝 **Bot (Summary):**"
                        else:
                            label = "**Bot:**"
                        
                        st.markdown(f"{label} {item['bot']}")
                        st.markdown("---")
            st.stop()

    query = None
    file_path = None

    # Input handling based on type
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

    # Submit button
    if st.button("🚀 Submit", type="primary"):
        # Validate inputs
        if input_type == "Text":
            if not query or not query.strip():
                st.warning("⚠️ Please enter a question.")
                st.stop()
        elif input_type == "Image":
            if not file_path:
                st.warning("⚠️ Please upload an image.")
                st.stop()
            if not query or not query.strip():
                st.warning("⚠️ Please add a question about the image.")
                st.stop()
        else:
            if not file_path:
                st.warning(f"⚠️ Please upload a {input_type.lower()} file.")
                st.stop()

        # Exit shortcut
        if input_type == "Text" and query.strip().lower() == "exit":
            st.session_state.page = "start"
            st.session_state.user_id = None
            st.session_state.chat_history = []
            st.success("✅ You have been logged out.")
            st.rerun()

        with st.spinner("🔄 Processing your request..."):
            # Transcribe media to text
            if input_type in ["Audio", "Video"]:
                try:
                    from openai import OpenAI
                    client = OpenAI()
                    with open(file_path, "rb") as f:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1", 
                            file=f
                        )
                    query = (transcript.text or "").strip()
                    
                    if not query:
                        st.warning(f"⚠️ Could not transcribe the {input_type.lower()}. Please try another file.")
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                        st.stop()
                        
                except Exception as e:
                    st.error(f"❌ Transcription failed: {e}")
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                    st.stop()

            # Run QA using hybrid system
            try:
                result = run_multimodal_qa(
                    user_id=st.session_state.user_id,
                    query=query,
                    input_type=input_type.lower(),
                    file_path=file_path,
                    llm_model=st.session_state.llm_model,
                    temperature=st.session_state.temperature
                )
                
                answer = result["answer"]
                source = result["source"]
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                st.stop()

            # Display result
            if input_type in ["Audio", "Video"]:
                shown_query = f"[{input_type} transcription] {query[:180]}{'...' if len(query) > 180 else ''}"
            elif input_type == "Image":
                shown_query = f"[Image] {query}"
            else:
                shown_query = query

            st.markdown("### 💬 Response")
            st.markdown(f"**You:** {shown_query}")
            
            if source == "retriever":
                label = "🔍 **Bot (Document Retrieval):**"
                source_types = result.get("source_types", [])
                if source_types:
                    st.caption(f"📚 Sources: {', '.join(source_types)}")
            elif source == "llm":
                label = "🤖 **Bot (LLM Direct):**"
            elif source == "summary":
                label = "📝 **Bot (History Summary):**"
            else:
                label = "**Bot:**"
            
            st.markdown(f"{label} {answer}")

            # Add to chat history
            st.session_state.chat_history.append({
                "user": shown_query,
                "bot": answer,
                "source": source
            })

        # Clean up temp file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

    # Show recent conversation
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📝 Recent Conversation")
        # Show last 3 exchanges
        for item in st.session_state.chat_history[-3:]:
            with st.container():
                st.markdown(f"**You:** {item['user']}")
                
                source = item.get('source', 'unknown')
                if source == "retriever":
                    st.markdown(f"🔍 **Bot:** {item['bot']}")
                elif source == "llm":
                    st.markdown(f"🤖 **Bot:** {item['bot']}")
                else:
                    st.markdown(f"**Bot:** {item['bot']}")
                st.markdown("---")