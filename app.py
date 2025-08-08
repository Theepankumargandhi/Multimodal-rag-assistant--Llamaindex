import streamlit as st
from PIL import Image
import tempfile
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from vector_search import (
    create_chain,
    save_to_neo4j,
    detect_query_type,
    summarize_history,
    rewrite_query_with_history,
    MultimodalRetriever,
)

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

load_dotenv()

# ------------------ Neo4j Load History ------------------
def load_user_history_from_neo4j(user_id: str):
    query = """
    MATCH (u:User {id: $user_id})-[:HAS_SESSION]->(:Session)-[:ASKED]->(q:Query)-[:ANSWERED_BY]->(a:Answer)
    RETURN q.text AS question, a.text AS answer
    ORDER BY q.timestamp
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        results = session.run(query, parameters={"user_id": user_id})
        return [(record["question"], record["answer"]) for record in results]

# ------------------ Session State Setup ------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "page" not in st.session_state:
    st.session_state.page = "start"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ Page 1: Landing ------------------
if st.session_state.page == "start":
    st.title("Multimodal Retrieval-Augmented Generation (RAG) Assistant")
    st.markdown("""
This system integrates document, audio, image, and video data into a unified RAG framework for enhanced contextual understanding.
It supports LLM-based reasoning with configurable settings and personalized chat history via Neo4j.
""")

    with st.expander("Project Info", expanded=False):
        st.markdown("""
### Multimodal RAG Assistant – Project Description

This assistant integrates **text, audio, image, and video data** into a unified **Retrieval-Augmented Generation (RAG)** system. It enhances user queries using document search, LLM reasoning, and multimodal embeddings to provide accurate and context-aware answers.

**Concepts & Tools**
- **LangChain** for pipeline orchestration
- **ChromaDB** for vector storage (text/audio/image/video)
- **OpenAI Whisper** for audio/video transcription
- **CLIP** for image embeddings
- **OpenAI / Groq LLMs** for QA and reranking
- **Neo4j** for user-specific chat history
- **RRF (Reciprocal Rank Fusion)** + **LLM-based reranking**

**Workflow**
1) Input (text or media) → 2) Classify (history/follow-up/new)
→ 3) Multimodal retrieval → 4) LLM rerank → 5) Answer
→ 6) Persist history to Neo4j
        """)

    user_id = st.text_input("Enter your User ID", key="user_id_input")
    st.markdown("---")
    st.markdown("### Model Settings")

    model_choice = st.selectbox("Choose LLM Model", ["OpenAI (gpt-4o-mini)", "Groq (llama3-70b-8192)"])
    temperature = st.slider("Set Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max Tokens", 128, 4096, 1024, 64)

    if "OpenAI" in model_choice:
        st.markdown("*`GPT-4o-mini` supports ~128k context; recommended generation: ~2048–4096 tokens.*")
    else:
        st.markdown("*`LLaMA3-70B` (Groq) supports ~8k context; recommended generation: ~2048–4096 tokens.*")

    if st.button("Continue"):
        if user_id.strip():
            st.session_state.user_id = user_id.strip()
            st.session_state.page = "chat"

            if "OpenAI" in model_choice:
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature, max_tokens=max_tokens)
            else:
                llm = ChatGroq(model="llama3-70b-8192", temperature=temperature, max_tokens=max_tokens)

            st.session_state.llm = llm
            st.session_state.qa_chain, st.session_state.memory = create_chain(llm)

            # hydrate memory from Neo4j
            past_chats = load_user_history_from_neo4j(user_id.strip())
            for q, a in past_chats:
                st.session_state.memory.chat_memory.add_user_message(q)
                st.session_state.memory.chat_memory.add_ai_message(a)

            st.rerun()
        else:
            st.warning("User ID cannot be empty.")

# ------------------ Page 2: Chat UI ------------------
elif st.session_state.page == "chat":
    st.title("Ask Your Question")
    st.markdown(f"**User ID:** `{st.session_state.user_id}`")

    if st.button("Sign Out"):
        st.session_state.page = "start"
        st.session_state.user_id = None
        st.session_state.chat_history = []
        del st.session_state.qa_chain
        del st.session_state.memory
        del st.session_state.llm
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        input_type = st.radio("Choose input type", ["Text", "Image", "Audio", "Video"], key="input_type_radio")
    with col2:
        if st.button("Show History"):
            with st.expander("Conversation History", expanded=True):
                for item in st.session_state.chat_history:
                    st.markdown(f"**You:** {item['user']}")
                    label = item['source']
                    label_text = (
                        "**Bot (Document):** " if label == "retriever" else
                        "**LLM-only:** " if label == "llm" else
                        "**Bot (summary):** " if label == "summary" else
                        "**Bot:** "
                    )
                    st.markdown(f"{label_text} {item['bot']}")
            st.stop()

    query = None
    file_path = None

    if input_type == "Text":
        query = st.text_area("Type your question here", height=100)

    elif input_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            # Require a short question with the image (no VLM here)
            query = st.text_area("Ask a question about this image (required)", height=100)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                image.save(tmp.name)
                file_path = tmp.name

    elif input_type == "Audio":
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
        if uploaded_file:
            st.audio(uploaded_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                file_path = tmp.name

    elif input_type == "Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
        if uploaded_file:
            st.video(uploaded_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                file_path = tmp.name

    if st.button("Submit"):
        # Validate inputs
        if input_type == "Text":
            if not query or not query.strip():
                st.warning("Please enter a question.")
                st.stop()
        elif input_type == "Image":
            if not file_path:
                st.warning("Please upload an image.")
                st.stop()
            if not query or not query.strip():
                st.warning("Please add a short question about the image.")
                st.stop()
        else:
            if not file_path:
                st.warning(f"Please upload a {input_type.lower()} file.")
                st.stop()

        # Exit shortcut for text
        if input_type == "Text" and query.strip().lower() == "exit":
            st.session_state.page = "start"
            st.session_state.user_id = None
            st.session_state.chat_history = []
            del st.session_state.qa_chain
            del st.session_state.memory
            del st.session_state.llm
            st.success("You have been logged out.")
            st.rerun()

        with st.spinner("Processing..."):
            memory = st.session_state.memory
            llm = st.session_state.llm
            history_msgs = memory.chat_memory.messages[-6:]

            # Media → transcribe to text
            if input_type in ["Audio", "Video"]:
                try:
                    from openai import OpenAI
                    client = OpenAI()
                    with open(file_path, "rb") as f:
                        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
                    query = (transcript.text or "").strip()
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                    st.stop()

                if not query:
                    st.warning(f"Could not transcribe the {input_type.lower()}. Please try another file.")
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                    st.stop()

            # Detect query type (now always has `query` text)
            qtype = detect_query_type(query, history_msgs)

            # HISTORY request
            if qtype.name == "HISTORY":
                summary = summarize_history(history_msgs, llm)
                memory.chat_memory.add_user_message(query)
                memory.chat_memory.add_ai_message(summary)
                source = "summary"
                answer = summary

            else:
                # FOLLOW_UP rewrite
                final_query = rewrite_query_with_history(query, history_msgs, llm) if qtype.name == "FOLLOW_UP" else query

                retriever = MultimodalRetriever(rerank_llm=llm)
                docs = retriever._get_relevant_documents(final_query)

                if not docs:
                    answer = llm.invoke(final_query).content.strip()
                    memory.chat_memory.add_user_message(query)
                    memory.chat_memory.add_ai_message(answer)
                    save_to_neo4j(
                        st.session_state.user_id,
                        f"{st.session_state.user_id}_streamlit",
                        query,
                        answer,
                        ["LLMOnly"],
                    )
                    source = "llm"
                else:
                    result = st.session_state.qa_chain({"question": final_query})
                    answer = result.get("answer", "").strip()

                    if "i don't know" in answer.lower():
                        answer = llm.invoke(final_query).content.strip()
                        memory.chat_memory.add_user_message(query)
                        memory.chat_memory.add_ai_message(answer)
                        save_to_neo4j(
                            st.session_state.user_id,
                            f"{st.session_state.user_id}_streamlit",
                            query,
                            answer,
                            ["LLMOnly"],
                        )
                        source = "llm"
                    else:
                        memory.chat_memory.add_user_message(query)
                        memory.chat_memory.add_ai_message(answer)
                        save_to_neo4j(
                            st.session_state.user_id,
                            f"{st.session_state.user_id}_streamlit",
                            query,
                            answer,
                            [],
                        )
                        source = "retriever"

            # Nicely show what the user did
            if input_type in ["Audio", "Video"]:
                shown_query = f"[{input_type} transcription] {query[:180]}{'...' if len(query) > 180 else ''}"
            elif input_type == "Image":
                shown_query = f"[Image] {query}"
            else:
                shown_query = query

            st.markdown(f"**You:** {shown_query}")
            label_text = (
                "**Bot (Document):** " if source == "retriever" else
                "**LLM-only:** " if source == "llm" else
                "**Bot (summary):** " if source == "summary" else
                "**Bot:** "
            )
            st.markdown(f"{label_text} {answer}")

            st.session_state.chat_history.append({
                "user": shown_query,
                "bot": answer,
                "source": source
            })

        # Clean up temp file if used
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
