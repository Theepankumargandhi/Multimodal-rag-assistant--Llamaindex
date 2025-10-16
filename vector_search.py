"""
Hybrid RAG implementation using LlamaIndex + LangChain.

- LlamaIndex: Handles retrieval, indexing, query engines
- LangChain: Handles LLM orchestration, Neo4j memory, conversation
"""
import os
import uuid
import re
from enum import Enum
from typing import List, Optional
from neo4j import GraphDatabase

# LangChain imports (for LLM & memory)
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.schema import Document as LangChainDocument
from langchain.memory import ConversationBufferMemory

# LlamaIndex imports (for RAG)
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL,
    CLIP_MODEL,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    TOP_K,
    SIMILARITY_THRESHOLD,
    # Graph-RAG flags
    GRAPH_RAG_ENABLED,
    GRAPH_MAX_HOPS,
    GRAPH_TOP_ENTITIES,
    TRIPLE_CONFIDENCE_MIN,
    RELATION_WHITELIST,
)
from llama_index_setup import (
    init_llama_settings,
    get_or_create_index,
    MultimodalLlamaRetriever
)

# --- Graph-RAG modules (stubs you created) ---
from graph_enrichment import extract_triples, upsert_triples  # noqa: F401
from graph_queries import find_relational_subgraph, format_facts_for_llm  # noqa: F401

__all__ = ["run_multimodal_qa", "init_rag_system"]

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging
logging.set_verbosity_error()

# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

VERBOSE = False

# Global variables for indices (loaded once)
_text_index = None
_audio_index = None
_image_index = None
_multimodal_retriever = None


def init_rag_system(llm_model: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Initialize the RAG system with LlamaIndex indices.
    Call this once at startup.
    """
    global _text_index, _audio_index, _image_index, _multimodal_retriever

    print("🚀 Initializing Hybrid RAG System (LlamaIndex + LangChain)...")

    # Initialize LlamaIndex settings
    init_llama_settings(llm_model=llm_model, temperature=temperature)

    # Load text/document index
    _text_index = get_or_create_index(
        persist_dir=VECTOR_DB_PATH,
        collection_name="documents",
        embed_model=OpenAIEmbedding(model=EMBEDDING_MODEL)
    )

    # Load audio index
    _audio_index = get_or_create_index(
        persist_dir=VECTOR_DB_PATH,
        collection_name="audio",
        embed_model=OpenAIEmbedding(model=EMBEDDING_MODEL)
    )

    # Load image index (with CLIP embeddings)
    _image_index = get_or_create_index(
        persist_dir=VECTOR_DB_PATH,
        collection_name="images",
        embed_model=HuggingFaceEmbedding(model_name=CLIP_MODEL)
    )

    # Create multimodal retriever
    _multimodal_retriever = MultimodalLlamaRetriever(
        text_index=_text_index,
        audio_index=_audio_index,
        image_index=_image_index,
        similarity_top_k=TOP_K,
        similarity_cutoff=SIMILARITY_THRESHOLD
    )

    print("✅ RAG System initialized successfully!")
    return _multimodal_retriever


def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Get LangChain LLM instance.
    Supports: gpt-4o-mini, llama-3.3-70b-versatile
    """
    if "gpt" in model_name.lower():
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif "llama" in model_name.lower():
        return ChatGroq(model=model_name, temperature=temperature)
    else:
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


def multimodal_search(query: str) -> List[LangChainDocument]:
    """
    Search across all modalities using LlamaIndex retriever.
    Returns LangChain documents for compatibility.
    """
    if _multimodal_retriever is None:
        init_rag_system()

    # Retrieve using LlamaIndex
    llama_nodes = _multimodal_retriever.retrieve(query)

    # Convert LlamaIndex nodes to LangChain documents
    langchain_docs = []
    for node in llama_nodes:
        doc = LangChainDocument(
            page_content=node.text,
            metadata=node.metadata
        )
        langchain_docs.append(doc)

    if VERBOSE:
        print(f"🔍 Retrieved {len(langchain_docs)} documents")

    return langchain_docs


def rerank_with_llm(
    query: str,
    docs: List[LangChainDocument],
    top_n: int = 3,
    llm=None
) -> List[LangChainDocument]:
    """
    LLM-based reranking (keeping LangChain implementation).
    """
    if not docs or not llm:
        return docs[:top_n]

    prompt = f"You are a reranker. Query: {query}\n\nDocuments:\n"
    for i, doc in enumerate(docs, start=1):
        snippet = doc.page_content.replace("\n", " ")[:400]
        prompt += f"{i}. {snippet}\n"

    prompt += f"\nReturn the top {top_n} most relevant document numbers in order (comma-separated)."

    try:
        resp = llm.invoke(prompt).content
        indices = [int(x) for x in re.findall(r"\d+", resp)]
        reranked = [docs[i-1] for i in indices if 1 <= i <= len(docs)]
        return reranked[:top_n] if reranked else docs[:top_n]
    except Exception as e:
        if VERBOSE:
            print(f"⚠️ Reranking failed: {e}")
        return docs[:top_n]


def save_to_neo4j(
    user_id: str,
    session_id: str,
    query: str,
    answer: str,
    sources: list
):
    """Save conversation to Neo4j (LangChain integration)."""
    try:
        with driver.session() as session:
            session.run(
                """
                MERGE (u:User {id: $user_id})
                MERGE (s:Session {id: $session_id, date: date()})
                MERGE (u)-[:HAS_SESSION]->(s)
                MERGE (q:Query {text: $query, timestamp: datetime()})
                MERGE (a:Answer {text: $answer})
                MERGE (s)-[:ASKED]->(q)-[:ANSWERED_BY]->(a)
                """,
                parameters={
                    "user_id": user_id,
                    "session_id": session_id,
                    "query": query,
                    "answer": answer
                }
            )

            for src in sources:
                session.run(
                    """
                    MERGE (src:Source {type: $type})
                    MERGE (a:Answer {text: $answer})
                    MERGE (a)-[:USED_SOURCE]->(src)
                    """,
                    parameters={"type": src, "answer": answer}
                )
    except Exception as e:
        print(f"⚠️ Neo4j save failed: {e}")


class QueryType(Enum):
    HISTORY = "history"
    FOLLOW_UP = "follow_up"
    DOCUMENT = "document"


def detect_query_type(query: str, history_msgs) -> QueryType:
    """Detect if query is about history, follow-up, or new document search."""
    text = query.lower().strip()

    # History patterns
    history_patterns = [
        r"what.*we.*\b(discuss|talk|cover|speak|spoke)\b",
        r"on what topic.*we.*\b(speak|spoke|discuss|covered)\b",
        r"what.*previous.*\b(conversation|discussion)\b",
        r"tell me.*\b(earlier|so far|before)\b",
        r"remind me", r"i forgot",
        r"show me our \b(chat|conversation|history)\b",
        r"what topics.*we.*(spoke|discussed)"
    ]
    if any(re.search(p, text) for p in history_patterns):
        return QueryType.HISTORY

    # Follow-up patterns
    follow_up_patterns = [
        r"\b(this|it|that|these|those)\b",
        r"what about",
        r"what is the use of",
        r"how does it work"
    ]
    if history_msgs and any(re.search(p, text) for p in follow_up_patterns):
        return QueryType.FOLLOW_UP

    return QueryType.DOCUMENT


def summarize_history(history_msgs, llm) -> str:
    """Summarize conversation history using LLM."""
    if not history_msgs:
        return "We haven't had any conversation yet in this session."

    history_text = "\n".join(
        f"User: {m.content}" if getattr(m, 'type', '') == 'human'
        else f"Bot: {m.content or getattr(m, 'page_content', '')}"
        for m in history_msgs[-10:]
    )

    prompt = f"""Summarize the key topics from this conversation history:

{history_text}

Provide a concise summary of what we discussed."""
    return llm.invoke(prompt).content.strip()


def rewrite_query_with_history(
    query: str,
    history_msgs,
    question_rewriter
) -> str:
    """Rewrite follow-up questions into standalone queries."""
    hist = []
    for m in history_msgs[-6:]:
        if getattr(m, 'type', '') == 'human':
            hist.append(f"User: {m.content}")
        else:
            hist.append(f"Bot: {m.content or getattr(m, 'page_content', '')}")

    history_text = "\n".join(hist)

    prompt = f"""Conversation history:
{history_text}

Current question: "{query}"

Rewrite this question into a standalone, self-contained question that includes necessary context from the history."""
    return question_rewriter.invoke(prompt).content.strip()


# --- graph: simple relational query detector ---
def is_relational_query(q: str) -> bool:
    ql = (q or "").lower()
    triggers = ["how", "why", "relationship", "connected", "between", "link", "association", "relate"]
    return any(t in ql for t in triggers)


def run_multimodal_qa(
    user_id: str,
    query: str,
    input_type: str,
    file_path: str = None,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.0
):
    """
    Main QA function using hybrid LlamaIndex + LangChain approach.

    Args:
        user_id: User identifier
        query: User's question
        input_type: "text", "audio", "image", or "video"
        file_path: Optional file path for media
        llm_model: LLM model name
        temperature: LLM temperature
    """
    # Initialize system if needed
    if _multimodal_retriever is None:
        init_rag_system(llm_model=llm_model, temperature=temperature)

    # Get LLM
    llm = get_llm(model_name=llm_model, temperature=temperature)

    # Create session
    session_id = f"{user_id}_{uuid.uuid4().hex[:6]}"

    # Conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # --- Load last few messages from Neo4j for continuity ---
    with driver.session() as session:
        results = session.run("""
            MATCH (u:User {id: $user_id})-[:HAS_SESSION]->(s:Session)-[:ASKED]->(q:Query)-[:ANSWERED_BY]->(a:Answer)
            RETURN q.text AS question, a.text AS answer
            ORDER BY q.timestamp DESC
            LIMIT 6
        """, parameters={"user_id": user_id})
        for record in results:
            memory.chat_memory.add_user_message(record["question"])
            memory.chat_memory.add_ai_message(record["answer"])

    history_msgs = memory.chat_memory.messages[-6:]

    # Detect query type
    qtype = detect_query_type(query, history_msgs)

    # Handle history queries
    if qtype == QueryType.HISTORY:
        summary = summarize_history(history_msgs, llm)
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(summary)
        save_to_neo4j(user_id, session_id, query, summary, ["ChatHistory"])
        return {"answer": summary, "source": "summary"}

    # Rewrite follow-up queries
    final_query = query
    if qtype == QueryType.FOLLOW_UP:
        final_query = rewrite_query_with_history(query, history_msgs, llm)
        if VERBOSE:
            print(f"🔄 Rewritten query: {final_query}")

    # Retrieve documents using LlamaIndex
    docs = multimodal_search(final_query)

    # If no documents found, use LLM directly
    if not docs:
        answer = llm.invoke(final_query).content.strip()
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(answer)
        save_to_neo4j(user_id, session_id, query, answer, ["LLMOnly"])
        return {"answer": answer, "source": "llm"}

    # Rerank documents
    reranked_docs = rerank_with_llm(final_query, docs, top_n=3, llm=llm)

    # ===========================
    # Graph-RAG: Enrichment hook
    # ===========================
    graph_facts = []
    graph_context = ""
    try:
        if GRAPH_RAG_ENABLED:
            # Build simple chunk dicts for extraction
            top_chunks = []
            for i, d in enumerate(reranked_docs):
                text = getattr(d, "page_content", "") or getattr(d, "text", "")
                md = getattr(d, "metadata", {}) or {}
                top_chunks.append({
                    "text": text,
                    "doc_id": md.get("doc_id", md.get("source", f"doc{i}")),
                    "chunk_id": md.get("chunk_id", str(i)),
                })

            # Extract and filter triples
            triples = extract_triples(top_chunks) or []
            triples = [
                t for t in triples
                if getattr(t, "confidence", 0.0) >= TRIPLE_CONFIDENCE_MIN
                and getattr(t, "rel", "").upper() in {r.upper() for r in RELATION_WHITELIST}
            ]

            if triples:
                upsert_triples(triples, driver)
    except Exception as ge:
        if VERBOSE:
            print(f"⚠️ Graph enrichment failed (non-blocking): {ge}")

    # ===========================
    # Build context from documents
    # ===========================
    context = "\n\n".join([
        f"Document {i+1} (Source: {doc.metadata.get('source_type', 'Unknown')}):\n{doc.page_content}"
        for i, doc in enumerate(reranked_docs)
    ])

    # ==========================================
    # Graph-RAG: Graph query + context fusion
    # ==========================================
    try:
        if GRAPH_RAG_ENABLED and is_relational_query(final_query):
            facts = find_relational_subgraph(
                final_query,
                driver,
                max_hops=GRAPH_MAX_HOPS,
                top_entities=GRAPH_TOP_ENTITIES
            ) or []
            graph_facts = facts
            graph_context = format_facts_for_llm(facts)
    except Exception as gq:
        if VERBOSE:
            print(f"⚠️ Graph query failed (non-blocking): {gq}")

    # Generate answer using RAG (+ optional graph context)
    rag_prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If "Graph Context" is present, treat those as high-confidence relationship facts and cite them.

Context:
{context}

{graph_context if graph_context else ""}

Question: {final_query}

Answer:"""

    final_answer = llm.invoke(rag_prompt).content.strip()

    # Fallback to LLM if answer is uncertain
    if re.match(r"(?i)i\s+don'?t\s+know", final_answer):
        final_answer = llm.invoke(final_query).content.strip()
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(final_answer)
        save_to_neo4j(user_id, session_id, query, final_answer, ["LLMOnly"])
        return {"answer": final_answer, "source": "llm"}

    # Save to memory and Neo4j
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(final_answer)

    source_types = list(set([doc.metadata.get('source_type', 'Unknown') for doc in reranked_docs]))
    save_to_neo4j(user_id, session_id, query, final_answer, source_types)

    return {
        "answer": final_answer,
        "source": "retriever",
        "source_types": source_types,
        "graph_facts": graph_facts if GRAPH_RAG_ENABLED else [],
        "graph_enabled": GRAPH_RAG_ENABLED,
    }


# For quick testing
if __name__ == "__main__":
    # Initialize system
    init_rag_system()

    # Test query
    query = input("Enter your search query: ").strip()

    result = run_multimodal_qa(
        user_id="test_user",
        query=query,
        input_type="text"
    )

    print(f"\n📝 Answer: {result['answer']}")
    print(f"📍 Source: {result['source']}")
