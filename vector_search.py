import os
import uuid
import re
from enum import Enum
from typing import List
from neo4j import GraphDatabase

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document, BaseRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

from config import VECTOR_DB_PATH, EMBEDDING_MODEL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

__all__ = ["llm", "multimodal_search", "rerank_with_llm", "save_to_neo4j", "run_multimodal_qa"]

# Set API Keys
if os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Suppress transformers warnings
from transformers.utils import logging
logging.set_verbosity_error()

# Embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
image_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/clip-ViT-B-32")

# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

VERBOSE = False

def load_chroma_if_exists(path, embedding_func):
    if os.path.exists(path) and os.listdir(path):
        if VERBOSE:
            print(f"Loading vector store from: {path}")
        return Chroma(persist_directory=path, embedding_function=embedding_func)
    if VERBOSE:
        print(f"⚠️ Skipping vector store at {path} (not found or empty)")
    return None

# Vector stores
text_video_store = load_chroma_if_exists(VECTOR_DB_PATH, embeddings)
audio_store = load_chroma_if_exists(os.path.join(VECTOR_DB_PATH, "audio_db"), embeddings)
image_store = load_chroma_if_exists(os.path.join(VECTOR_DB_PATH, "image_db"), image_embeddings)

def reciprocal_rank_fusion(tv_results, audio_results, image_results, k: int = 10) -> List[Document]:
    combined = {}
    for source_name, results in [
        ("Document", tv_results),
        ("Audio", audio_results),
        ("Image", image_results)
    ]:
        for rank, (doc, score) in enumerate(results, start=1):
            rr = 1 / (rank + 60)
            key = doc.page_content
            if key not in combined:
                combined[key] = {"doc": doc, "score": 0}
            combined[key]["score"] += rr
            combined[key]["doc"].metadata["source_type"] = source_name
    sorted_docs = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs[:k]]

def multimodal_search(query: str, k: int = 10, threshold: float = 0.3) -> List[Document]:
    tv = [(d, s) for d, s in (text_video_store.similarity_search_with_score(query, k=k) if text_video_store else []) if s >= threshold]
    aud = [(d, s) for d, s in (audio_store.similarity_search_with_score(query, k=k) if audio_store else []) if s >= threshold]
    img = [(d, s) for d, s in (image_store.similarity_search_with_score(query, k=k) if image_store else []) if s >= threshold]
    return reciprocal_rank_fusion(tv, aud, img, k)

def rerank_with_llm(query: str, docs: List[Document], top_n: int = 3, llm=None) -> List[Document]:
    if not docs or not llm:
        return docs[:top_n]
    prompt = f"You are a reranker. Query: {query}\nDocuments:\n"
    for i, doc in enumerate(docs, start=1):
        snippet = doc.page_content.replace("\n", " ")[:400]
        prompt += f"{i}. {snippet}\n"
    prompt += "\nReturn top 3 document numbers in order."
    resp = llm.invoke(prompt).content
    try:
        indices = [int(x) for x in re.findall(r"\d+", resp)]
        return [docs[i-1] for i in indices if 1 <= i <= len(docs)]
    except:
        return docs[:top_n]

class MultimodalRetriever(BaseRetriever):
    def __init__(self, rerank_llm):
        super().__init__()
        self._rerank_llm = rerank_llm

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = multimodal_search(query)
        return rerank_with_llm(query, docs, llm=self._rerank_llm)

def summarize_history(history_msgs, llm) -> str:
    if not history_msgs:
        return "We haven’t had any conversation yet in this session."
    history_text = "\n".join(
        (f"User: {m.content}" if getattr(m, 'type', '') == 'human' else f"Bot: {m.content or m.page_content}")
        for m in history_msgs[-10:]
    )
    prompt = f"Summarize the key topics from this conversation history:\n{history_text}\nProvide a concise summary."
    return llm.invoke(prompt).content.strip()

def create_chain(llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=MultimodalRetriever(rerank_llm=llm),
        memory=memory,
        condense_question_llm=llm,
        return_source_documents=False
    )
    return chain, memory

def save_to_neo4j(user_id: str, session_id: str, query: str, answer: str, sources: list[str]):
    with driver.session() as session:
        session.run(
            """
            MERGE (u:User {id: $user_id})
            MERGE (s:Session {id: $session_id, date: date()})
            MERGE (u)-[:HAS_SESSION]->(s)
            MERGE (q:Query {text: $query, timestamp: datetime()})
            MERGE (a:Answer {text: $answer})
            MERGE (s)-[:ASKED]->(q)-[:ANSWERED_BY]->(a)
            """, parameters={"user_id": user_id, "session_id": session_id, "query": query, "answer": answer}
        )
        for src in sources:
            session.run(
                """
                MERGE (src:Source {type: $type})
                MERGE (a:Answer {text: $answer})
                MERGE (a)-[:USED_SOURCE]->(src)
                """, parameters={"type": src, "answer": answer}
            )

class QueryType(Enum):
    HISTORY = "history"
    FOLLOW_UP = "follow_up"
    DOCUMENT = "document"

def detect_query_type(query: str, history_msgs) -> QueryType:
    text = query.lower().strip()
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
    follow_up_patterns = [r"\b(this|it|that|these|those)\b", r"what about", r"what is the use of", r"how does it work"]
    if history_msgs and any(re.search(p, text) for p in follow_up_patterns):
        return QueryType.FOLLOW_UP
    return QueryType.DOCUMENT

def rewrite_query_with_history(query: str, history_msgs, question_rewriter) -> str:
    hist = []
    for m in history_msgs[-6:]:
        if getattr(m, 'type', '') == 'human':
            hist.append(f"User: {m.content}")
        else:
            hist.append(f"Bot: {m.content or m.page_content}")
    history_text = "\n".join(hist)
    prompt = f"Conversation history:\n{history_text}\nRewrite the question '{query}' into a standalone question."
    return question_rewriter.invoke(prompt).content.strip()

def run_multimodal_qa(user_id: str, query: str, input_type: str, file_path: str = None, llm=None):
    session_id = f"{user_id}_{uuid.uuid4().hex[:6]}"
    qa_chain, memory = create_chain(llm)
    history_msgs = memory.chat_memory.messages[-6:]

    qtype = detect_query_type(query, history_msgs)

    if qtype == QueryType.HISTORY:
        summary = summarize_history(history_msgs, llm)
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(summary)
        save_to_neo4j(user_id, session_id, query, summary, ["ChatHistory"])
        return {"answer": summary, "source": "summary"}

    final_query = rewrite_query_with_history(query, history_msgs, llm) if qtype == QueryType.FOLLOW_UP else query
    docs = MultimodalRetriever(rerank_llm=llm)._get_relevant_documents(final_query)

    if not docs:
        answer = llm.invoke(final_query).content.strip()
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(answer)
        save_to_neo4j(user_id, session_id, query, answer, ["LLMOnly"])
        return {"answer": answer, "source": "llm"}

    result = qa_chain({"question": final_query})
    final_answer = result.get("answer", "").strip()

    if re.match(r"(?i)i\s+don'?t\s+know", final_answer):
        final_answer = llm.invoke(final_query).content.strip()
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(final_answer)
        save_to_neo4j(user_id, session_id, query, final_answer, ["LLMOnly"])
        return {"answer": final_answer, "source": "llm"}

    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(final_answer)
    save_to_neo4j(user_id, session_id, query, final_answer, [])
    return {"answer": final_answer, "source": "retriever"}

# For quick local testing only
if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    query = input("Enter your search query: ").strip()
    results = multimodal_search(query)
    if not results:
        print(" No results found in any modality.")
    else:
        print(f"\n Found {len(results)} results:")
        for i, doc in enumerate(results, start=1):
            print(f"\n--- Result {i} ---")
            print(f"Source Type: {doc.metadata.get('source_type', 'Unknown')}")
            print(f"Content Preview: {doc.page_content[:200]}...")
