# config.py
import os
from dotenv import load_dotenv

# Load .env once
load_dotenv()

def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}

# ===============================
# API Keys (NEVER hardcode these)
# ===============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", None)  # Optional for reranking

# Propagate to libs that read from env
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ===============================
# Data Paths
# ===============================
DOCS_PATH = "data/docs"
AUDIO_PATH = "data/audio"
VIDEO_PATH = "data/video"
IMAGE_PATH = "data/images"

# ===============================
# Vector Database
# ===============================
VECTOR_DB_PATH = "vectorstore"

# ===============================
# Embedding Models
# ===============================
# Tip: if you're on newer OpenAI embeddings, consider "text-embedding-3-small" or "text-embedding-3-large"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
CLIP_MODEL = os.getenv("CLIP_MODEL", "sentence-transformers/clip-ViT-B-32")  # for images

# ===============================
# Neo4j Configuration
# ===============================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ===============================
# LlamaIndex / Retrieval Settings
# ===============================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "10"))  # number of results to retrieve
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# ===============================
# LLM Settings
# ===============================
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))

# ===============================
# Graph-RAG Enrichment Settings
# ===============================
GRAPH_RAG_ENABLED = _get_bool("GRAPH_RAG_ENABLED", True)   # UI can toggle this at runtime
GRAPH_MAX_HOPS = int(os.getenv("GRAPH_MAX_HOPS", "2"))
GRAPH_TOP_ENTITIES = int(os.getenv("GRAPH_TOP_ENTITIES", "5"))
TRIPLE_CONFIDENCE_MIN = float(os.getenv("TRIPLE_CONFIDENCE_MIN", "0.55"))
RELATION_WHITELIST = [
    r.strip().upper()
    for r in os.getenv(
        "RELATION_WHITELIST",
        "USES,PART_OF,CAUSES,RELATES_TO,PARTNERS_WITH,INVESTS_IN,DEVELOPS,INTEGRATES"
    ).split(",")
    if r.strip()
]

# ===============================
# Misc
# ===============================
VERBOSE = _get_bool("VERBOSE", False)
