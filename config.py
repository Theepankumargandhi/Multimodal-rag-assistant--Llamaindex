import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys (NEVER hardcode these!)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", None)  # Optional for advanced reranking

# Set environment variables for libraries
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Data Paths
DOCS_PATH = "data/docs"
AUDIO_PATH = "data/audio"
VIDEO_PATH = "data/video"
IMAGE_PATH = "data/images"

# Vector Database
VECTOR_DB_PATH = "vectorstore"

# Embedding Models
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI
CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"  # For images

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# LlamaIndex Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 10  # Number of results to retrieve
SIMILARITY_THRESHOLD = 0.3

# LLM Settings
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024