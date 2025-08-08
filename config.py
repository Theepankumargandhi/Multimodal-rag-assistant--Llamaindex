# config.py
import os

# Set OpenAI Key
os.environ["OPENAI_API_KEY"] = " "
# Paths
DOCS_PATH = "data/docs"
AUDIO_PATH = "data/audio"
VIDEO_PATH = "data/video"
IMAGE_PATH="data/images"
VECTOR_DB_PATH = "vectorstore"

# Embedding Model
EMBEDDING_MODEL = "text-embedding-ada-002"
NEO4J_URI = " "   # Or AuraDB URI
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = ""
