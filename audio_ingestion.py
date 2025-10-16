"""
Audio ingestion using LlamaIndex.
Transcribes audio files using OpenAI Whisper and stores in ChromaDB.
"""
import os
from pathlib import Path
from openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document as LlamaDocument
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

from config import AUDIO_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL
from llama_index_setup import init_llama_settings

client = OpenAI()

def transcribe_audio(audio_file: str) -> str:
    """Transcribe an audio file using OpenAI Whisper API."""
    try:
        with open(audio_file, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=f
            )
        return transcript.text
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None


def ingest_audio_files():
    """
    Ingest audio files using LlamaIndex.
    """
    print("🎤 Starting audio ingestion with LlamaIndex...")
    
    # Initialize LlamaIndex
    init_llama_settings()
    
    persist_dir = VECTOR_DB_PATH
    
    if not os.path.exists(AUDIO_PATH):
        print(f"⚠️ Directory not found: {AUDIO_PATH}")
        return
    
    # Find audio files
    audio_files = []
    for ext in [".mp3", ".wav", ".m4a", ".ogg"]:
        audio_files.extend(Path(AUDIO_PATH).glob(f"*{ext}"))
    
    if not audio_files:
        print(f"⚠️ No audio files found in {AUDIO_PATH}")
        return
    
    print(f"📂 Found {len(audio_files)} audio file(s)")
    
    # Transcribe all audio files
    documents = []
    for audio_file in audio_files:
        print(f"🎵 Processing: {audio_file.name}")
        
        transcript_text = transcribe_audio(str(audio_file))
        
        if not transcript_text or not transcript_text.strip():
            print(f"⚠️ Empty transcript for {audio_file.name}, skipping...")
            continue
        
        # Create LlamaIndex document
        doc = LlamaDocument(
            text=transcript_text,
            metadata={
                "source": audio_file.name,
                "source_type": "Audio",
                "file_path": str(audio_file.resolve())
            }
        )
        documents.append(doc)
        print(f"✅ Transcribed: {audio_file.name} ({len(transcript_text)} chars)")
    
    if not documents:
        print("⚠️ No transcripts to ingest!")
        return
    
    try:
        # Initialize ChromaDB
        os.makedirs(persist_dir, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Create or get collection
        chroma_collection = chroma_client.get_or_create_collection("audio")
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create embedding model
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
        
        # Create index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        print(f"✅ Successfully ingested {len(documents)} audio transcript(s) into ChromaDB")
        print(f"📍 Location: {persist_dir}")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    ingest_audio_files()