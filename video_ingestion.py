"""
Video ingestion using LlamaIndex.
Directly transcribes video files using OpenAI Whisper API and stores in ChromaDB.
"""
import os
from pathlib import Path
from openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document as LlamaDocument
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

from config import VIDEO_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL
from llama_index_setup import init_llama_settings

client = OpenAI()
MAX_FILE_SIZE_MB = 25  # Whisper API limit


def transcribe_video(video_file: str) -> str:
    """
    Transcribe a video file using OpenAI Whisper API.
    Returns None if file exceeds size limit or transcription fails.
    """
    try:
        size_mb = os.path.getsize(video_file) / (1024 * 1024)
        
        if size_mb > MAX_FILE_SIZE_MB:
            print(f"⚠️ Skipping {os.path.basename(video_file)} ({size_mb:.2f}MB) — exceeds Whisper {MAX_FILE_SIZE_MB}MB limit")
            return None
        
        with open(video_file, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=f
            )
        return transcript.text
        
    except Exception as e:
        print(f"❌ Transcription failed for {os.path.basename(video_file)}: {e}")
        return None


def ingest_videos():
    """
    Ingest video files using LlamaIndex.
    Transcribes videos and stores transcripts in ChromaDB.
    """
    print("🎬 Starting video ingestion with LlamaIndex...")
    
    # Initialize LlamaIndex
    init_llama_settings()
    
    if not os.path.exists(VIDEO_PATH):
        print(f"⚠️ Directory not found: {VIDEO_PATH}")
        return
    
    # Find video files
    video_files = [
        os.path.join(VIDEO_PATH, f) 
        for f in os.listdir(VIDEO_PATH) 
        if f.endswith((".mp4", ".mov", ".avi", ".mkv"))
    ]
    
    if not video_files:
        print(f"⚠️ No video files found in {VIDEO_PATH}")
        return
    
    print(f"📂 Found {len(video_files)} video file(s)")
    
    # Transcribe all videos
    documents = []
    for video_file in video_files:
        file_name = os.path.basename(video_file)
        print(f"🎥 Processing video: {file_name}")
        
        transcript_text = transcribe_video(video_file)
        
        if not transcript_text or not transcript_text.strip():
            print(f"⚠️ Empty transcript for {file_name}, skipping...")
            continue
        
        # Create LlamaIndex document
        doc = LlamaDocument(
            text=transcript_text,
            metadata={
                "source": file_name,
                "source_type": "Video",
                "file_path": video_file
            }
        )
        documents.append(doc)
        print(f"✅ Transcribed: {file_name} ({len(transcript_text)} chars)")
    
    if not documents:
        print("⚠️ No video transcripts to ingest!")
        return
    
    try:
        # Initialize ChromaDB
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        
        # Store in the SAME collection as documents (matching original behavior)
        chroma_collection = chroma_client.get_or_create_collection("documents")
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create embedding model
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
        
        # Create index and store
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        print(f"✅ Successfully ingested {len(documents)} video transcript(s) into ChromaDB")
        print(f"📍 Location: {VECTOR_DB_PATH}")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    ingest_videos()