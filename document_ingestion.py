"""
Document ingestion using LlamaIndex.
Supports PDF and TXT files.
"""
import os
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

from config import DOCS_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL
from llama_index_setup import init_llama_settings

def ingest_documents():
    """
    Ingest documents using LlamaIndex SimpleDirectoryReader.
    Automatically handles PDF and TXT files.
    """
    print("📄 Starting document ingestion with LlamaIndex...")
    
    # Initialize LlamaIndex
    init_llama_settings()
    
    if not os.path.exists(DOCS_PATH):
        print(f"⚠️ Directory not found: {DOCS_PATH}")
        return
    
    # Check if directory has files
    files = list(Path(DOCS_PATH).glob("*"))
    if not files:
        print(f"⚠️ No files found in {DOCS_PATH}")
        return
    
    print(f"📂 Found {len(files)} file(s) in {DOCS_PATH}")
    
    try:
        # Load documents using SimpleDirectoryReader
        # This automatically handles PDF, TXT, DOCX, etc.
        reader = SimpleDirectoryReader(
            input_dir=DOCS_PATH,
            recursive=False,
            required_exts=[".pdf", ".txt"]  # Only PDF and TXT
        )
        documents = reader.load_data()
        
        if not documents:
            print("⚠️ No documents loaded!")
            return
        
        print(f"✅ Loaded {len(documents)} document(s)")
        
        # Add metadata
        for doc in documents:
            doc.metadata["source_type"] = "Document"
        
        # Initialize ChromaDB
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        
        # Create or get collection
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
        
        print(f"✅ Successfully ingested {len(documents)} document(s) into ChromaDB")
        print(f"📍 Location: {VECTOR_DB_PATH}")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    ingest_documents()