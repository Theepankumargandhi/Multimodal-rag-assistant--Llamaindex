"""
Image ingestion using LlamaIndex.
Uses CLIP embeddings for image search.
"""
import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document as LlamaDocument
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

from config import IMAGE_PATH, VECTOR_DB_PATH, CLIP_MODEL
from llama_index_setup import init_llama_settings


def ingest_images():
    """
    Ingest images using LlamaIndex with CLIP embeddings.
    Stores image paths as documents.
    """
    print("🖼️  Starting image ingestion with LlamaIndex...")
    
    # Initialize LlamaIndex (but we'll override embed_model for images)
    init_llama_settings()
    
    persist_dir = VECTOR_DB_PATH
    
    if not os.path.exists(IMAGE_PATH):
        print(f"⚠️ Directory not found: {IMAGE_PATH}")
        return
    
    # Find image files
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        image_files.extend(Path(IMAGE_PATH).glob(f"*{ext}"))
    
    if not image_files:
        print(f"⚠️ No image files found in {IMAGE_PATH}")
        return
    
    print(f"📂 Found {len(image_files)} image file(s)")
    
    # Create documents with image paths
    documents = []
    for img_file in image_files:
        try:
            # Store image path as document content
            # CLIP will generate embeddings from the actual image
            doc = LlamaDocument(
                text=str(img_file.resolve()),  # Store full path
                metadata={
                    "source": img_file.name,
                    "source_type": "Image",
                    "file_path": str(img_file.resolve())
                }
            )
            documents.append(doc)
            print(f"✅ Added: {img_file.name}")
            
        except Exception as e:
            print(f"⚠️ Failed to process {img_file.name}: {e}")
    
    if not documents:
        print("⚠️ No images to ingest!")
        return
    
    try:
        # Initialize ChromaDB
        os.makedirs(persist_dir, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Create or get collection for images
        chroma_collection = chroma_client.get_or_create_collection("images")
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create CLIP embedding model for images
        embed_model = HuggingFaceEmbedding(model_name=CLIP_MODEL)
        
        # Create index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        print(f"✅ Successfully ingested {len(documents)} image(s) into ChromaDB")
        print(f"📍 Location: {persist_dir}")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    ingest_images()