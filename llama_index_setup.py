"""
LlamaIndex initialization and core RAG components.
This file handles all LlamaIndex-specific setup.
"""
import os
from typing import List, Optional
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage,
    Settings
)
from llama_index.core.schema import Document as LlamaDocument
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

import chromadb

from config import (
    VECTOR_DB_PATH, 
    EMBEDDING_MODEL, 
    CLIP_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    SIMILARITY_THRESHOLD
)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    "init_llama_settings",
    "get_or_create_index",
    "create_multimodal_query_engine",
    "MultimodalLlamaRetriever"
]


def init_llama_settings(llm_model: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Initialize global LlamaIndex settings.
    
    Args:
        llm_model: "gpt-4o-mini" or "llama-3.3-70b-versatile"
        temperature: LLM temperature
    """
    # Set embedding model
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
    
    # Set LLM
    if "gpt" in llm_model.lower():
        Settings.llm = OpenAI(model=llm_model, temperature=temperature)
    elif "llama" in llm_model.lower():
        Settings.llm = Groq(model=llm_model, temperature=temperature)
    else:
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=temperature)
    
    # Set chunk settings
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    
    print(f"✅ LlamaIndex initialized with {llm_model}")


def get_or_create_index(
    persist_dir: str,
    collection_name: str = "default",
    embed_model = None
) -> Optional[VectorStoreIndex]:
    """
    Load existing index or return None if doesn't exist.
    
    Args:
        persist_dir: Path to ChromaDB directory
        collection_name: Chroma collection name
        embed_model: Embedding model (defaults to Settings.embed_model)
    """
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        print(f"⚠️ No index found at {persist_dir}")
        return None
    
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_collection(collection_name)
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model or Settings.embed_model
        )
        
        print(f"✅ Loaded index from {persist_dir}")
        return index
        
    except Exception as e:
        print(f"⚠️ Failed to load index from {persist_dir}: {e}")
        return None


def create_multimodal_query_engine(
    text_index: Optional[VectorStoreIndex],
    audio_index: Optional[VectorStoreIndex],
    image_index: Optional[VectorStoreIndex],
    similarity_top_k: int = TOP_K,
    similarity_cutoff: float = SIMILARITY_THRESHOLD
):
    """
    Create a multimodal query engine that searches across all modalities.
    
    Returns a custom retriever that fuses results from all available indices.
    """
    return MultimodalLlamaRetriever(
        text_index=text_index,
        audio_index=audio_index,
        image_index=image_index,
        similarity_top_k=similarity_top_k,
        similarity_cutoff=similarity_cutoff
    )


class MultimodalLlamaRetriever:
    """
    Custom retriever that performs fusion across text, audio, and image indices.
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """
    
    def __init__(
        self,
        text_index: Optional[VectorStoreIndex],
        audio_index: Optional[VectorStoreIndex],
        image_index: Optional[VectorStoreIndex],
        similarity_top_k: int = 10,
        similarity_cutoff: float = 0.3
    ):
        self.text_index = text_index
        self.audio_index = audio_index
        self.image_index = image_index
        self.similarity_top_k = similarity_top_k
        self.similarity_cutoff = similarity_cutoff
        
        # Create retrievers for each modality
        self.text_retriever = self._create_retriever(text_index, "Document")
        self.audio_retriever = self._create_retriever(audio_index, "Audio")
        self.image_retriever = self._create_retriever(image_index, "Image")
    
    def _create_retriever(self, index: Optional[VectorStoreIndex], source_type: str):
        """Create a retriever with similarity postprocessor."""
        if index is None:
            return None
        
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.similarity_top_k
        )
        
        # Add similarity filtering
        postprocessor = SimilarityPostprocessor(
            similarity_cutoff=self.similarity_cutoff
        )
        
        return {"retriever": retriever, "postprocessor": postprocessor, "source": source_type}
    
    def retrieve(self, query: str) -> List[LlamaDocument]:
        """
        Retrieve documents from all modalities and fuse using RRF.
        """
        all_results = []
        
        # Retrieve from each modality
        for retriever_config in [self.text_retriever, self.audio_retriever, self.image_retriever]:
            if retriever_config is None:
                continue
            
            try:
                retriever = retriever_config["retriever"]
                postprocessor = retriever_config["postprocessor"]
                source_type = retriever_config["source"]
                
                # Retrieve nodes
                nodes = retriever.retrieve(query)
                
                # Apply similarity filtering
                filtered_nodes = postprocessor.postprocess_nodes(nodes)
                
                # Add source type to metadata
                for node in filtered_nodes:
                    node.metadata["source_type"] = source_type
                
                all_results.append((source_type, filtered_nodes))
                
            except Exception as e:
                print(f"⚠️ Retrieval failed for {retriever_config['source']}: {e}")
        
        # Fuse results using Reciprocal Rank Fusion
        fused_docs = self._reciprocal_rank_fusion(all_results)
        
        return fused_docs
    
    def _reciprocal_rank_fusion(self, results: List[tuple], k: int = 60) -> List[LlamaDocument]:
        """
        Combine results from multiple retrievers using RRF.
        
        Args:
            results: List of (source_name, nodes) tuples
            k: RRF constant (default 60)
        """
        combined = {}
        
        for source_name, nodes in results:
            for rank, node in enumerate(nodes, start=1):
                rr_score = 1.0 / (rank + k)
                
                # Use node text as key for deduplication
                key = node.text
                
                if key not in combined:
                    combined[key] = {
                        "node": node,
                        "score": 0.0
                    }
                
                combined[key]["score"] += rr_score
        
        # Sort by score
        sorted_items = sorted(
            combined.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        # Return top documents
        return [item["node"] for item in sorted_items[:self.similarity_top_k]]
    
    def as_query_engine(self):
        """
        Create a query engine from this retriever.
        """
        return RetrieverQueryEngine.from_args(
            retriever=self,
            response_mode="compact"
        )