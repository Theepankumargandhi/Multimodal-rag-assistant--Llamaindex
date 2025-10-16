"""
LlamaIndex initialization and core RAG components.
This file handles all LlamaIndex-specific setup.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from llama_index.core import (
    VectorStoreIndex,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

import chromadb

from config import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL,
    CLIP_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    SIMILARITY_THRESHOLD,
)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    "init_llama_settings",
    "get_or_create_index",
    "create_multimodal_query_engine",
    "MultimodalLlamaRetriever",
]


# ------------------------------
# Small wrapper for downstream use
# ------------------------------
@dataclass
class SimpleDoc:
    """Thin wrapper so downstream code can rely on .text and .metadata."""
    text: str
    metadata: Dict[str, Any]


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
    embed_model=None
) -> Optional[VectorStoreIndex]:
    """
    Load existing index (Chroma-backed). Returns None if not present.

    Args:
        persist_dir: Path to ChromaDB directory
        collection_name: Chroma collection name
        embed_model: Embedding model (defaults to Settings.embed_model)
    """
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        print(f"⚠️ No index found at {persist_dir}")
        return None

    try:
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_collection(collection_name)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model or Settings.embed_model
        )

        print(f"✅ Loaded index '{collection_name}' from {persist_dir}")
        return index

    except Exception as e:
        print(f"⚠️ Failed to load index '{collection_name}' from {persist_dir}: {e}")
        return None


def create_multimodal_query_engine(
    text_index: Optional[VectorStoreIndex],
    audio_index: Optional[VectorStoreIndex],
    image_index: Optional[VectorStoreIndex],
    similarity_top_k: int = TOP_K,
    similarity_cutoff: float = SIMILARITY_THRESHOLD
):
    """
    Create a multimodal retriever that searches across all modalities.
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

    IMPORTANT: We return a list[SimpleDoc] so downstream code can rely on
    .text and .metadata (doc_id, chunk_id, source_type).
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

        postprocessor = SimilarityPostprocessor(
            similarity_cutoff=self.similarity_cutoff
        )

        return {"retriever": retriever, "postprocessor": postprocessor, "source": source_type}

    # -------- helpers to normalize NodeWithScore to SimpleDoc --------
    def _extract_text_and_metadata(self, node_like, fallback_source: str, idx: int) -> SimpleDoc:
        """
        Handle both NodeWithScore and underlying Node/TextNode.
        Guarantee text and metadata exist, with sane defaults.
        """
        # Try to get underlying node if it's NodeWithScore
        base = getattr(node_like, "node", node_like)

        # Text
        text = getattr(base, "text", None)
        if not text:
            # Some nodes expose get_content()
            get_content = getattr(base, "get_content", None)
            text = get_content() if callable(get_content) else ""
        if not isinstance(text, str):
            text = str(text or "")

        # Metadata
        md = {}
        meta_attr = getattr(base, "metadata", None)
        if isinstance(meta_attr, dict):
            md = dict(meta_attr)
        else:
            # Some versions expose .metadata as Metadata class with .to_dict()
            to_dict = getattr(meta_attr, "to_dict", None)
            if callable(to_dict):
                md = to_dict()
        if not isinstance(md, dict):
            md = {}

        # Ensure source_type and ids
        md.setdefault("source_type", fallback_source)

        # doc_id fallback
        if "doc_id" not in md:
            md["doc_id"] = (
                md.get("source")
                or md.get("file_name")
                or getattr(base, "doc_id", None)
                or getattr(node_like, "id_", None)
                or f"{fallback_source.lower()}_doc"
            )

        # chunk_id fallback
        if "chunk_id" not in md:
            md["chunk_id"] = (
                getattr(base, "id_", None)
                or getattr(node_like, "id_", None)
                or f"{md['doc_id']}::{idx}"
            )

        return SimpleDoc(text=text, metadata=md)

    def retrieve(self, query: str) -> List[SimpleDoc]:
        """
        Retrieve documents from all modalities and fuse using RRF.
        Returns: list[SimpleDoc]
        """
        all_results: List[tuple[str, List[Any]]] = []

        # Retrieve from each modality
        for retriever_config in [self.text_retriever, self.audio_retriever, self.image_retriever]:
            if retriever_config is None:
                continue

            try:
                retriever = retriever_config["retriever"]
                postprocessor = retriever_config["postprocessor"]
                source_type = retriever_config["source"]

                nodes = retriever.retrieve(query)  # list[NodeWithScore]
                filtered_nodes = postprocessor.postprocess_nodes(nodes)  # same type

                # Convert each to SimpleDoc *now* (so downstream is uniform)
                simple_docs: List[SimpleDoc] = []
                for i, n in enumerate(filtered_nodes):
                    simple_docs.append(self._extract_text_and_metadata(n, source_type, i))

                all_results.append((source_type, simple_docs))

            except Exception as e:
                print(f"⚠️ Retrieval failed for {retriever_config['source']}: {e}")

        # Fuse results using Reciprocal Rank Fusion
        fused_docs = self._reciprocal_rank_fusion(all_results)
        return fused_docs

    def _reciprocal_rank_fusion(self, results: List[tuple], k: int = 60) -> List[SimpleDoc]:
        """
        Combine results from multiple retrievers using RRF.

        Args:
            results: List of (source_name, [SimpleDoc]) tuples
            k: RRF constant (default 60)
        """
        combined: Dict[str, Dict[str, Any]] = {}

        for source_name, docs in results:
            for rank, sd in enumerate(docs, start=1):
                rr_score = 1.0 / (rank + k)

                # Use normalized text as key for deduplication
                key = sd.text.strip()
                if not key:
                    # If empty, dedupe by doc_id+chunk_id
                    key = f"{sd.metadata.get('doc_id','unknown')}::{sd.metadata.get('chunk_id','0')}"

                if key not in combined:
                    combined[key] = {"doc": sd, "score": 0.0}

                combined[key]["score"] += rr_score

        # Sort by score
        sorted_items = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

        # Return top-k docs
        return [item["doc"] for item in sorted_items[: self.similarity_top_k]]

    # (Optional) kept for API completeness; unused in current flow
    def as_query_engine(self):
        """
        If you ever need a QueryEngine wrapper around this retriever,
        you can implement a tiny adapter that returns final text.
        """
        # Not used in the current code path
        return self
