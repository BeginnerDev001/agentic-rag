"""
Vector store wrapper using ChromaDB for storing and searching SEC filing chunks.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import chromadb
from chromadb.config import Settings


DEFAULT_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))
DEFAULT_COLLECTION_NAME = "sec_10k_chunks"


class VectorStore:
    """
    ChromaDB persistent vector store manager for RAG chunk retrieval.

    Handles:
    - Collection creation / loading with HNSW cosine similarity space
    - Batch insertion of chunks and embeddings
    - Metadata sanitisation (ChromaDB requires str/int/float/bool primitive metadata values)
    - Metadata filtering (e.g. filter by ticker, fiscal_year, section_id)
    - Similarity query returning structured search results
    """

    def __init__(
        self,
        persist_dir: Union[str, Path] = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_metric: str = "cosine",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

    def count(self) -> int:
        """Return total number of chunks currently stored in the collection."""
        return self.collection.count()

    def sanitize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
        """
        Clean metadata dict to comply with ChromaDB constraints:
        - Only str, int, float, bool allowed as metadata values.
        - None values or complex types are converted/omitted.
        """
        sanitized = {}
        for k, v in meta.items():
            if k in ("text", "chunk_id"):
                continue  # 'text' is stored as document, 'chunk_id' as id
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        batch_size: int = 500,
    ) -> int:
        """
        Add chunks and their corresponding embedding vectors to ChromaDB.

        Args:
            chunks: List of chunk dictionaries (deserialised from JSONL).
            embeddings: List of embedding vectors matching chunks.
            batch_size: Batch size for ChromaDB upsert calls.

        Returns:
            Number of chunks successfully added.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        if not chunks:
            return 0

        total_added = 0
        n = len(chunks)

        for i in range(0, n, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeds = embeddings[i : i + batch_size]

            # Deduplicate within batch to prevent DuplicateIDError
            seen_ids = set()
            unique_chunks = []
            unique_embeds = []
            for c, emb in zip(batch_chunks, batch_embeds):
                cid = c["chunk_id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    unique_chunks.append(c)
                    unique_embeds.append(emb)

            ids = [c["chunk_id"] for c in unique_chunks]
            documents = [c["text"] for c in unique_chunks]
            metadatas = [self.sanitize_metadata(c) for c in unique_chunks]

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=unique_embeds,
                metadatas=metadatas,
            )
            total_added += len(unique_chunks)

        return total_added

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using a query vector.

        Args:
            query_embedding: Vector embedding of the search query.
            n_results: Top k results to retrieve.
            where_filter: ChromaDB metadata filter dict (e.g. {"ticker": "AAPL"}).

        Returns:
            List of result dicts, each containing:
            - chunk_id
            - text
            - score (cosine similarity score)
            - metadata
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, max(1, self.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        output: List[Dict[str, Any]] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return output

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i in range(len(ids)):
            dist = distances[i] if i < len(distances) else 0.0
            # For cosine distance in Chroma: score = 1 - distance
            similarity_score = 1.0 - dist if self.distance_metric == "cosine" else -dist

            res = {
                "chunk_id": ids[i],
                "text": documents[i] if i < len(documents) else "",
                "score": float(similarity_score),
                "distance": float(dist),
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }
            output.append(res)

        return output

    def clear(self) -> None:
        """Clear all contents of the current collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )
