"""Manual verification script for EmbeddingModel."""
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.embeddings import EmbeddingModel

def main():
    print("Initializing EmbeddingModel...")
    embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    print(f"Model name: {embedder.model_name}")
    print(f"Dimension: {embedder.dimension}")

    sample_chunks = [
        "Apple Inc. total net sales for fiscal year 2025 were $391.0 billion.",
        "Microsoft Corporation Intelligent Cloud revenue was $96.0 billion in FY2024.",
        "Tesla Inc. delivered 1.79 million vehicles in 2024.",
    ]

    print("\nEmbedding sample chunks...")
    vecs = embedder.embed_texts(sample_chunks)
    print(f"Generated {len(vecs)} vectors, each of length {len(vecs[0])}")

    query = "What were Apple sales in 2025?"
    print(f"\nEmbedding query: '{query}'")
    q_vec = embedder.embed_query(query)

    q_arr = np.array(q_vec)
    print("\nCosine Similarities:")
    for text, v in zip(sample_chunks, vecs):
        sim = np.dot(q_arr, np.array(v))
        print(f"  Similarity {sim:.4f} -> '{text}'")

if __name__ == "__main__":
    main()
