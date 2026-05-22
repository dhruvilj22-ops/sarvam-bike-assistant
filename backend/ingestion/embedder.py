"""
Embedding: adds vector to each chunk using OpenAI text-embedding-3-small.
Mock returns a fixed deterministic vector so the pipeline runs without API calls.
"""
import os
from typing import Dict, List

import numpy as np

EMBEDDING_DIM = 1536  # text-embedding-3-small output dimension
_MOCK_VECTOR = list(np.random.default_rng(42).standard_normal(EMBEDDING_DIM).astype(float))

_BATCH_SIZE = 100


def _embed_real(texts: List[str]) -> List[List[float]]:
    # OpenAI text-embedding-3-small — cost-effective and strong on technical text
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vectors = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def embed_chunks(chunks: List[Dict], use_mocks: bool = False) -> List[Dict]:
    """Add 'vector' field to each chunk. Returns the same list (mutated in place)."""
    if use_mocks:
        for chunk in chunks:
            chunk["vector"] = _MOCK_VECTOR
        return chunks

    texts = [c["text"] for c in chunks]
    vectors = _embed_real(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk["vector"] = vec
    return chunks
