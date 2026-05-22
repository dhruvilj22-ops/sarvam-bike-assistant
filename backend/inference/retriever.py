"""
Hybrid retrieval: Qdrant vector search + BM25, merged via Reciprocal Rank Fusion.
Intent adjusts the vec_k / bm25_k split before merging.
"""
import os
from typing import Dict, List, Tuple

from ingestion.indexer import bm25_search, vector_search

_RRF_K = 60


def _rrf_merge(
    vec_results: List[Tuple[Dict, float]],
    bm25_results: List[Tuple[Dict, float]],
) -> List[Tuple[Dict, float]]:
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    for rank, (chunk, _) in enumerate(vec_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_map[cid] = chunk

    # Only merge BM25 results with positive scores
    for rank, (chunk, bm25_score) in enumerate(bm25_results):
        if bm25_score <= 0:
            continue
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_map[cid] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(chunk_map[cid], scores[cid]) for cid in sorted_ids]


def retrieve(
    query: str,
    document_id: str,
    intent: str = "diagnostic",
    top_k: int = 7,
    use_mocks: bool = False,
) -> List[Tuple[Dict, float]]:
    """
    Return top_k (chunk, rrf_score) pairs merged from vector + BM25.
    """
    if intent == "specification":
        vec_k, bm25_k = 4, 10
    elif intent == "procedure":
        vec_k, bm25_k = 10, 4
    elif intent == "diagnostic":
        vec_k, bm25_k = 9, 5
    else:
        vec_k, bm25_k = 7, 7

    if use_mocks:
        from ingestion.embedder import _MOCK_VECTOR
        query_vector = _MOCK_VECTOR
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
        query_vector = resp.data[0].embedding

    vec_results = vector_search(query_vector, document_id, top_k=vec_k)
    bm25_results = bm25_search(query.lower().split(), document_id, top_k=bm25_k)

    merged = _rrf_merge(vec_results, bm25_results)
    return merged[:top_k]
