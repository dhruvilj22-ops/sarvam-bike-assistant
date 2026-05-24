"""
Hybrid retrieval: Qdrant vector search + BM25, merged via Reciprocal Rank Fusion.
Intent adjusts the vec_k / bm25_k split before merging.
"""
import os
import logging
from typing import Dict, List, Tuple
import httpx

from ingestion.indexer import bm25_search, vector_search

_RRF_K = 60
logger = logging.getLogger(__name__)


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
    default_vec_k = int(os.getenv("RAG_VEC_K", "20"))
    default_bm25_k = int(os.getenv("RAG_BM25_K", "20"))
    if intent == "specification":
        vec_k, bm25_k = max(8, default_vec_k // 2), max(default_bm25_k, default_vec_k)
    elif intent == "procedure":
        vec_k, bm25_k = max(default_vec_k, default_bm25_k // 2), max(8, default_bm25_k // 2)
    elif intent == "diagnostic":
        vec_k, bm25_k = default_vec_k, max(10, default_bm25_k // 2)
    else:
        vec_k, bm25_k = default_vec_k, default_bm25_k

    if use_mocks:
        from ingestion.embedder import _MOCK_VECTOR
        query_vector = _MOCK_VECTOR
    else:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=httpx.Client(trust_env=False, timeout=60.0),
        )
        resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
        query_vector = resp.data[0].embedding

    vec_results = vector_search(query_vector, document_id, top_k=vec_k)
    bm25_results = bm25_search(query.lower().split(), document_id, top_k=bm25_k)

    merged = _rrf_merge(vec_results, bm25_results)
    safe_top_k = int(os.getenv("RAG_TOP_K", str(top_k)))
    logger.info(
        "retrieve_summary document_id=%s intent=%s vec_k=%s bm25_k=%s vec_hits=%s bm25_hits=%s merged_hits=%s return_top_k=%s",
        document_id,
        intent,
        vec_k,
        bm25_k,
        len(vec_results),
        len(bm25_results),
        len(merged),
        safe_top_k,
    )
    for i, (chunk, score) in enumerate(merged[:3], start=1):
        logger.info(
            "retrieve_top rank=%s chunk_id=%s page=%s section=%s score=%.6f",
            i,
            chunk.get("chunk_id", ""),
            chunk.get("page_number", 0),
            chunk.get("section_number", ""),
            float(score),
        )
    return merged[:safe_top_k]
